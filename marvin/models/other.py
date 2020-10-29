'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This file contains all the alternative models that can be used
to try and solve our benchmark. Some of the models have been altered
to better accomodate the restrictions of our environment.
'''

import math
import random

import numpy as np
import torch
from torch import nn, tanh
from torch.autograd import Variable

from marvin.models.communication_module import AttComms
from marvin.models.general import MaskedSoftmax, device

random.seed(0)


class GVIN(nn.Module):
    """Our reimplementation of the Generalized Value Iteration Network
    as described in https://arxiv.org/abs/1706.02416. We only implement
    the neural network kernel since we assume all spatial information
    to be unknown.
    """
    def __init__(self, args):
        super(GVIN, self).__init__()
        self.args = args

        self.reward_encoding = nn.Linear(
            self.args.num_inputs + args.comm_channels, 1)
        self.action_decoding = nn.Linear(1, args.ch_q)

        self.adj = None
        self.embedder = nn.Linear(args.ch_q, 1)
        self.encoder = nn.Linear(self.args.num_inputs + args.comm_channels,
                                 args.ch_q)

        # The variable that hold the directional kernel
        self.k1 = None
        # The variable that holds the spatial kernel
        self.k2 = None
        # The variable that holds the embedded kernel
        self.k3 = None

        self.comm_decoding = nn.Linear(args.ch_q, args.comm_channels)

        # if we run with eval communication vector does not store grad
        if self.args.eval:
            self.comms = [torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device)
                        for _ in range(self.args.num_agents)]
        else:
            self.comms = [Variable(torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device))
                          for _ in range(self.args.num_agents)]

        self.comm_protocol = AttComms(self.args)

        self.msoftmax = MaskedSoftmax()

    def reset(self):
        """Simply reset the communication vector for a new graph"""

        if self.args.eval:
            self.comms = [torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device)
                          for _ in range(self.args.num_agents)]
        else:
            self.comms = [Variable(torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device))
                          for _ in range(self.args.num_agents)]

    def initialize_adj(self, adj):
        """Initialize the adjacency matrix kernel

        Arguments:
            adj {torch.Tensor} -- the adjacency matrix
        """
        if self.reset:
            adj = adj.float()

            a_norm = adj + torch.eye(adj.shape[0], device=device)
            d_inv = torch.diag(a_norm.sum(0))

            for i in range(adj.shape[0]):
                d_inv[i, i] = 1 / (d_inv[i, i] + np.finfo(np.float32).eps)

            k3 = torch.matmul(torch.matmul(d_inv ** 0.5, a_norm), d_inv ** 0.5)

            # This kernel now only holds the normalized GCN version of the
            # Adjacency matrix
            self.k3 = k3
            self.adj = adj

    def final_feature_process(self, out, mask):
        """Does the final masking and processing of the node values

        Arguments:
            out {torch.Tensor} -- (..., n, 1) tensor for the output
                node value
            mask {torch.Tensor} -- (..., n) mask tensor

        Returns:
            torch.Tensor -- (..., n) tensor of the squeezed softmaxed
                values
        """
        mask = mask.clone()

        if not self.args.rl:
            # We don't softmax for supervised so the loss function has
            # access to the raw values
            to_mask = torch.zeros(out.shape, device=device).float()
            to_mask[mask == 0] = -math.inf
            return out.squeeze() + to_mask
        else:
            # otherwise we use standard masked softmax
            return self.msoftmax(out.squeeze(), mask)

    def forward(self, x, graph, index, pos, mask=None):
        """
        Arguments:
            x {torch.Tensor} -- input tensor of all node features
            graph {ProcessedGraph} -- processed graph shapes
            index {int} -- index of the agent (to access the correct
                saved communication vector)
            pos {int} -- node index of the agent's current position

        Keyword Arguments:
            mask {torch.Tensor} -- which nodes to include (default: {None})

        Returns:
            torch.Tensor -- (n,) tensor of the value at each node
        """

        # if the mask is undefined, make sure no nodes are masked
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=device)

        comms = self.comm_protocol(self.comms, index).squeeze(0)  # (n, comm)
        self.initialize_adj(graph.adj.float())

        # Create the original rewards
        r = self.reward_encoding(torch.cat((x, comms), 1)).squeeze(1)  # (n,)
        enc = self.encoder(torch.cat((x, comms), 1))  # (n, ch_e)

        cols = enc.unsqueeze(0).repeat(enc.shape[0], 1, 1)  # (n, n, ch_e)
        rows = enc.unsqueeze(1).repeat(1, enc.shape[0], 1)  # (n, n, ch_e)

        diff = cols - rows  # (n, n, ch_e)

        # Calculate the actual kernel from the GCN matrix
        kernel3 = self.k3 * self.embedder(diff).squeeze(2)  # (n, n)

        # Initialize the original values to be zero
        v = Variable(torch.zeros(r.shape, device=device))

        for _ in range(self.args.k):
            k3 = torch.matmul(
                kernel3, r + self.args.gamma * v).unsqueeze(1)  # (n, 1)

            q = self.action_decoding(k3)  # (n, ch_q)
            v = torch.max(q, dim=-1)[0]  # (n,)

        comms = tanh(self.comm_decoding(q))
        self.comms[index] = comms.unsqueeze(0)

        return self.final_feature_process(v, mask)


class GATNet(nn.Module):
    """
    Graph Attention Encoder with our attention communication module
    """

    def __init__(self, args):
        super(GATNet, self).__init__()
        self.args = args

        self.encoding = nn.Linear(self.args.num_inputs + args.comm_channels,
                                  args.ch_q)
        self.decoding = nn.Linear(args.ch_q, 1)

        self.att_decoding = nn.ModuleList([nn.Linear(args.ch_q, args.ch_q)
                                           for _ in range(args.k)])
        self.att = nn.ModuleList([nn.Linear(args.ch_q * 2, 1)
                                  for _ in range(args.k)])
        self.att_vectors = nn.ModuleList([nn.Linear(args.ch_q, args.ch_q)
                                          for _ in range(args.k)])

        self.comm_decoding = nn.Linear(args.ch_q, args.comm_channels)

        # if we run with eval communication vector does not store grad
        if self.args.eval:
            self.comms = [torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device)
                          for _ in range(self.args.num_agents)]
        else:
            self.comms = [Variable(torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device))
                          for _ in range(self.args.num_agents)]

        self.comm_protocol = AttComms(self.args)

        self.softmax = MaskedSoftmax()

    def dense_matrix(self, graph):
        """Returns the attribute of the processed graph to represent the
        dense adjacency matrix.

        Arguments:
            graph {Processed Graph} -- graph object

        Returns:
            torch.Tensor -- (n, n) tensor that contains the desired matrix
        """
        return graph.dense_norm.float()

    def transition_matrix(self, graph):
        """Returns the attribute of the processed graph to represent the
        main transition adjacency matrix.

        Arguments:
            graph {Processed Graph} -- graph object

        Returns:
            torch.Tensor -- (n, n) tensor that contains the desired matrix
        """
        return graph.transition_matrix.float()

    def reset(self):
        """Simply reset the communication vector for a new graph"""

        if self.args.eval:
            self.comms = [torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device)
                          for _ in range(self.args.num_agents)]
        else:
            self.comms = [Variable(torch.zeros((
                1, self.args.num_nodes, self.args.comm_channels),
                    device=device))
                          for _ in range(self.args.num_agents)]

    def final_feature_process(self, out, mask):
        """Does the final masking and processing of the node values

        Arguments:
            out {torch.Tensor} -- (..., n, 1) tensor for the output
                node value
            mask {torch.Tensor} -- (..., n) mask tensor

        Returns:
            torch.Tensor -- (..., n) tensor of the squeezed softmaxed
                values
        """
        mask = mask.clone()

        if not self.args.rl:
            # We don't softmax for supervised so the loss function has
            # access to the raw values
            to_mask = torch.zeros(out.shape, device=device).float()
            to_mask[mask == 0] = -math.inf
            return out.squeeze() + to_mask
        else:
            # otherwise we use standard masked softmax
            return self.softmax(out.squeeze(), mask)

    def forward(self, x, graph, index, pos, mask=None):
        """
        Arguments:
            x {torch.Tensor} -- input tensor of all node features
            graph {ProcessedGraph} -- processed graph shapes
            index {int} -- index of the agent (to access the correct
                saved communication vector)
            pos {int} -- node index of the agent's current position

        Keyword Arguments:
            mask {torch.Tensor} -- which nodes to include (default: {None})

        Returns:
            torch.Tensor -- (n,) tensor of the value at each node
        """

        # if the mask is undefined, make sure no nodes are masked
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=device)

        comms = self.comm_protocol(self.comms, index)  # (n, comm)

        input_features = torch.cat([x, comms], -1)  # (n, comm + input)
        x = self.encoding(input_features)  # (n, q)

        trans = self.transition_matrix(graph).transpose(0, 1)  # (n, n)
        trans[trans == 0] = -math.inf

        for k in range(self.args.k):
            att = self.att_decoding[k](x)  # (n, q)
            att_t = att.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (n, q)
            cat_details = torch.cat((att_t, att_t.transpose(0, 1)), -1)

            focus = torch.softmax(
                self.att[k](cat_details).squeeze(-1) + trans, dim=1
            )  # (n, q)

            vectors = self.att_vectors[k](x)  # (n, q)
            x = torch.matmul(focus, vectors)

        x = x.unsqueeze(0)

        comms = tanh(self.comm_decoding(x))

        out = self.decoding(x).squeeze()  # (n,)

        s = self.final_feature_process(out, mask)
        self.comms[index] = comms

        return s
