'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This section contains the various variations of our MARVIN model that can be used.
'''

import math
import random

import torch
from torch import nn, relu, softmax, tanh
from torch.autograd import Variable

from marvin.models.communication_module import (AttComms, AverageComms,
                                                MaxComms, NoDirectComms)
from marvin.models.general import MaskedSoftmax, device

random.seed(0)


class LearnedDistance(nn.Module):
    def __init__(self, dh, dk, FF):
        """The learned distance exchange matrix that computes information exchange

        Arguments:
            dh {int} -- input tensor size
            dk {int} -- key tensor size
            FF {int} -- Feed forward size
        """
        super(LearnedDistance, self).__init__()
        self.v = nn.Linear(dh, dh)
        self.q = nn.Linear(dh, dk)
        self.k = nn.Linear(dh, dk)
        self.dist_w1 = nn.Linear(3, FF)
        self.dist_w2 = nn.Linear(FF, FF)
        self.dist_w3 = nn.Linear(FF, 1)

    def forward(self, x, adj, dense):
        """
        Arguments:
            x {torch.Tensor} -- input tensor
            adj {torch.Tensor} -- adjacency matrix (sparse version)
                (n, n) shape
            dense {torch.Tensor} -- dense distance matrix
                (n, n) shape

        Returns:
            torch.Tensor -- output tensor
                (n, dh) shape
        """

        x = x.repeat(x.shape[1], 1, 1)
        y = x.transpose(0, 1)

        q = self.q(x)  # (n, n, dk)
        k = self.k(y)  # (n, n, dk)
        v = self.v(x)  # (n, n, dh)

        out = torch.einsum('ijk,ijk->ij', q, k)  # (n, n)
        focus = relu(self.dist_w1(torch.stack(
            (out, adj, dense), 2)))  # (n, n, FF)
        focus = relu(self.dist_w2(focus))
        focus = softmax(self.dist_w3(focus).squeeze(2), 1)  # (n, n)

        feature = torch.einsum('ijk,ij->ik', v, focus)  # (n, dh)
        return feature.unsqueeze(0)


class NoLSTM(nn.Module):
    """
    MARVIN without the LSTM module
    """

    def __init__(self, args):
        super(NoLSTM, self).__init__()
        self.args = args

        self.softmax = MaskedSoftmax()

        self.encoding = nn.Linear(
            args.num_inputs + args.comm_channels, args.ch_q)
        self.decoding = nn.Linear(args.ch_q, 1)
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

        # we define the communication protocol to be used
        if self.args.comm_channels == 0:
            self.comm_protocol = NoDirectComms(self.args)
        elif self.args.max_comms:
            self.comm_protocol = MaxComms(self.args)
        elif self.args.avg_comms:
            self.comm_protocol = AverageComms(self.args)
        else:
            self.comm_protocol = AttComms(self.args)

        self.gcnenc = LearnedDistance(
            self.args.ch_q, self.args.ch_q, self.args.ch_q)

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
        return graph.adj.float()

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

        input_features = torch.cat([x, comms], -1)  # (n, input + comm)
        x = self.encoding(input_features).unsqueeze(0)  # (n, q)

        transition = self.transition_matrix(graph)  # (n, n)
        dense = self.dense_matrix(graph)  # (n, n)

        for _ in range(self.args.k):
            x1 = self.gcnenc(x, transition, dense)  # (n, q)
            x = x1 + x

        comms = tanh(self.comm_decoding(x))  # (n, comms)

        out = self.decoding(x).squeeze()  # (n,)
        s = self.final_feature_process(out, graph, pos, mask)  # (n,)

        self.comms[index] = comms

        return s


class MARVIN(NoLSTM):
    """
    Our main model
    """

    def __init__(self, args):
        super(MARVIN, self).__init__(args)

        self.lstm = nn.LSTM(args.ch_q, args.ch_q, 2)
        self.lstm.flatten_parameters()

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

        input_features = torch.cat([x, comms], -1)
        x = self.encoding(input_features).unsqueeze(0)  # (n, input + comm)

        transition = self.transition_matrix(graph)  # (n, n)
        dense = self.dense_matrix(graph)  # (n, n)

        x1 = self.gcnenc(x, transition, dense)  # (n, q)
        x1, h = self.lstm(x1)
        x = x1 + x  # (n, q)

        for _ in range(1, self.args.k):
            x1 = self.gcnenc(x, transition, dense)
            x1, h = self.lstm(x1, h)
            x = x1 + x  # (n, q)

        comms = tanh(self.comm_decoding(x))  # (n, comm)

        out = self.decoding(x).squeeze()
        s = self.final_feature_process(out, mask)  # (n,)

        self.comms[index] = comms

        return s
