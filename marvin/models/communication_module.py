'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This contains a variety of models that can be used for the communication
protocol. All return a tensor of a specific shape.
'''

import random

import torch
from torch import nn
from torch.nn.parameter import Parameter

from marvin.models.general import device

random.seed(0)


class AverageComms(nn.Module):
    """
    Module that simply take the average of the communication
    channels like in commnet
    """
    def __init__(self, args):
        super(AverageComms, self).__init__()
        self.args = args

    def reset(self):
        pass

    def forward(self, input_comms, index=None):
        return torch.stack(input_comms, 0).mean(0).squeeze(0)


class MaxComms(nn.Module):
    """
    Module that simply take the maximum of the communication channels,
    mimicing a form of maxpooling.
    """
    def __init__(self, args):
        super(MaxComms, self).__init__()
        self.args = args

    def forward(self, input_comms, index=None):
        return torch.stack(input_comms, 0).max(0)[0].squeeze(0)


class AttComms(nn.Module):
    """Our single headed attention based communcation module. This module
    exchanges information between agents based on how compatible they are
    as determined by the attention vectors.
    """
    def __init__(self, args):
        super(AttComms, self).__init__()
        self.args = args
        self.v = nn.Linear(self.args.comm_channels, self.args.comm_channels)
        self.q = nn.Linear(self.args.comm_channels, self.args.comm_channels)
        self.k = nn.Linear(self.args.comm_channels, self.args.comm_channels)

        self.gamma = Parameter(torch.tensor(1.0).float())
        self.beta = Parameter(torch.tensor(0.0).float())

    def forward(self, input_comms, index=None):
        x = torch.cat(input_comms, 0)  # (num agents, num nodes, comm features)
        # ipdb.set_trace()
        v = self.v(x)  # (num agents, num nodes, comm features)
        q = self.q(x)  # (num agents, num nodes, comm features)
        k = self.k(x[index])  # (num nodes, comm features)

        att = torch.einsum("ijk,jk->ij", q, k)  # (num agents, num nodes)
        att = torch.softmax(att, 0)  # (num agents, num_nodes)

        features = torch.einsum('ij,ijk->jk', att, v)

        return features


class NoDirectComms(nn.Module):
    """
    Module for mimicking if there were no communication channels.
    We use a seperate class since 0 comm channels can result in things
    breaking in the other protocols.
    """
    def __init__(self, args):
        super(NoDirectComms, self).__init__()
        self.args = args

    def forward(self, input_comms, index=None):
        return torch.zeros((self.args.num_nodes, self.args.comm_channels),
                           device=device)
