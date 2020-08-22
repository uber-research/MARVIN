'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This contains generic helper modules that are used by the main networks.
'''

import random

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from torch import nn

from marvin.utils.utils import optimized

random.seed(0)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProcessedGraph:
    """The processed graph with all the different forms of the
    modified adjacency matrix.
    """
    def __init__(self, adj):
        """Creates all of the preprocessed version of the graph and
        saves them in one object

        Arguments:
            adj {np.ndarray} -- (n, n) adjacency matrix
        """
        self.points = None
        self.speeds = None

        max_weight = adj.max()
        adj = adj / max_weight
        self.adj = adj

        # Not including weights
        self.transition_matrix = (self.adj > 0).float()

        dense = adj.cpu().numpy()

        if adj.shape[0] > 100:
            sparse = csr_matrix(dense.astype(np.float))
            dist_matrix = floyd_warshall(csgraph=sparse, directed=True,
                                         return_predecessors=False)
            pred = floyd_warshall(csgraph=sparse.transpose(), directed=True,
                                  return_predecessors=True)[1].transpose()
        else:
            dist_matrix = optimized.fw(dense.astype(np.float))
            pred = optimized.fw_pred(dense.astype(np.float))

        self.pred = pred
        dense = torch.tensor(dist_matrix, device=device)
        self.dense = dense.float()
        self.actual_distance = self.dense * max_weight
        self.dense_norm = dense.float().clone()
        self.dense_norm = (self.dense_norm - self.dense_norm.mean()) \
            / (1e-5 + self.dense_norm.std())


class MaskedSoftmax(nn.Module):
    """Creates a masked softmax object"""
    def __init__(self):
        super(MaskedSoftmax, self).__init__()

    @classmethod
    def forward(self, x, mask=None):
        """Performs the softmax operation given a mask

        Arguments:
            x {torch.Tensor} -- Tensor where the final dimension is softmaxed

        Keyword Arguments:
            mask {torch.Tensor} -- Mask tensor of shape (..., number of nodes)
                (default: {None})

        Returns:
            torch.Tensor -- Tensor of the same input shape
        """
        if mask is not None:
            mask = mask > 0
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(0)[0]
        x_exp = (x_masked - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()

        return x_exp / x_exp.sum(0).unsqueeze(-1)
