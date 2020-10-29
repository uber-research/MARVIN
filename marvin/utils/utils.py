'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This file contains some generic utility functions, and the means through which
the trainers can use the LKH optimal solver.
'''

import copy
import os
import re
from ctypes import c_double, c_int, cdll
from random import randint, seed

import numpy as np
from numpy.ctypeslib import ndpointer

seed(0)


def calc_cost(cost_mat, route):
    """Calculate the tour cost of the route

    Arguments:
        cost_mat {np.ndarray} -- cost matrix
        route {List[int]} -- route to calculate the cost of

    Returns:
        float -- total cost of the traversal
    """
    return cost_mat[np.roll(route, 1), route].sum()


def _create_mapping(adj, pass_count, same_start):
    """We duplicate nodes that are visited multiple times. This maps back to
    the indices of the original nodes after the solution has been found

    Arguments:
        adj {np.ndarray} -- adjacency matrix
        pass_count {List[intt]} -- number of times each node must be passed
        same_start {bool} -- if the agents start at the same point

    Returns:
        Tuple[np.ndarray -- modified adjacency matrix
            List[int] -- mapping that maps the duplicated nodes to their
            original indices

    """

    mapping = [i for i in range(len(pass_count))]

    for i in range(len(pass_count)):
        for _ in range(1, int(pass_count[i])):
            adj = np.concatenate(
                (np.copy(adj), np.copy(adj[i][None, :])), 0)
            adj = np.concatenate(
                (np.copy(adj), np.copy(adj[:, i][:, None])), 1)
            mapping.append(i)

    return adj, mapping


def _modify_adj(adj, starting_poses, same_start, is_tour):
    """Modifies the adjacency matrix as required for the problem

    Arguments:
        adj {np.ndarray} -- (n, n) dimensional numpy adjacency matrix
        starting_poses {List[int]} -- original positions of each of the agents
        same_start {bool} -- if the agents start at the same node
        is_tour {bool} -- if we want the matrix to include the return tour cost

    Returns:
        np.ndarray --  modified adjacency matrix that
    """

    if same_start and not is_tour:
        # we allow the agents to return to start at no cost
        start = starting_poses[0]
        adj[[0, start]] = adj[[start, 0]]
        adj = adj.transpose()
        adj[[0, start]] = adj[[start, 0]]
        adj = adj.transpose()
        adj[:, 0] = 0
    elif same_start and is_tour:
        # swap our starting point with the first node
        start = starting_poses[0]
        adj[[0, start]] = adj[[start, 0]]
        adj = adj.transpose()
        adj[[0, start]] = adj[[start, 0]]
        adj = adj.transpose()
    elif not same_start and is_tour:
        # We add a new starting node that can go to the actual starting nodes
        # at no cost. We then make the cost to return to the starting nodes
        # the minimum distance between each node and all possible return depots
        # This means the agents are assumed to be able to return to any depot
        start = np.array(
            [[2000 for _ in range(adj.shape[0])]])
        for s in starting_poses:
            start[0, s] = 0.0

        temp_start = [s+1 for s in starting_poses]

        adj = np.concatenate((start, adj), 0)
        ret = adj[temp_start].min(0, keepdims=True)
        ret = np.concatenate((np.zeros((1, 1)), ret), 1).transpose()
        adj = np.concatenate((ret, adj), 1)

    elif not same_start and not is_tour:
        # add a new starting node that has zero cost to travel to the original
        # starting nodes and zero cost to return to this node from any node

        start = np.array(
            [[2000 for _ in range(adj.shape[0])]])
        for s in starting_poses:
            start[0, s] = 0.01

        adj = np.concatenate((start, adj), 0)
        ret = np.zeros((1, adj.shape[0])).transpose()
        adj = np.concatenate((ret, adj), 1)

    # in the standard case do nothing since this is what is expected by default

    for i in range(len(adj)):
        adj[i, i] = 2000

    return adj


def _parse_multi_agents(output_file, largest, mapping, same_start,
                        is_tour, num_agents, starting_poses):
    """Parses in lines of a file for the tour of a set of agents and the cost
    of that tour

    Arguments:
        output_file {List[str]} -- lines of the processed output file
        largest {float} -- largest value in the matrix for cost calculation
        mapping {List[int]} -- the mapping back to the original indices
        same_start {bool} -- if the agents start at the same node
        is_tour {bool} -- if we want the matrix to include the return tour cost
        num_agents {int} -- number of agents
        starting_poses -- starting positions of the agents

    Returns:
        Tuple[costs, tours] --
            costs: List[int] -- the costs of each agent
            tours: List[List[int]] -- indices of the nodes in each tour
    """

    tours = []
    costs = []

    for i in range(num_agents):
        numbers = re.findall('\d+', output_file[2 + i])
        numbers = list(map(int, numbers))

        if same_start:
            steps = [n - 1 for n in numbers[:-3]]
            for s in range(len(steps)):
                if steps[s] == 0:
                    steps[s] = starting_poses[0]
                elif steps[s] == starting_poses[0]:
                    steps[s] = 0
                steps[s] = mapping[steps[s]]
        else:
            steps = [mapping[n - 2] for n in numbers[1:-3]]

        tours.append(steps)

        costs.append(numbers[-1] / 200. * largest)

    return costs, tours


def _parse_single_agents(output_file, largest, mapping,
                         same_start, is_tour, starting_poses):
    """Parses file that has the solution to a single agent tsp traversal

    Arguments:
        output_file {List[str]} -- lines of the processed output file
        largest {float} -- largest value in the matrix for cost calculation
        mapping {List[int]} -- the mapping back to the original indices
        is_tour {bool} -- if we want the matrix to include the return tour cost
        same_start {bool} -- if the agents start at the same node
        starting_poses -- starting positions of the agents

    Returns:
        Tuple[costs, tours] --
            costs: List[int] -- the costs of each agent
            tours: List[List[int]] -- indices of the nodes in each tour
    """
    numbers = re.findall('\d+', output_file[1])
    numbers = map(int, numbers)
    distance = max(numbers) / 200. * largest

    if is_tour:
        index = 6
    else:
        index = 7
    tour = []

    while int(output_file[index]) != -1:
        if same_start:
            tour.append(mapping[int(output_file[index]) - 1])
        else:
            tour.append(mapping[int(output_file[index]) - 2])
        index += 1

    if not is_tour and same_start:
        for s in range(len(tour)):
            if tour[s] == 0:
                tour[s] = starting_poses[0]
            elif tour[s] == starting_poses[0]:
                tour[s] = 0

    return [distance], [tour]


class optimized:
    """General library of optimized utility functions
    """

    lib = cdll.LoadLibrary('marvin/utils/optimized.so')
    LK_path = "marvin/utils/"

    @classmethod
    def fw(cls, adj):
        """This function automatically calculates the dense distance matrix from
        the sparse adjacency matrix. We code our own c++ version of the floyd
        warshal algorithm to do this so that the distance from each node to
        itself is also calculated.

        Arguments:
            adj {np.ndarray} -- (n, n) dimensional numpy adjacency matrix

        Returns:
            np.ndarray -- (n, n) dimensional numpy adjacency matrix
        """

        adj = np.copy(adj.astype(np.float))
        size = adj.shape[0]
        matrix_type = ndpointer(dtype=c_double, shape=adj.shape)

        mat = adj.ctypes.data_as(matrix_type)
        size = c_int(size)
        cls.lib.fw_run.restype = matrix_type
        return cls.lib.fw_run(mat, size)

    @classmethod
    def fw_pred(cls, adj):
        """
        Calculates the a shortest path next step matrix, where the output
        of adj[i,j] is the next node in the shortest path between nodes
        i and j.

        Arguments:
            adj {np.ndarray} -- sparse adjacency matrix to find the
                shortest paths on

        Returns:
            np.ndarray -- (n, n) shortest path matrix
        """
        adj = np.copy(adj.astype(np.float))
        adj = np.copy(adj)
        size = adj.shape[0]
        matrix_type = ndpointer(dtype=c_double, shape=adj.shape)
        ret_type = ndpointer(dtype=c_int, shape=adj.shape)

        mat = adj.ctypes.data_as(matrix_type)
        size = c_int(size)
        cls.lib.fw_pred_run.restype = ret_type
        return cls.lib.fw_pred_run(mat, size)

    @classmethod
    def LKH(cls, adj_old, pass_count=None, k=5, num_agents=1,
            starting_poses=None, identifier='', is_tour=False):
        """Calculates the optimal path and the costs of the optimal paths
        using the LKH3 solver.

        Arguments:
            adj_old {np.ndarray} -- adjacency matrix

        Keyword Arguments:
            pass_count {[type]} -- number of passes each node requires to
                be completed
            k {int} -- number of actions that solver can do in one
                move(default: {3})
            num_agents {int} -- number of agents performing the
                traversal (default: {1})
            starting_poses {[type]} -- original nodes the agents start
                at (default: {None})
            identifier {str} -- file identifier that the LKH solver will
                use to ensure it's solution doesn't interfere with
                other solutions being calculated (default: {''})
            is_tour {bool} -- If we want the agents to return to their
                starting depots. (default: {False})

        Returns:
            Tuple[List[int], List[int]]
                1: List of the costs of performing each of the traversals
                for each of the agents
        """
        adj = np.copy(adj_old)

        # we assume if unspecified that each node must be visited exactly once
        if pass_count is None:
            pass_count = [1 for _ in range(adj.shape[0])]
        else:
            pass_count = copy.deepcopy(pass_count)

        if starting_poses is None:
            starting_poses = [randint(0, adj.shape[0] - 1)
                              for _ in range(num_agents)]

        for p in starting_poses:
            pass_count[p] = 1.0

        largest = adj.max()
        same_start = len(starting_poses) != len(set(starting_poses))

        adj, mapping = _create_mapping(adj, pass_count, same_start)

        # first transform the sparse adjacency matrix into the fully connected
        # distance matrix
        adj = cls.fw(adj)
        adj = _modify_adj(adj, starting_poses, same_start, is_tour)

        num_nodes = adj.shape[0]
        file_name = "{}output{}.atsp".format(cls.LK_path, identifier)
        f = open(file_name, 'w')

        # stringify the adjacency matrix
        adj_string = [[str(int(adj[i, j] / largest * 200))
                       for j in range(adj.shape[1])] for i in range(
                           adj.shape[0]
                        )]

        # write the necessary files for the solver
        f.write("NAME: {} \n \
        TYPE: ATSP\n \
        COMMENT: Asymmetric single agent test case \n \
        DIMENSION: {} \n \
        EDGE_WEIGHT_TYPE: EXPLICIT \n \
        EDGE_WEIGHT_FORMAT: FULL_MATRIX \n \
        EDGE_WEIGHT_SECTION \n\
        {} \n \
        \n EOF".format(file_name, num_nodes, "\n".join(
            [" ".join(a) for a in adj_string])
        ))
        f.close()

        f = open("{}basic{}.par".format(cls.LK_path, identifier), 'w')
        f.write("PROBLEM_FILE = {}output{}.atsp \n \
        MOVE_TYPE = {} \n \
        RUNS = 1 \n \
        MTSP_MIN_SIZE = {} \n \
        MTSP_MAX_SIZE = {} \n \
        SEED = 1 \n \
        SALESMEN = {} \n \
        TIME_LIMIT = 10 \n \
        MTSP_OBJECTIVE = MINSUM \n \
        OUTPUT_TOUR_FILE = {}output_single{}.txt \n \
        MTSP_SOLUTION_FILE = {}output{}.txt".format(
            cls.LK_path, identifier,
            k,
            num_nodes // num_agents,
            num_nodes // num_agents,
            num_agents,
            cls.LK_path,
            identifier,
            cls.LK_path,
            identifier)
        )

        f.close()
        os.system(
            #2>&1
            "{}LKH {}basic{}.par > /dev/null".format(
                cls.LK_path, cls.LK_path, identifier)
            )

        os.remove("{}basic{}.par".format(cls.LK_path, identifier))
        os.remove(file_name)

        if num_agents > 1:
            # multiagent
            output_file = open("{}output{}.txt".format(
                cls.LK_path, identifier), 'r').readlines()

            costs, tours = _parse_multi_agents(
                output_file, largest, mapping, same_start,
                is_tour, num_agents, starting_poses
            )

            os.remove("{}output{}.txt".format(
                cls.LK_path, identifier))
        else:
            # single agent
            output_file = open("{}output_single{}.txt".format(
                cls.LK_path, identifier), 'r').readlines()

            costs, tours = _parse_single_agents(
                output_file, largest, mapping, same_start,
                is_tour, starting_poses
            )

        os.remove("{}output_single{}.txt".format(
                cls.LK_path, identifier))

        if min(map(len, tours)) == 0:
            # solver failed, try again with different starting positions
            return cls.LKH(adj_old, pass_count, k, num_agents,
                           None, identifier, is_tour)
        else:
            # occasionally the solver fails and does not start the tour at the
            # correct location
            starting_error = any([tours[i][0] not in starting_poses
                                  for i in range(len(tours))])
            if starting_error:
                return cls.LKH(adj_old, pass_count, k, num_agents,
                               None, identifier, is_tour)

        return costs, tours
