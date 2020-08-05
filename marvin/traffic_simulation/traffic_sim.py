'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
The file provides a function that can simulate traffic one timestep into
the future using convex optimization. This helps to simulate how traffic eventually
builds up.
'''

import copy
import random

import cvxpy as cp
import numpy as np

random.seed(0)

# run using this paper as a basis:
# https://journals-scholarsportal-info.myaccess.library.utoronto.ca/pdf/01912615/v45i0001/289_agcofodmsotf.xml
# A generic class of first order node models for dynamic macroscopic
# simulation of traffic flows

# using
# https://onlinelibrary-wiley-com.myaccess.library.utoronto.ca/doi/epdf/10.1111/j.1467-8659.2009.01613.x
# Continuum Traffic Simulation
# for speed formula u = u_max (1 - rho ^ l)


def perform_timestep(transition, capacities, supply,
                     output_cap, damping=0.1):
    """Performs one simulation timestep into the future.

    Arguments:
        transition {np.ndarray} -- The amount of traffic that travels from
            one node to the next. This describes the fixed ratios that travel
            from each node to all the output nodes (n, n)
        capacities {np.ndarray} -- Maximum congestion that can exist at each
            node (we generally set this to be 1) (n,)
        supply {np.ndarray} -- Total traffic congestion at each node (n,)
        output_cap {nd.ndarray} -- maximum amount the nodes can output in
            a single timestep (n,)

    Keyword Arguments:
        damping {np.ndarray} -- factor to decrease the traffic by
            to help smooth out traffic (default: {0.1})

    Returns:
        Tuple[np.ndarray, np.ndarray] --
            1: the new congestion amount at each node (n,)
            2: the amount of traffic travelling over each each during
                this timestep (n, n)
    """

    vehicle_flow = cp.Variable(capacities.shape)
    transition_matrix = copy.deepcopy(transition)

    diagonal = np.diag(transition_matrix.sum(0))
    for i in range(diagonal.shape[0]):
        if diagonal[i, i] > 0:
            diagonal[i, i] = 1 / diagonal[i, i]

    transition_matrix = transition_matrix @ diagonal
    transition_matrix = np.clip(transition_matrix, 1e-5, 1)

    ones = np.ones(capacities.shape)

    # we want to maximize the ammount of congestion flowing given the
    # constraints and given the movement capacity of each node
    prob = cp.Problem(cp.Maximize(ones.T @ vehicle_flow),
                      [vehicle_flow <= damping * output_cap,
                       vehicle_flow <= supply,
                       transition_matrix @ vehicle_flow <= capacities - supply,
                       transition_matrix @ vehicle_flow <= damping * output_cap,
                       vehicle_flow >= 0, transition_matrix @ vehicle_flow >= 0])

    prob.solve(solver='SCS')
    if vehicle_flow.value is None:
        # Solver failed, so we return the original congestions
        return supply, np.zeros(transition_matrix.shape)

    vehicle_flow = np.clip(vehicle_flow.value, 0, 1)

    supply = np.clip(supply - vehicle_flow + transition_matrix @ vehicle_flow, 0, 1)

    return supply, np.multiply(transition_matrix, np.tile(vehicle_flow[None, :], (transition_matrix.shape[0], 1)))
