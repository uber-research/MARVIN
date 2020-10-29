'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This is the file that has all of the input arguments the user can set when training
the model.
'''

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''The arguments will be pased to the main training loop through an object that
we will refer to as `args`'''

parser = argparse.ArgumentParser()
parser.add_argument('--num_nodes',
                    type=int,
                    default=16,
                    help='number of nodes in the graph (for internal use)')
parser.add_argument('--log_dir',
                    type=str,
                    default="logs",
                    help='number of nodes in the graph (for internal use)')
parser.add_argument('--ch_q',
                    type=int,
                    default=16,
                    help='dimension of the inner latent vector')
parser.add_argument('--k',
                    type=int,
                    default=5,
                    help='number of iterations taken in VIN')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--epochs',
                    type=int,
                    default=10000,
                    help='number of epochs to train for')
parser.add_argument('--batch_size',
                    type=int,
                    default=50,
                    help='number of graphs for each batch')
parser.add_argument('--tag',
                    type=str,
                    default='',
                    help='tag used to identify the logdir')
parser.add_argument('--load',
                    type=str,
                    default=None,
                    help='path to the model we want to load')
parser.add_argument('--num_agents',
                    type=int,
                    default=2,
                    help='number of agents')
parser.add_argument('--comm_channels',
                    type=int,
                    default=16,
                    help='number of communication channels')
parser.add_argument('--trainer',
                    type=str,
                    default='pg',
                    help='Which trainer to use when \
                        training (pg = policy gradient, ppo = ppo)')
parser.add_argument('--ppo_epochs',
                    type=int,
                    default=5,
                    help='How many epochs the ppo training should \
                        go on for before resetting the reference model')
parser.add_argument('--ppo_eps',
                    type=float,
                    default=0.2,
                    help='clip point for the ppo cutoff of the action ratio')
parser.add_argument('--eps',
                    type=float,
                    default=0.2,
                    help='epsilon for exploration')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='momentum value for optimizer')
parser.add_argument('--num_inputs',
                    type=int,
                    default=10,
                    help='Number of input node features')
parser.add_argument('--seed',
                    type=int,
                    default=-1,
                    help='Seed the state of the graph')
parser.add_argument('--gamma',
                    type=int,
                    default=0.7,
                    help='gamma discount for RL')
parser.add_argument('--pass_range',
                    type=int,
                    default=3,
                    help='maximum number of times an agent \
                        may have to revisit a node')
parser.add_argument('--model',
                    type=str,
                    default='vin',
                    help='which model to train with, can be one of: \n\
                        vin\n\
                        gvin\n\
                        nolstm\n\
                        gat\n')
parser.add_argument('--discount',
                    type=float,
                    default=0.0,
                    help='Proportion of the loss that should be \
                        based on the discounted reward over the episodic \
                        reward')
parser.add_argument('--max_grad_norm',
                    type=float,
                    default=1.0,
                    help='maximum norm of the gradient vector')
parser.add_argument('--min_size',
                    type=int,
                    default=10,
                    help='minimum size of the training/testing graphs')
parser.add_argument('--max_size',
                    type=int,
                    default=25,
                    help='maximum size of the training/testing graphs')
parser.add_argument('--max_clamp',
                    type=float,
                    default=10,
                    help='maximum ratio to allow for when training with ppo')
parser.add_argument('--decay_every',
                    type=int,
                    default=2000,
                    help='number of epochs before each decay')
parser.add_argument('--save_every',
                    type=int,
                    default=20,
                    help='number of epochs before each saving the model')
parser.add_argument('--val_every',
                    type=int,
                    default=20,
                    help='number of epochs between evaluation runs')
parser.add_argument('--SGD', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='If we want to use SGD optimizer')
parser.add_argument('--max_comms', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Gets the maximum of the comms')
parser.add_argument('--avg_comms', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Use basic commnet comms')
parser.add_argument('--asynch', type=str2bool, nargs='?',
                    const=True, default=True,
                    help='If the agents act asynchronously at test time')
parser.add_argument('--traffic', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='If we want to include realistic slowdowns due to \
                        traffic')
parser.add_argument('--gaussian_dist', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Set the number of passes each vehicle needs to take \
                        to occur according to a gaussian distribution')
parser.add_argument('--bimodal_dist', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Set the number of passes each vehicle needs to take \
                        to occur according to a bimodal (2-4) distribution')
parser.add_argument('--exp_dist', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Set the number of passes each vehicle needs to take \
                        to occur according to an exponential (2-4) \
                            distribution')
parser.add_argument('--rl', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='if we want to train with reinforcement learning')
parser.add_argument('--tune', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='whether to run for another set of iterations \
                        but this time with a lower lr')
parser.add_argument('--eval', type=str2bool, nargs='?',
                    const=True, default=False,
                    help='Validate on the testset')
parser.add_argument('--random_dataset', type=str2bool, nargs='?',
                    const=True, default=True,
                    help='Train using a randomly generated tsp dataset')
