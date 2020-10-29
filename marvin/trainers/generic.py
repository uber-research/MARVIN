'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This is the generic trainer that contains the training-agnostic evaluation script
and the other basic functions that are used by all of the training methods.
'''

import copy
import math
import os
import pickle
import random
import statistics
from time import time

import numpy as np
import tensorboardX
import torch
from torch import nn, optim
from tqdm import tqdm

from marvin.graph_gen.generate import DataGenerator
from marvin.models import models
from marvin.models.general import device
from marvin.traffic_simulation.traffic_sim import perform_timestep
from marvin.utils.utils import optimized

random.seed(0)


class Trainer:
    """Generic Trainer object to be inherited"""

    def __init__(self, args):
        print("Beginning Initialization...")

        self.args = args
        self.set_seed()
        self.initialize_models()
        self.initialize_constants()
        self.initialize_tensorboard()

        if self.args.random_dataset:
            self.testset = \
                DataGenerator.random_dataset(100, self.args)
        elif self.args.eval:
            self.testset = \
                DataGenerator.dataset_testset(400, args=self.args)
        else:
            self.testset = \
                DataGenerator.dataset_valset(100, args=self.args)

        if self.args.load:
            self.load_model(self.model, self.args.load)

    def initialize_tensorboard(self):
        """Initializes the tensorboard object to use to store values"""
        self.tensorboard = tensorboardX.SummaryWriter(
            log_dir='{}/logs_\
{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed={}'.format(
                self.args.log_dir,
                self.trainer,
                self.args.tag,
                self.args.max_size,
                self.args.min_size,
                self.args.model,
                self.args.num_agents,
                self.args.pass_range,
                self.args.ch_q,
                self.args.k,
                self.args.eps,
                self.args.lr,
                self.args.batch_size,
                self.args.trainer,
                self.args.seed
            ))

    def initialize_constants(self):
        """Initialize the constant values that will be used"""
        self.eps = self.args.eps
        self.epoch = 0
        self.ce = nn.CrossEntropyLoss()
        self.identifier = str(time())
        # ## specify for each trainer the name
        self.trainer = ''

    def set_seed(self):
        """Initialize the seed in every needed context"""
        if self.args.seed == -1:
            self.args.seed = int(time())

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        DataGenerator.shuffle_dataset()

    def allocate_graph(self, graph):
        """Moves the tensor objects to the correct device"""
        dictionary = graph.__dict__
        for key, val in dictionary.items():
            if type(val) == torch.Tensor:
                dictionary[key] = val.to(device)
            elif type(val) == list:
                for i in range(len(val)):
                    if type(val[i]) == torch.Tensor:
                        val[i] = val[i].to(device)
                dictionary[key] = val

        graph.__dict__.update(dictionary)
        return graph

    def initialize_models(self):
        """initialize the necessary models to be used"""
        self.model = models[self.args.model](self.args).to(device=device)
        self.best_model = copy.deepcopy(self.model)
        self.min_cost = np.inf
        self.max_rate = -np.inf

    def load_model(self, model, path):
        """Loads the model based on the path

        Arguments:
            model {Model} -- One of the various models
            path {str} -- patht to one of the models
        """
        load_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(load_dict)

    def calculate_gini(self, dist):
        """Calculates the gini coefficient for how even the input
        distribution is

        Arguments:
            dist {List[int]} -- total accumulation of values in each
                of the possible categories

        Returns:
            int -- gini coefficient
        """
        dist = sorted(dist)
        dist.insert(0, 0)
        ordered = np.array(dist)
        for i in range(1, len(ordered)):
            ordered[i] = ordered[i-1] + ordered[i]

        return 1 - np.trapz(ordered) * 2 / (len(ordered) - 1) / ordered[-1]

    def clip_grad_norms(self, param_groups, max_norm=np.inf):
        """Function for clipping the gradient norm of the model.
        If not given a maximum normal vector we do not clip

        Arguments:
            param_groups {List[torch.nn.Parameter]} -- all model parameters

        Keyword Arguments:
            max_norm {int} -- maximum norm of the gradient (default: {np.inf})

        Returns:
            Tuple[int, int] -- gradient norm and the clipped gradient norm
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] \
            if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped

    def reset_models(self):
        """Resets the models after each graph traversal"""
        self.model.reset()

    def next_epoch(self):
        """Do whatever initialization you need at the start of each
        epoch"""
        raise NotImplementedError

    def get_static_features(self, graph):
        """Calculates the input features that do not change during a rollout

        Arguments:
            graph: {ProcessedGraph} -- object that we are working on

        Returns:
            torch.Tensor -- Input graph features (number of nodes, 4)
        """
        # number of output edges
        f1 = (graph.adj > 0).sum(0).view(-1).float()
        # number of input edges
        f2 = (graph.adj > 0).sum(1).view(-1).float()
        # total weight of the output edges
        f3 = graph.adj.sum(0).view(-1).float()
        # total weight of the input edges
        f4 = graph.adj.sum(1).view(-1).float()

        return torch.stack((f1, f2, f3, f4), 1)

    def get_dynamic_features(self, graph, position, undiscovered, pass_count, congestion):
        """Calculates the input features that do change during a rollout

        Arguments:
            graph: {ProcessedGraph} -- object that we are working on
            position: {int} -- index of the agent's current position
            undiscovered: {torch.Tensor} -- binary indicator if an agent has
                visited each node
            pass_count: {torch.Tensor} -- how many more times each node must
                be visited in order to be completely mapped
            congestion: {torch.Tensor} -- the current congestion density index
                at each node

        Returns:
            torch.Tensor -- Input graph features (number of nodes, 6)
        """

        # The agent's position
        f5 = torch.eye(graph.adj.shape[0],
                        device=device)[position].float()

        # The nodes still undiscovered by the agent
        f6 = undiscovered.float()

        mask = (pass_count > 0).float()

        # The nodes that have been visited but need to be revisited
        f7 = (pass_count > 0).float() * \
            (undiscovered == 0).float()

        # distance between that agent and every other node
        f8 = graph.dense[position].float()

        # congestion at each node
        f9 = ((sum(undiscovered) < self.args.num_agents).float() * \
            torch.tensor(congestion, device=device).float())

        # which nodes are immediately adjacent to the agent's
        # current position
        f10 = (graph.adj[position] > 0).float()

        return torch.stack((f5, f6, f7, f8, f9, f10), 1)


    def get_num_passes(self):
        """Gets the number of passes required at each node for
        the desired distribution"""

        if self.args.gaussian_dist:
            pass_count = torch.tensor([
                int(max(1, random.gauss(self.args.pass_range, 1)))
                for _ in range(self.args.num_nodes)],
                device=device).float()
        elif self.args.bimodal_dist:
            pass_count = torch.tensor([2 + random.randint(0, 1) * 2
                                       for _ in range(self.args.num_nodes)],
                                      device=device).float()
        elif self.args.exp_dist:
            pass_count = torch.tensor([
                math.ceil(-math.log(random.random()) / math.log(2))
                for _ in range(self.args.num_nodes)],
                device=device).float()
        else:
            pass_count = torch.tensor(
                    [random.randint(1, max(1, self.args.pass_range))
                        for _ in range(self.args.num_nodes)],
                    device=device).float()

        return pass_count

    def train(self):
        """
        Call to train for the set number of epochs
        """
        if self.args.SGD:
            optimizer = optim.SGD(list(self.model.parameters()),
                                  lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = optim.Adam(list(self.model.parameters()),
                                   lr=self.args.lr)

        # Set the learning rate decay
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.args.decay_every, 0.1)

        # epoch number to start on
        starting_epoch = self.epoch

        for epoch in tqdm(
                range(starting_epoch,
                        starting_epoch + self.args.epochs)):

            self.next_epoch()

            self.epoch = epoch
            if not self.args.eval:
                loss, train_info, info_name = self.run_batch()

                # Do the optimization step
                optimizer.zero_grad()
                loss.backward()
                grad_norms, _ = self.clip_grad_norms(
                    optimizer.param_groups, self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Add tensorboard parameters
                self.tensorboard.add_scalar('loss',
                                            loss.item(),
                                            global_step=epoch)
                self.tensorboard.add_scalar('grad norm',
                                            grad_norms[0],
                                            global_step=epoch)
                self.tensorboard.add_scalar('learning rate',
                                            optimizer.param_groups[-1][
                                                'lr'
                                            ],
                                            global_step=epoch)
                self.tensorboard.add_scalar(info_name,
                                            train_info,
                                            global_step=epoch)

            # every val_every epochs validate results
            if epoch % self.args.val_every == 0:
                c = self.validation_stats()
                print("Average validation cost: {}\n".format(c[0]))

                # updates tensorboard params
                self.tensorboard.add_scalar('total cost',
                                            c[0],
                                            global_step=epoch)
                self.tensorboard.add_scalar('Relative Performance',
                                            c[1],
                                            global_step=epoch)
                self.tensorboard.add_scalar('Gini coefficient',
                                            c[2],
                                            global_step=epoch)
                self.tensorboard.add_scalar('Error',
                                            c[3],
                                            global_step=epoch)
                self.tensorboard.add_scalar('Optimal Cost',
                                            c[4],
                                            global_step=epoch)

                # update best found model if necessary
                if c[0] < self.min_cost:
                    self.best_model.load_state_dict(
                        self.model.state_dict()
                    )
                    self.min_cost = c[0]

            # preemptively save the best model every 1000 epochs
            if epoch % self.args.save_every == 0:
                model_dict = self.model.state_dict()
                torch.save(model_dict, os.path.join(
                    self.tensorboard.file_writer.get_logdir(),
                    'weights.pb'
                ))

                model_dict = self.best_model.state_dict()
                torch.save(model_dict, os.path.join(
                    self.tensorboard.file_writer.get_logdir(),
                    'best_weights{}.pb'.format(self.min_cost)
                ))

        # saves model after training
        model_dict = self.model.state_dict()
        torch.save(model_dict, os.path.join(
            self.tensorboard.file_writer.get_logdir(),
            'weights.pb')
        )

        model_dict = self.best_model.state_dict()
        torch.save(model_dict, os.path.join(
            self.tensorboard.file_writer.get_logdir(),
            'best_weights{}.pb'.format(self.min_cost))
        )

        # begin again with smaller learning rate if necessary
        if self.args.tune:
            self.args.tune = False
            self.args.lr /= 100
            print('Beginning Tuning')
            self.train()

    def validation_stats(self):
        """Perform the validation of the model on the validation
        or test set.

        Returns:
            List[float] -- stats to return for validation:
                1: Average cost of the traversal
                2: Average performance relative to the optimal
                3: Gini coefficient
                4: Standard deviation of the graph traversal distance
                5: Average optimal cost of the traversal
        """

        # Information to save for later analysis and path visualization
        data_to_save = {"pass_counts": [], "paths": [], "graphs": [],
                        "point_graph": [], "opt_paths": [],
                        'path_graph': [],
                        'values': [[] for _ in range(self.args.num_agents)],
                        'pos_values': [[] for _ in range(self.args.num_agents)]}

        self.model = self.model.eval()
        total_cost = []
        relative_performance = []
        optimal_cost = []
        total_agent_costs = [0 for _ in range(self.args.num_agents)]
        print("Validating...")

        for x in tqdm(range(len(self.testset))):
            cost = 0
            agent_costs = [0 for _ in range(self.args.num_agents)]

            graph = self.allocate_graph(self.testset[x])
            self.args.num_nodes = graph.adj.shape[0]

            self.model.reset()
            undiscovered = [torch.ones(
                self.args.num_nodes, device=device).float()
                            for _ in range(self.args.num_agents)]

            # how many times each node must be traversed
            pass_count = self.get_num_passes()

            congestion = np.random.uniform(0, 1, self.args.num_nodes)
            capacities = np.ones(self.args.num_nodes)
            simple_adj = graph.adj.cpu().numpy()

            in_routes = (graph.adj.cpu().numpy() > 0).sum(0)

            # Return the adjacency matrix to the original costs
            solution_matrix = graph.adj.cpu().numpy() * \
                float(graph.actual_distance.max() / graph.dense.max())

            if self.args.traffic:
                # calculate what a realistic set of congestion states would be
                # doesn't work without cvxpy

                for _ in range(10):
                    congestion, edge_val = perform_timestep(
                        simple_adj, capacities, congestion, in_routes
                    )

                solution_matrix = (solution_matrix.transpose() /
                                   np.clip(1 - congestion ** 3, 0.25, 1)
                                   ).transpose()

            # calculate the optimal cost of this traversal
            opt_cost, opt_path = optimized.LKH(
                solution_matrix, pass_count=pass_count,
                num_agents=self.args.num_agents,
                identifier=self.identifier, is_tour=True)

            optimal_cost.append(sum(opt_cost))

            data_to_save["opt_paths"].append(copy.deepcopy(opt_path))
            pos = []

            for i in range(self.args.num_agents):
                pos.append(opt_path[i][0])

            og_pos = copy.deepcopy(pos)

            # Initialize the location of the agents as being discovered
            for i in range(self.args.num_agents):
                undiscovered[i][pos[i]] = 0.0
                pass_count[pos[i]] = 0

            # Information about the graph to be saved
            data_to_save['pass_counts'].append(
                copy.deepcopy(pass_count.cpu().numpy())
            )
            data_to_save['paths'].append([copy.deepcopy(pos)])
            data_to_save['graphs'].append(
                copy.deepcopy(graph.adj.cpu().numpy())
            )
            data_to_save["point_graph"].append(copy.deepcopy(graph.points))
            data_to_save['path_graph'].append(copy.deepcopy(graph.pred))

            static_features = self.get_static_features(graph)

            costs_so_far = [0 for _ in range(self.args.num_agents)]
            while pass_count.sum() > 0:
                # The next agent to go is the one with the least accumulated
                # cost if asynch, otherwise its just the next agent
                i = costs_so_far.index(min(costs_so_far))

                if pass_count.sum() <= 0:
                    continue

                dynamic_features = self.get_dynamic_features(graph, pos[i],
                    undiscovered[i], pass_count, congestion)

                feature = torch.cat((static_features, dynamic_features), 1)

                # normalize the featues
                feature = feature / (torch.clamp(feature.max(0)[0], min=1.))

                mask = (pass_count > 0).float()

                action = self.model(feature, graph, i, pos[i], mask=mask)

                data_to_save['values'][i].append(action.detach().cpu().numpy())
                data_to_save['pos_values'][i].append(pos[i])

                action = int(torch.argmax(action))

                # We call on agents sequentially instead of when they finish
                # their last action, this simulates that
                if not self.args.asynch:
                    costs_so_far[i] += 1

                starting = True

                # perform all local steps necessary to perform the desired
                # global step
                while (pos[i] != action) or starting:
                    starting = False
                    next = int(graph.pred[pos[i], action])

                    # Factor due to traffic
                    div = max(1 - congestion[next] ** 3, 0.25) \
                        if self.args.traffic else 1

                    cost += graph.actual_distance[pos[i], next] / div

                    if self.args.asynch:
                        costs_so_far[i] += graph.actual_distance[pos[i], next] / div

                    agent_costs[i] += graph.actual_distance[pos[i], next] / div

                    pos[i] = next
                    undiscovered[i][pos[i]] = 0.0
                    pass_count[pos[i]] = max(0, pass_count[pos[i]] - 1)

                    data_to_save['paths'][-1].append(copy.deepcopy(pos))

            # calculate the cost of returning to the starting depot
            for i in range(len(pos)):
                while (pos[i] != og_pos[i]):
                    next = int(graph.pred[pos[i], og_pos[i]])
                    div = max(1 - congestion[next] ** 3, 0.25) \
                        if self.args.traffic else 1

                    cost += graph.actual_distance[pos[i], next] / div
                    agent_costs[i] += graph.actual_distance[pos[i], next] / div

                    data_to_save['paths'][-1].append(copy.deepcopy(pos))
                    pos[i] = next

            # add up the invididual agent cost for the gini coefficient
            agent_costs = sorted(agent_costs)
            for i in range(self.args.num_agents):
                total_agent_costs[i] += agent_costs[i]

            data_to_save['paths'][-1].append([-1, -1])

            total_cost.append(cost)
            relative_performance.append(sum(opt_cost) / cost)

        self.model = self.model.train()

        # save the path data
        if not os.path.isdir('{}/testing_data'.format(self.args.log_dir)):
            os.mkdir('{}/testing_data'.format(self.args.log_dir))

        path = '{}/testing_data/logs_\
{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed={}.pkl'.format(
                self.args.log_dir,
                self.trainer,
                self.args.tag,
                self.args.max_size,
                self.args.min_size,
                self.args.model,
                self.args.num_agents,
                self.args.pass_range,
                self.args.ch_q,
                self.args.k,
                self.args.eps,
                self.args.lr,
                self.args.batch_size,
                self.args.trainer,
                self.args.seed
            )

        pickle.dump(data_to_save, open(path, 'wb'))

        total_cost = [float(t) for t in total_cost]

        return sum(total_cost) / \
            (len(self.testset) + np.finfo(np.float32).eps), \
            sum(relative_performance) / \
               (float(len(relative_performance)) + np.finfo(np.float32).eps), \
            self.calculate_gini(total_agent_costs), \
            statistics.stdev(total_cost) / (len(self.testset) ** 0.5), \
            float(sum(optimal_cost)) / \
                 (len(self.testset) + np.finfo(np.float32).eps)
