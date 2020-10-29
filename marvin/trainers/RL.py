'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Here is the reinforcement learning trainer. It trains the model using either
policy gradent (pg) or proximal policy optimization (ppo).
'''

import copy
import random

import numpy as np
import torch
from torch.distributions import Categorical

from marvin.graph_gen.generate import DataGenerator
from marvin.models.general import device
from marvin.traffic_simulation.traffic_sim import perform_timestep
from marvin.trainers.generic import Trainer

random.seed(0)


class RLTrainer(Trainer):
    def initialize_constants(self):
        """Adds the trainer name for the tensorboard file"""
        super().initialize_constants()
        self.trainer = "RL"

    def initialize_models(self):
        """Create seperate reference ppo model"""
        super().initialize_models()

        if self.args.trainer.lower() == 'ppo':
            self.ppo_model = copy.deepcopy(self.model).eval()
            self.ppo_model.args = self.model.args
            self.ppo_model.comm_protocol.args = self.model.args

    def reset_models(self):
        """reset ppo model if necessary"""
        super().reset_models()

        if self.args.trainer.lower() == 'ppo':
            self.ppo_model.reset()

    def next_epoch(self):
        """reset reference ppo model if necessary"""
        if self.args.trainer.lower() == 'ppo' and \
                self.epoch % self.args.ppo_epochs:
            self.ppo_model.load_state_dict(self.model.state_dict())

    def get_action_probs(self, log_prob, feature, graph, i, pos, action):
        """Calculates the probability of taking a given action according
        to what training method you wish to use

        Arguments:
            log_prob {torch.Tensor} -- log probability of the given action
                (1,)
            feature {torch.Tensor} -- input feature to the model
                (n, input_shape)
            graph {ProcessedGraph} -- Processed graph data information
            i {int} -- index of the agent
            pos {int} -- position of the agent in the graph
            action {int} -- which action we ultimately decide to take

        Returns:
            torch.Tensor -- value that acts as your action probability
        """

        if self.args.trainer.lower() == 'ppo':
            old_pred = self.ppo_model(feature, graph, i, pos).detach()
            c = Categorical(old_pred)
            old_prob = c.log_prob(action)
            return torch.exp(log_prob) / torch.exp(old_prob)
        else:
            return log_prob

    def combine_action_rewards(self, actions, rewards):
        """Combines the action probabilities and their associated rewards
        according to whatever training method chosen

        Arguments:
            actions {torch.Tensor} -- action probabilties
            rewards {torch.Tensor} -- normalized rewards

        Returns:
            torch.Tensor -- Final loss of the agents given rewards and actions
        """

        if self.args.trainer.lower() == 'ppo':
            clamped = torch.clamp(
                actions, 1 - self.args.ppo_eps, 1 + self.args.ppo_eps
            )
            weaker_clamp = torch.clamp(actions, 0, self.args.max_clamp)

            return torch.mean(
                torch.max(torch.mul(clamped, rewards).mul(-1),
                          torch.mul(weaker_clamp, rewards).mul(-1))
                )
        else:
            return torch.mean((actions * rewards).mul(-1))

    def run_batch(self):
        """Run and calculate the loss for one batch

        Returns:
            Tuple[torch.Tensor, float, str] --
                1: the loss of the batch to be minized
                2: the average reward in this episode
                3: the name of the above metric
        """
        self.model = self.model.train()
        episodic_rewards = []
        discounted_rewards = []
        history = torch.tensor([], device=device)

        for _ in range(self.args.batch_size):
            if self.args.random_dataset:
                graph = self.allocate_graph(DataGenerator.random_dataset(1, self.args)[0])
            else:
                graph = self.allocate_graph(DataGenerator.get_graph(self.args))

            self.args.num_nodes = graph.adj.shape[0]

            pos = [random.randint(0, self.args.num_nodes - 1)
                   for _ in range(self.args.num_agents)]

            self.reset_models()
            undiscovered = [torch.ones(
                self.args.num_nodes, device=device).float()
                            for _ in range(self.args.num_agents)]

            # how many times each node must be traversed
            pass_count = self.get_num_passes()
            og_pos = copy.deepcopy(pos)

            # Initialize the location of the agents as being discovered
            for i in range(self.args.num_agents):
                undiscovered[i][pos[i]] = 0.0
                pass_count[pos[i]] = 0

            static_features = self.get_static_features(graph)

            congestion = np.random.uniform(0, 1, self.args.num_nodes)
            capacities = np.ones(self.args.num_nodes)
            simple_adj = graph.adj.cpu().numpy()

            in_routes = (graph.adj.cpu().numpy() > 0).sum(0)

            if self.args.traffic:
                # calculate what a realistic set of congestion states would be
                for _ in range(10):
                    congestion, edge_val = perform_timestep(
                        simple_adj, capacities, congestion, in_routes
                    )

            costs = []

            while pass_count.sum() > 0:
                for i in range(self.args.num_agents):
                    if pass_count.sum() <= 0:
                        continue

                    dynamic_features = self.get_dynamic_features(graph, pos[i],
                        undiscovered[i], pass_count, congestion)

                    feature = torch.cat((static_features, dynamic_features), 1)

                    # normalize the features
                    feature = feature / (torch.clamp(
                        feature.max(0)[0], min=1.)
                    )

                    mask = (pass_count > 0).float()

                    prediction = self.model(
                        feature, graph, i, pos[i], mask=mask
                    )

                    # action distribution put out by the model
                    dist = Categorical(prediction)

                    # get action that the agent will take
                    if random.random() < self.eps:
                        # random action for exploration purposes
                        action = torch.multinomial(mask, 1).squeeze()
                    else:
                        action = dist.sample()

                    # Add the action probability to history
                    prob = self.get_action_probs(dist.log_prob(action),
                                                 feature,
                                                 graph,
                                                 i,
                                                 pos[i],
                                                 action).view(-1)

                    if history.shape != (0,):
                        history = torch.cat((history, prob))
                    else:
                        history = prob

                    action = int(action)
                    temp_costs = 0
                    starting = True

                    # perform all local steps necessary to perform the desired
                    # global step
                    while pos[i] != action or starting:
                        starting = False
                        next = int(graph.pred[pos[i], action])

                        # Factor due to traffic
                        div = max(1 - congestion[next] ** 3, 0.25) \
                            if self.args.traffic else 1

                        temp_costs += graph.actual_distance[pos[i], next] / div

                        pos[i] = next
                        undiscovered[i][pos[i]] = 0.0
                        pass_count[pos[i]] = max(0, pass_count[pos[i]] - 1)

                    costs.append(temp_costs)

            total_cost = sum(costs)

            for i in range(len(pos)):
                temp_costs = 0

                starting = True
                while (pos[i] != og_pos[i]) or starting:
                    starting = False
                    next = int(graph.pred[pos[i], og_pos[i]])

                    # Factor due to traffic
                    div = max(1 - congestion[next] ** 3, 0.25) \
                        if self.args.traffic else 1

                    temp_costs += graph.actual_distance[pos[i], next] / div

                    pos[i] = next

                total_cost += temp_costs

            Rg = 0
            episodic = []
            discounted = []

            # calculates the episodic and greedy rewards at each point in
            # the last episode
            for r in range(len(costs) - 1, -1, -1):
                if sum(costs) == 0:
                    R = 0
                    Rg = 0
                else:
                    R = -total_cost / self.args.num_nodes
                    Rg = -costs[r] + self.args.gamma * Rg

                episodic.insert(0, R)
                discounted.insert(0, Rg)

            episodic_rewards += episodic
            discounted_rewards += discounted

        # Normalize and turn rewards into tensors
        episodic_rewards = torch.tensor(
            episodic_rewards, device=device).float()
        discounted_rewards = torch.tensor(
            discounted_rewards, device=device).float()

        # For metrics
        old_rewards = torch.mean(episodic_rewards).detach().clone()

        # normalize the rewards
        episodic_rewards = (episodic_rewards - episodic_rewards.mean()) \
            / (episodic_rewards.std() + np.finfo(np.float32).eps)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) \
            / (discounted_rewards.std() + np.finfo(np.float32).eps)

        # Calculate the episodic loss
        loss1 = self.combine_action_rewards(history, episodic_rewards)
        # Calculate the greedy loss
        loss2 = self.combine_action_rewards(history, discounted_rewards)

        # combine losses with the correct ratio
        loss = loss1 + self.args.discount * loss2
        return loss, float(old_rewards), 'Average Reward'
