'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This file processes the map graphs and creates objects that can be called upon
to receive training and test data.
'''

import os
import pickle
from random import seed, shuffle

import torch  # noqa F401

seed(0)


class LoadGraph():
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


class NoViableGraph(Exception):
    """Raised when no viable graph is found when loading graphs"""
    pass


class DataGenerator:
    train_path = 'marvin/data/train_graphs'
    train_files = os.listdir(
        train_path
    )

    val_path = 'marvin/data/val_graphs'
    val_files = os.listdir(
        val_path
    )

    test_path = 'marvin/data/test_graphs'
    test_files = os.listdir(
        test_path
    )

    index = 0

    @classmethod
    def shuffle_dataset(cls):
        """Shuffles the training dataset"""

        shuffle(cls.train_files)

    def _load_dataset(self, dataset, dataset_path, num_graphs,
                      args=None, min_size=10, max_size=30):
        """Loads a dataset as an array of preprocessed graphs

        Arguments:
            dataset {List[str]} -- list of files making up the dataset
            dataset_path {str} -- overall path to the top directory of the
                dataset
            num_graphs {int} -- max number of graphs to load (could be less)

        Keyword Arguments:
            args {argparse.ArgumentParser} --
                [arguments object] (default: {None})
            min_size {int} -- [min graph size to be loaded] (default: {10})
            max_size {int} -- [max graph size to be loaded] (default: {30})

        Returns:
            List[ProcessedGraph] -- List of all processed graphs
        """

        shuffle(dataset)
        graphs = []

        if args is None:
            max_nodes = max_size
            min_nodes = min_size
        else:
            max_nodes = args.max_size
            min_nodes = args.min_size

        for i in range(len(dataset)):
            if len(graphs) >= num_graphs:
                break

            p = dataset[i]

            num_nodes = int(p.split("_")[0])

            if num_nodes < max_nodes and num_nodes > min_nodes:
                g = LoadGraph(
                    pickle.load(
                        open(os.path.join(dataset_path, p),
                             "rb"))
                    )
                graphs.append(g)

        return graphs

    @classmethod
    def dataset_testset(cls, num_graphs, **kwargs):
        return cls._load_dataset(cls, cls.test_files, cls.test_path,
                                 num_graphs, **kwargs)

    @classmethod
    def dataset_valset(cls, num_graphs, **kwargs):
        return cls._load_dataset(cls, cls.val_files, cls.val_path,
                                 num_graphs, **kwargs)

    @classmethod
    def get_graph(cls, args=None, min_size=10, max_size=30):
        """Loads a graph object within the sizing constraints

        Keyword Arguments:
            args {argparse.ArgumentParser} --
                [arguments object] (default: {None})
            min_size {int} -- [min graph size to be loaded] (default: {10})
            max_size {int} -- [max graph size to be loaded] (default: {30})

        Returns:
            ProcessedGraph -- A Processed graph object with all
                transformed versions
            of the graph
        """
        if args is None:
            max_nodes = max_size
            min_nodes = min_size
        else:
            max_nodes = args.max_size
            min_nodes = args.min_size

        # tries to find a graph with the correct sizing constraints
        # over 10000 graph
        count = 0
        while count < 10000:
            if cls.index == len(cls.train_files):
                shuffle(cls.train_files)
                cls.index = 0

            path = cls.train_files[cls.index]

            num_nodes = int(path.split("_")[0])

            cls.index += 1
            if num_nodes < max_nodes and num_nodes > min_nodes:
                return LoadGraph(
                    pickle.load(open(os.path.join(
                        cls.train_path,
                        path
                    ), 'rb'))
                )

        raise NoViableGraph("No graph within the sizing constraints was found")
