#!/usr/bin/env python3

"""
Copyright 2019, Zixin Luo, HKUST.
Inference script.
"""


import sys
from abc import ABCMeta, abstractmethod
import collections
import tensorflow as tf

sys.path.append('..')

from utils.tf import load_frozen_model


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BaseModel(metaclass=ABCMeta):
    """Base model class."""

    @abstractmethod
    def _model(self):
        """Implements the graph of the model."""
        raise NotImplementedError

    @abstractmethod
    def _run(self, data):
        raise NotImplementedError

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    def run_test_data(self, data):
        """"""
        out_data = self._run(data)
        return out_data

    def __init__(self, model_path, **config):
        self.model_path = model_path
        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self._init_model()
        graph = load_frozen_model(self.model_path, print_nodes=False)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
