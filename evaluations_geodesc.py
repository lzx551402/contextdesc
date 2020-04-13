#!/usr/bin/env python3

"""
Copyright 2019, Zixin Luo, HKUST.
Evaluation script.
"""

import os
import yaml

import h5py
import numpy as np
import tensorflow as tf
import progressbar

from datasets import get_dataset
from models import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")


def extract_loc_feat(config):
    """Extract local features."""
    prog_bar = progressbar.ProgressBar()
    config['stage'] = 'loc'
    dataset = get_dataset(config['data_name'])(**config)
    prog_bar.max_value = dataset.data_length
    test_set = dataset.get_test_set()

    model = get_model('geodesc_model')(config['pretrained']['loc_model'], **(config['loc_feat']))
    idx = 0
    while True:
        try:
            data = next(test_set)
            # detect SIFT keypoints and crop image patches.
            kpts, descs = model.run_test_data(data['image'])
            data['dump_data'] = []
            data['dump_data'].append(descs)
            data['dump_data'].append(kpts[:, 0:2])
            dataset.format_data(data)
            prog_bar.update(idx)
            idx += 1
        except dataset.end_set:
            break
    model.close()


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(config['dump_root']):
        os.mkdir(config['dump_root'])
    # extract local features and keypoint matchability.
    if config['loc_feat']['infer']:
        extract_loc_feat(config)


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['config'])
    tf.app.run()
