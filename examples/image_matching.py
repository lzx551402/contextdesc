#!/usr/bin/env python
"""
Copyright 2019, Zixin Luo, HKUST.
Inference script.
"""

from __future__ import print_function

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))

from utils.tf import load_frozen_model
from utils.opencvhelper import SiftWrapper, MatcherWrapper

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_prefix', '../models/contextdesc-e2e',
                           """Path to the local and augmentation model.""")
tf.app.flags.DEFINE_string('regional_model', '../models/retrieval_resnet50.pb',
                           """Path to the regional model.""")
tf.app.flags.DEFINE_integer('max_kpt_num', 2048,
                            """Maximum number of keypoints. Sampled by octave.""")
tf.app.flags.DEFINE_string('img1_path', '../imgs/test_img1.png',
                           """Path to the first image.""")
tf.app.flags.DEFINE_string('img2_path', '../imgs/test_img2.png',
                           """Path to the second image.""")
tf.app.flags.DEFINE_boolean('ratio_test', False,
                            """Whether to apply ratio test in matching.""")


def get_kpt_and_patch(img_paths):
    """
    Detect SIFT keypoints and crop image patches.
    Args:
        img_paths: a list of image paths.
    Returns:
        all_patches: Image patches (Nx32x32).
        all_npy_kpt: NumPy array, normalized keypoint coordinates ([-1, 1]).
        all_cv_kpt: OpenCV KeyPoint, unnormalized keypoint coordinates.
        all_sift_desc: SIFT features (Nx128).
        all_imgs: RGB images.
    """
    sift_wrapper = SiftWrapper(n_sample=FLAGS.max_kpt_num, peak_thld=0.04)
    sift_wrapper.standardize = False # the network has handled this step.
    sift_wrapper.create()

    all_patches = []
    all_npy_kpts = []
    all_cv_kpts = []
    all_sift_desc = []
    all_imgs = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., ::-1]
        npy_kpts, cv_kpts = sift_wrapper.detect(gray)
        sift_desc = sift_wrapper.compute(gray, cv_kpts)

        kpt_xy = np.stack(((npy_kpts[:, 0] - gray.shape[1] / 2) / (gray.shape[1] / 2),
                           (npy_kpts[:, 1] - gray.shape[0] / 2) / (gray.shape[0] / 2)), axis=-1)
        sift_wrapper.build_pyramid(gray)
        patches = sift_wrapper.get_patches(cv_kpts)
        all_patches.append(patches)
        all_npy_kpts.append(kpt_xy)
        all_cv_kpts.append(cv_kpts)
        all_sift_desc.append(sift_desc)
        all_imgs.append(img)

    return all_patches, all_npy_kpts, all_cv_kpts, all_sift_desc, all_imgs


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    local_model_path = FLAGS.model_prefix + '.pb'
    regional_model_path = FLAGS.regional_model
    aug_model_path = FLAGS.model_prefix + '-aug.pb'

    input_imgs = [FLAGS.img1_path, FLAGS.img2_path]
    num_img = len(input_imgs)
    # detect SIFT keypoints and crop image patches.
    patches, npy_kpts, cv_kpts, sift_desc, imgs = get_kpt_and_patch(input_imgs)
    # extract local features and keypoint matchability.
    local_graph = load_frozen_model(local_model_path, print_nodes=False)
    local_feat = []
    l2_local_feat = []
    kpt_m = []
    with tf.Session(graph=local_graph) as sess:
        for i in patches:
            local_returns = sess.run(["conv6_feat:0", "kpt_mb:0", "l2norm_feat:0"],
                                     feed_dict={"input:0": np.expand_dims(i, -1)})
            local_feat.append(local_returns[0])
            kpt_m.append(local_returns[1])
            l2_local_feat.append(local_returns[2])
    tf.reset_default_graph()
    # extract regional features.
    regional_graph = load_frozen_model(regional_model_path, print_nodes=False)
    regional_feat = []
    with tf.Session(graph=regional_graph) as sess:
        for i in imgs:
            regional_returns = sess.run("res5c:0", feed_dict={"input:0": np.expand_dims(i, 0)})
            regional_feat.append(regional_returns)
    tf.reset_default_graph()
    # local feature augmentation.
    aug_graph = load_frozen_model(aug_model_path, print_nodes=False)
    aug_feat = []
    with tf.Session(graph=aug_graph) as sess:
        for i in range(num_img):
            aug_returns = sess.run("l2norm:0", feed_dict={
                "input/local_feat:0": np.expand_dims(local_feat[i], 0),
                "input/regional_feat:0": regional_feat[i],
                "input/kpt_m:0": np.expand_dims(kpt_m[i], 0),
                "input/kpt_xy:0": np.expand_dims(npy_kpts[i], 0),
            })
            aug_returns = np.squeeze(aug_returns, axis=0)
            aug_feat.append(aug_returns)
    tf.reset_default_graph()
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    sift_match, sift_mask = matcher.get_matches(
        sift_desc[0], sift_desc[1], cv_kpts[0], cv_kpts[1],
        ratio=0.8 if FLAGS.ratio_test else None, cross_check=False, err_thld=4, info='sift')

    base_match, base_mask = matcher.get_matches(
        l2_local_feat[0], l2_local_feat[1], cv_kpts[0], cv_kpts[1],
        ratio=0.89 if FLAGS.ratio_test else None, cross_check=False, err_thld=4, info='base')

    aug_match, aug_mask = matcher.get_matches(
        aug_feat[0], aug_feat[1], cv_kpts[0], cv_kpts[1],
        ratio=0.89 if FLAGS.ratio_test else None, cross_check=False, err_thld=4, info='aug')

    sift_disp = matcher.draw_matches(
        imgs[0], cv_kpts[0], imgs[1], cv_kpts[1], sift_match, sift_mask)
    base_disp = matcher.draw_matches(
        imgs[0], cv_kpts[0], imgs[1], cv_kpts[1], base_match, base_mask)
    aug_disp = matcher.draw_matches(
        imgs[0], cv_kpts[0], imgs[1], cv_kpts[1], aug_match, aug_mask)

    rows, cols = sift_disp.shape[0:2]
    white = (np.ones((rows / 50, cols, 3)) * 255).astype(np.uint8)
    disp = np.concatenate([sift_disp, white, base_disp, white, aug_disp], axis=0)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(disp)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
