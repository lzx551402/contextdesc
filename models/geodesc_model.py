import sys
from queue import Queue
from threading import Thread

import os
from struct import unpack
import numpy as np
import cv2
import tensorflow as tf

from .base_model import BaseModel

sys.path.append('..')

from utils.opencvhelper import SiftWrapper
from utils.spatial_transformer import transformer_crop


class GeodescModel(BaseModel):
    output_tensors = ["squeeze_1:0"]
    default_config = {'n_feature': 0, "n_sample": 0,
                      'batch_size': 512, 'sift_wrapper': None, 'upright': False, 'scale_diff': False,
                      'dense_desc': False, 'sift_desc': False, 'peak_thld': 0.0067, 'edge_thld': 10, 'max_dim': 1280}

    def _init_model(self):
        self.sift_wrapper = SiftWrapper(
            n_feature=self.config['n_feature'],
            n_sample=self.config['n_sample'],
            peak_thld=self.config['peak_thld'],
            edge_thld=self.config['edge_thld']
            )
        self.sift_wrapper.standardize = True
        self.sift_wrapper.ori_off = self.config['upright']
        self.sift_wrapper.pyr_off = not self.config['scale_diff']
        self.sift_wrapper.create()

    def _run(self, data, **kwargs):
        def _worker(patch_queue, sess, loc_feat):
            """The worker thread."""
            while True:
                patch_data = patch_queue.get()
                if patch_data is None:
                    return
                loc_returns = sess.run(self.output_tensors,
                                       feed_dict={"input:0": np.expand_dims(patch_data, -1)})
                loc_returns = loc_returns[0]
                if len(loc_returns.shape) == 1:
                    loc_returns = np.expand_dims(loc_returns, axis=0)
                loc_feat.append(loc_returns)
                patch_queue.task_done()
        gray_img = np.squeeze(data, axis=-1).astype(np.uint8)
        # detect SIFT keypoints.
        npy_kpts, cv_kpts = self.sift_wrapper.detect(gray_img)
        num_patch = len(cv_kpts)

        self.sift_wrapper.build_pyramid(gray_img)
        all_patches = self.sift_wrapper.get_patches(cv_kpts)
        # get iteration number
        batch_size = self.config['batch_size']
        if num_patch % batch_size > 0:
            loop_num = int(np.floor(float(num_patch) / float(batch_size)))
        else:
            loop_num = int(num_patch / batch_size - 1)
        # create input thread
        loc_feat = []
        patch_queue = Queue()
        worker_thread = Thread(target=_worker, args=(patch_queue, self.sess, loc_feat))
        worker_thread.daemon = True
        worker_thread.start()
        # enqueue
        for i in range(loop_num + 1):
            if i < loop_num:
                patch_queue.put(all_patches[i * batch_size: (i + 1) * batch_size])
            else:
                patch_queue.put(all_patches[i * batch_size:])
        # poison pill
        patch_queue.put(None)
        # wait for extraction.
        worker_thread.join()
        loc_feat = np.concatenate(loc_feat, axis=0)
        loc_feat = (loc_feat * 128 + 128)
        return npy_kpts, loc_feat

    def _construct_network(self):
        """Model for patch description."""
        return
