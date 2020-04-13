import sys

import numpy as np
import tensorflow as tf

from .base_model import BaseModel

sys.path.append('..')

from utils.opencvhelper import SiftWrapper


class RootsiftModel(BaseModel):
    default_config = {'n_feature': 0, "n_sample": 0,
                      'batch_size': 512, 'sift_wrapper': None, 'upright': False, 'scale_diff': False,
                      'dense_desc': False, 'sift_desc': False, 'peak_thld': 0.0067, 'max_dim': 1280}

    def _init_model(self):
        self.sift_wrapper = SiftWrapper(
            n_feature=self.config['n_feature'],
            n_sample=self.config['n_sample'],
            peak_thld=self.config['peak_thld'])
        self.sift_wrapper.standardize = False  # the network has handled this step.
        self.sift_wrapper.ori_off = self.config['upright']
        self.sift_wrapper.pyr_off = not self.config['scale_diff']
        self.sift_wrapper.create()

    def _run(self, data):
        assert data.shape[-1] == 1
        gray_img = np.squeeze(data, axis=-1).astype(np.uint8)
        # detect SIFT keypoints.
        npy_kpts, cv_kpts = self.sift_wrapper.detect(gray_img)
        sift_desc = self.sift_wrapper.compute(gray_img, cv_kpts)
        return npy_kpts, sift_desc

    def _construct_network(self):
        """Model for patch description."""
        return