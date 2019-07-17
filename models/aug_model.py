import numpy as np
import cv2

from .base_model import BaseModel


class AugModel(BaseModel):
    output_tensors = "l2norm:0"
    default_config = {'quantz': True}

    def _model(self):
        return

    def _init_model(self):
        return

    def _run(self, data):
        reg_feat = data[0]
        loc_info = data[1]
        raw_kpts = loc_info[:, 0:5]
        npy_kpts = loc_info[:, 5:7]
        loc_feat = loc_info[:, 7:-1]
        kpt_mb = loc_info[:, -1][..., np.newaxis]

        returns = self.sess.run(self.output_tensors, feed_dict={
            "input/local_feat:0": np.expand_dims(loc_feat, 0),
            "input/regional_feat:0": np.expand_dims(reg_feat, 0),
            "input/kpt_m:0": np.expand_dims(kpt_mb, 0),
            "input/kpt_xy:0": np.expand_dims(npy_kpts, 0),
        })
        aug_feat = np.squeeze(returns, axis=0)
        if self.config['quantz']:
            aug_feat = (aug_feat * 128 + 128).astype(np.uint8)
        return aug_feat, raw_kpts
