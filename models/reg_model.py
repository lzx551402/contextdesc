import numpy as np
import cv2

from .base_model import BaseModel


class RegModel(BaseModel):
    output_tensors = "res5c:0"
    default_config = {'max_dim': 1024}

    def _model(self):
        return

    def _init_model(self):
        return

    def _run(self, data):
        # TODO: allow for larger batch size
        assert len(data.shape) == 3
        max_dim = max(data.shape[0], data.shape[1])
        if max_dim > self.config['max_dim']:
            downsample_ratio = self.config['max_dim'] / float(max_dim)
            data = cv2.resize(data, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
        feed_dict = {"input:0": np.expand_dims(data, 0)}
        returns = self.sess.run(self.output_tensors, feed_dict=feed_dict)
        return np.squeeze(returns, axis=0)
