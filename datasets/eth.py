import os
import glob
from struct import pack
import tensorflow as tf
import numpy as np

from utils.common import Notify
from .base_dataset import BaseDataset


class Eth(BaseDataset):
    default_config = {
        'num_parallel_calls': 10, 'truncate': None
    }

    def _init_dataset(self, **config):
        print(Notify.INFO, "Initializing dataset:", config['data_name'], Notify.ENDC)
        base_path = config['data_root']
        suffix = self.config['post_format']['suffix']

        img_paths = []
        dump_paths = []

        data_split = config['data_split']

        types = ('*.jpg', '*.png', '*.JPG', '*.PNG')
        for seq in data_split:
            dump_folder = os.path.join(config['dump_root'], seq)
            if not os.path.exists(dump_folder):
                os.makedirs(dump_folder)

            dataset_folder = os.path.join(base_path, seq)
            image_folder = os.path.join(dataset_folder, 'images')
            for filetype in types:
                image_list = glob.glob(os.path.join(image_folder, filetype))
                img_paths.extend(image_list)

        if config['truncate'] is not None:
            print(Notify.WARNING, "Truncate from",
                  config['truncate'][0], "to", config['truncate'][1], Notify.ENDC)
            img_paths = img_paths[config['truncate'][0]:config['truncate'][1]]

        self.data_length = len(img_paths)
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])

        seq_names = [i.split('/')[-3] for i in img_paths]
        img_names = [os.path.splitext(os.path.basename(i))[0] for i in img_paths]
        dump_paths = [os.path.join(config['dump_root'], seq_names[i],
                                   img_names[i] + '.h5') for i in range(len(img_names))]

        files = {'image_paths': img_paths, 'dump_paths': dump_paths}
        return files

    def _format_data(self, data):
        dump_path = data['dump_path'].decode('utf-8')
        seq_name = dump_path.split('/')[-2]
        dataset_folder = os.path.join(self.config['submission_root'], seq_name)
        dump_folder = os.path.join(dataset_folder, 'reconstruction' + self.config['post_format']['suffix'])
        kpt_dump_folder = os.path.join(dump_folder, 'keypoints')
        desc_dump_folder = os.path.join(dump_folder, 'descriptors')
        if not os.path.exists(kpt_dump_folder):
            os.makedirs(kpt_dump_folder)
        if not os.path.exists(desc_dump_folder):
            os.makedirs(desc_dump_folder)

        kpt_dump_path = os.path.join(kpt_dump_folder, os.path.basename(data['image_path'].decode('utf-8') + '.bin'))
        desc_dump_path = os.path.join(desc_dump_folder, os.path.basename(data['image_path'].decode('utf-8') + '.bin'))

        desc = data['dump_data'][0].astype(np.float32)
        kpt = data['dump_data'][1].astype(np.float32)

        zeros = np.zeros_like(kpt)
        kpt = np.concatenate([kpt, zeros], axis=-1)

        num_features = desc.shape[0]
        loc_dim = kpt.shape[1]
        feat_dim = desc.shape[1]

        det_head = np.stack((num_features, loc_dim)).astype(np.int32)
        det_head = pack('2i', *det_head)

        desc_head = np.stack((num_features, feat_dim)).astype(np.int32)
        desc_head = pack('2i', *desc_head)

        kpt = pack('f' * loc_dim * num_features, *(kpt.flatten()))
        desc = pack('f' * feat_dim * num_features, *(desc.flatten()))

        with open(kpt_dump_path, 'wb') as fout:
            fout.write(det_head)
            if num_features > 0:
                fout.write(kpt)

        with open(desc_dump_path, 'wb') as fout:
            fout.write(desc_head)
            if num_features > 0:
                fout.write(desc)