# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os
from utils.handle_config import ConfigHandler

class DatasetCatalog(object):
    cfgs = ConfigHandler()
    using_params = cfgs.get_using_params()
    DATA_DIR = '.'
    DATASETS = {
        'train_dataset': {
            'img_dir': using_params.train.img_dir,
            'ann_file': using_params.train.ann_file
        },
        'valid_dataset': {
            'img_dir': using_params.test.img_dir,
            'ann_file': using_params.test.ann_file
        },
        'test_dataset': {
            'img_dir': using_params.test.img_dir,
            'ann_file': using_params.test.ann_file
        },
        }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs['img_dir']),
            ann_file=os.path.join(data_dir, attrs['ann_file']),
        )
        return dict(
            factory='COCODataset',
            args=args,
        )

        return None
