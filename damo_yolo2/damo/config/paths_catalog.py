# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'volleyball_action_train/images',
            'ann_file': 'volleyball_action_train/coco_annotations/annotation.json'
        },
        'coco_2017_val': {
            'img_dir': 'volleyball_action_test/images',
            'ann_file': 'volleyball_action_test/coco_annotations/annotation.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'volleyball_action_test/images',
            'ann_file': 'volleyball_action_test/coco_annotations/annotation.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
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
        else:
            raise RuntimeError('Only support coco format now!')
        return None
