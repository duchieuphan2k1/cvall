#!/usr/bin/env python3
import sys
sys.path.append('.')
import os
import yaml
from damo_yolo2.damo.config import Config as MyConfig
from pathlib import Path
from easydict import EasyDict

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        model_path = dir_path

        # model_path = "/".join(dir_path.split("\\")[:-1])
        yaml_path = os.path.join(model_path, "params.yaml")

        model_params = EasyDict(yaml.safe_load(open(yaml_path, 'r')))

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = model_params.miscs.eval_interval_epochs
        self.miscs.ckpt_interval_epochs = model_params.miscs.ckpt_interval_epochs
        # optimizer
        self.train.batch_size = model_params.train.batch_size
        self.train.base_lr_per_img = model_params.train.base_lr_per_img
        self.train.min_lr_ratio = model_params.train.min_lr_ratio
        self.train.weight_decay = model_params.train.weight_decay
        self.train.momentum = model_params.train.momentum
        self.train.no_aug_epochs = model_params.train.no_aug_epochs
        self.train.warmup_epochs = model_params.train.warmup_epochs
        self.train.total_epochs = model_params.train.total_epochs

        # augment
        self.train.augment.transform.image_max_range = model_params.train.augment.transform.image_max_range
        self.train.augment.mosaic_mixup.mixup_prob = model_params.train.augment.mosaic_mixup.mixup_prob
        self.train.augment.mosaic_mixup.degrees = model_params.train.augment.mosaic_mixup.degrees
        self.train.augment.mosaic_mixup.translate = model_params.train.augment.mosaic_mixup.translate
        self.train.augment.mosaic_mixup.shear = model_params.train.augment.mosaic_mixup.shear
        self.train.augment.mosaic_mixup.mosaic_scale = model_params.train.augment.mosaic_mixup.mosaic_scale

        self.dataset.train_ann = ('train_dataset', )
        self.dataset.val_ann = ('valid_dataset', )
        self.dataset.trainset = model_params.dataset.trainset
        self.dataset.testset = model_params.dataset.testset

        self.model_name = model_params.model_name

        # backbone
        structure = self.read_structure(
            'damo_yolo2/damo/base_models/backbones/nas_backbones/tinynas_L25_k1kx.txt')
        TinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 0.75,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': len(model_params.dataset.class_names),
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': model_params.model.ZeroHead.reg_max,
            'act': model_params.model.ZeroHead.act,
            'nms_conf_thre': model_params.model.ZeroHead.nms_conf_thre,
            'nms_iou_thre': model_params.model.ZeroHead.nms_iou_thre,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = model_params.dataset.class_names