from src.controller.handle_config import ConfigHandler
from src.controller.handle_path import PathHandler
from src.controller.handle_dataset import DatasetHandler
import os
import shutil
import yaml
from datetime import datetime

class ModelHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.dataset_handler = DatasetHandler()
        self.general_config = self.cfg_handler.get_general_config()
        self.path_handler = PathHandler()
        self.default_params = {
            'model_name': None,
            'desc': None,
            'created_date': None,
            'miscs': {
                'eval_interval_epochs': 10,
                'ckpt_interval_epochs': 10
            },
            'train': {
                'total_epochs': 300,
                'batch_size': 4,
                'base_lr_per_img': 0.00015625,
                'min_lr_ratio': 0.05,
                'weight_decay': 0.0005,
                'momentum': 0.9,
                'no_aug_epochs': 16,
                'warmup_epochs': 5,
                'augment': {
                    'transform': {
                        'image_max_range': [640, 640]
                    },
                    'mosaic_mixup': {
                        'mixup_prob': 0.15,
                        'degrees': 10.0,
                        'translate': 0.2,
                        'shear': 2.0,
                        'mosaic_scale': [0.1, 2.0]
                    }
                }
            },
            'dataset':{
                'trainset': 'coco_train_2017',
                'testset': 'coco_valid_2017',
                'class_names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
            },
            'model':{
                'ZeroHead':{
                    'act': 'silu',
                    'reg_max': 16,
                    'nms_conf_thre': 0.5,
                    'nms_iou_thre': 0.7
                }
            }
        }
    
    def get_classes_by_name(self, model_name):
        model_params = self.get_model_info_by_name(model_name)
        return model_params['dataset']['class_names']

    def create_model(self, model_name, trainset, testset, desc=None):
        model_dir = self.path_handler.get_model_dir_by_name(model_name)
        class_names = self.dataset_handler.get_info_by_name(trainset)['class_list']
        os.mkdir(model_dir)

        model_params = self.default_params
        model_params['model_name'] = model_name
        model_params['dataset']['trainset'] = trainset
        model_params['dataset']['testset'] = testset
        model_params['desc'] = desc
        model_params['dataset']['class_names'] = class_names
        model_params['created_date'] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

        model_params_path = self.path_handler.get_model_params_path_by_name(model_name)
        self.cfg_handler.dump_config_by_path(model_params_path, model_params)

        base_config_path = self.path_handler.get_config_path_by_name("base_model")
        config_dir = self.path_handler.get_config_dir_by_name(model_name)
        shutil.copy2(base_config_path, config_dir)
        return True

    def delete_model(self, model_name):
        model_dir = self.path_handler.get_model_dir_by_name(model_name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        model_experiment_dir = self.path_handler.get_model_exp_dir_by_name(model_name)
        if os.path.exists(model_experiment_dir):
            shutil.rmtree(model_experiment_dir)
        return True

    def update_num_epochs(self, model_name, number_epochs):
        model_params_path = self.path_handler.get_model_params_path_by_name(model_name)
        model_params = yaml.safe_load(open(model_params_path, 'r'))
        model_params['train']['total_epochs'] = number_epochs
        self.cfg_handler.dump_config_by_path(model_params_path, model_params)
        return True

    def get_model_info_by_name(self, model_name):
        model_params_path = self.path_handler.get_model_params_path_by_name(model_name)
        model_info = yaml.safe_load(open(model_params_path, 'r'))
        return model_info

    def get_all_general_info(self):
        all_general_info = []

        all_models = os.listdir(self.path_handler.general_config.path.models_dir)
        for mdl in all_models:
            mdl_info = self.get_model_info_by_name(mdl)
            all_general_info.append({
                'model_name': mdl,
                'trainset': mdl_info['dataset']['trainset'],
                'testset': mdl_info['dataset']['testset'],
                'desc': mdl_info['desc'],
                'created_date': mdl_info['created_date']
            })
        return all_general_info

