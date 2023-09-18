import os
import yaml
from easydict import EasyDict as edict

class ConfigHandler:
    def __init__(self):
        self.general_config = yaml.safe_load(open("all_configs/general_config.yaml", 'r'))
        self.default_data_augment = yaml.safe_load(open("all_configs/default_data_augment.yaml", 'r'))

    def get_general_config(self):
        return edict(self.general_config)
    
    def load_default_augment(self, dict=False):
        if dict:
            return self.default_data_augment
        return edict(self.default_data_augment)
            
    def load_config_by_path(self, path):
        cfg = yaml.safe_load(open(path, 'r'))
        return edict(cfg)
    
    def dump_config_by_path(self, path, json_object):
        with open(path, 'w') as f:
            yaml.dump(json_object, f)
        return True
