from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
import os
import yaml

class ModelHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.general_config = self.cfg_handler.get_general_config()
        self.path_handler = PathHandler()

    def get_model_info_by_name(self, model_name):
        params_path = self.path_handler.get_model_params_path_by_name(model_name)
        model_params = yaml.safe_load(open(params_path, 'r'))
        return model_params
    
    def get_classes_by_name(self, model_name):
        model_params = self.get_model_info_by_name(model_name)
        return model_params['dataset']['class_names']