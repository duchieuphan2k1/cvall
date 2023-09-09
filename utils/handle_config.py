import yaml
from easydict import EasyDict as edict

class ConfigHandler:
    def __init__(self):
        self.general_config = yaml.safe_load(open("all_configs/general_config.yaml", 'r'))
    
    def get_general_config(self):
        return edict(self.general_config)
            