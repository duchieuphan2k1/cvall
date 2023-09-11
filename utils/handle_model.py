from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
import os

class ModelHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.general_config = self.cfg_handler.get_general_config()
        self.path_handler = PathHandler()