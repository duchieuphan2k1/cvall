from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
import os
import shutil
from datetime import datetime
import yaml

class DatasetHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.general_config = self.cfg_handler.get_general_config()
        self.path_handler = PathHandler()
    
    def create_dataset(self, dataset_name, dataset_secarino, dataset_type, dataset_decs, class_list):
        dataset_path = self.path_handler.get_dataset_path_by_name(dataset_name)
        os.mkdir(dataset_path)
        image_path = self.path_handler.get_image_path_by_name(dataset_name)
        os.mkdir(image_path)

        dataset_info = {
            "dataset_name": dataset_name,
            "dataset_secarino": dataset_secarino,
            "dataset_type": dataset_type,
            "dataset_decs": dataset_decs,
            "class_list": class_list,
            "preparation_progress": 1,
            "created_date": datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        }
        dataset_info_path = self.path_handler.get_dataset_info_path_by_name(dataset_name)

        with open(dataset_info_path, 'w') as f:
            yaml.dump(dataset_info, f)
        return True

    def get_info_by_name(self, dataset_name):
        dataset_info_path = self.path_handler.get_dataset_info_path_by_name(dataset_name)
        dataset_info = yaml.safe_load(open(dataset_info_path, 'r'))
        return dataset_info
    
    def get_all_info(self):
        all_info = []
        datasets_dir = self.general_config.path.datasets_dir
        all_datasets = os.listdir(datasets_dir)
        for dataset_name in all_datasets:
            all_info.append(self.get_info_by_name(dataset_name))
        
        return all_info

    def delete_dataset(self, dataset_name):
        dataset_path = self.path_handler.get_dataset_path_by_name(dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

        return True