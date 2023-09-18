from utils.handle_config import ConfigHandler
from utils.handle_path import PathHandler
import os
import shutil
from datetime import datetime
import yaml
import cv2
import json
from tqdm import tqdm

class DatasetHandler:
    def __init__(self):
        self.cfg_handler = ConfigHandler()
        self.general_config = self.cfg_handler.get_general_config()
        self.path_handler = PathHandler()
    
    def create_dataset(self, dataset_name, dataset_secarino, dataset_type, dataset_decs, class_list, augment=0, preparation_progress=1):
        dataset_path = self.path_handler.get_dataset_path_by_name(dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.mkdir(dataset_path)
        
        image_path = self.path_handler.get_image_path_by_name(dataset_name)
        os.mkdir(image_path)

        dataset_info = {
            "dataset_name": dataset_name,
            "dataset_secarino": dataset_secarino,
            "dataset_type": dataset_type,
            "dataset_decs": dataset_decs,
            "class_list": class_list,
            "preparation_progress": preparation_progress,
            "created_date": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            "augment": augment
        }
        dataset_info_path = self.path_handler.get_dataset_info_path_by_name(dataset_name)

        with open(dataset_info_path, 'w') as f:
            yaml.dump(dataset_info, f)
        return True
    
    def update_preparation_progress(self, dataset_name, preparation_progress):
        dataset_info_path = self.path_handler.get_dataset_info_path_by_name(dataset_name)
        dataset_info = yaml.safe_load(open(dataset_info_path, 'r'))

        dataset_info['preparation_progress'] = preparation_progress

        with open(dataset_info_path, 'w') as f:
            yaml.dump(dataset_info, f)
        return True

    def get_nbr_objects(self, dataset_name):
        segment_dir = self.path_handler.get_labelme_segmentation_path(dataset_name)
        nbr_objects = 0
        for segment_file in os.listdir(segment_dir):
            segment_path = os.path.join(segment_dir, segment_file)
            segment_info = json.load(open(segment_path, 'r'))
            nbr_objects+=len(segment_info['shapes'])
        
        objects_dir = self.path_handler.get_object_dir(dataset_name)
        nbr_extracted_objects = len(os.listdir(objects_dir))
        return {
            "nbr_objects": nbr_objects,
            "nbr_extracted_objects": nbr_extracted_objects
        }

    def get_info_by_name(self, dataset_name):
        dataset_info_path = self.path_handler.get_dataset_info_path_by_name(dataset_name)
        dataset_info = yaml.safe_load(open(dataset_info_path, 'r'))
        image_path = self.path_handler.get_image_path_by_name(dataset_name)
        dataset_info['nbr_images'] = len(os.listdir(image_path))

        labelme_path = self.path_handler.get_labelme_annotation_path(dataset_name)
        segment_path = self.path_handler.get_labelme_segmentation_path(dataset_name)
        dataset_info['nbr_auto_annotated'] = len(os.listdir(labelme_path))
        dataset_info['nbr_segmented'] = len(os.listdir(segment_path))

        if dataset_info['preparation_progress'] == 1:
            dataset_info['upload_dataset_progress'] = "Not Yet"
            dataset_info['annotation_progress'] = "Not Yet"
            dataset_info['augment_progress'] = "Not Yet"
        elif dataset_info['preparation_progress'] == 2:
            dataset_info['upload_dataset_progress'] = "Done"
            dataset_info['annotation_progress'] = "Not Yet"
            dataset_info['augment_progress'] = "Not Yet"
        elif dataset_info['preparation_progress'] == 3:
            dataset_info['upload_dataset_progress'] = "Done"
            dataset_info['annotation_progress'] = "Done"
            dataset_info['augment_progress'] = "Not Yet"
        elif dataset_info['preparation_progress'] == 3:
            dataset_info['upload_dataset_progress'] = "Done"
            dataset_info['annotation_progress'] = "Done"
            dataset_info['augment_progress'] = "Done"

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
    
    def extract_images(self, dataset_name, video_name, video_fps):
        dataset_path = self.path_handler.get_dataset_path_by_name(dataset_name)
        video_path = os.path.join(dataset_path, video_name)
        image_path = self.path_handler.get_image_path_by_name(dataset_name)
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        ratio = int(fps/video_fps)

        ratio = 1 if ratio==0 else ratio
        count = 0
        for i in tqdm(range(length)):
            success,image = vidcap.read()
            if i%ratio == 0:
                cv2.imwrite(os.path.join(image_path, "img_{:06d}.jpg".format(count)), image)
                count+=1



