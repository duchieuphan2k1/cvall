from src.controller.handle_config import ConfigHandler
from src.controller.handle_path import PathHandler
import os
import shutil
from datetime import datetime
import numpy as np
import yaml
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
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
    
    def plot_info(self, dataset_name):
        dataset_info = self.get_info_by_name(dataset_name)
        class_list = dataset_info['class_list']
        class_nbr_info = {}
        object_image_ratio = []
        for class_name in class_list:
            class_nbr_info[class_name] = 0
        
        # segment_dir = self.path_handler.get_labelme_segmentation_path(dataset_name)
        # for segment_file in os.listdir(segment_dir):
        #     segment_path = os.path.join(segment_dir, segment_file)
        #     segment_info = json.load(open(segment_path, 'r'))
        #     img_area = segment_info['imageHeight']*segment_info['imageWidth']
        #     for obj in segment_info['shapes']:
        #         class_nbr_info[obj['label']]+=1
        #         obj_area = cv2.contourArea(np.array([obj['points']]).astype(int))
        #         object_image_ratio.append(round(np.sqrt(obj_area/img_area), 3))

        annot_dir = self.path_handler.get_labelme_annotation_path(dataset_name)
        for annot_file in os.listdir(annot_dir):
            annot_path = os.path.join(annot_dir, annot_file)
            segment_info = json.load(open(annot_path, 'r'))
            img_area = segment_info['imageHeight']*segment_info['imageWidth']
            for obj in segment_info['shapes']:
                class_nbr_info[obj['label']]+=1
                obj_area = (obj['points'][1][0] - obj['points'][0][0])*(obj['points'][1][1] - obj['points'][0][1])
                object_image_ratio.append(round(np.sqrt(obj_area/img_area), 3))

        keys = list(class_nbr_info.keys())
        values = list(class_nbr_info.values())
        fig = plt.figure()
        # creating the bar plot
        plt.bar(keys, values)

        plt.xlabel("Class Name")
        plt.ylabel("Number of Objects")
        plt.title("Number of objects per class")
        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(values):
            plt.text(xlocs[i], v, str(v))
        
        number_plot_path = self.path_handler.get_class_number_plot_path(dataset_name)
        plt.savefig(number_plot_path)
        plt.close()

        plt.hist(object_image_ratio)
        plt.xlabel("Object Size/ Image Size Ratio")
        plt.ylabel("Number of Ratios")
        plt.title("Object Size/ Image Size Distribution")
        size_plot_path = self.path_handler.get_size_plot_path(dataset_name)
        plt.savefig(size_plot_path)
        plt.close()

        return number_plot_path, size_plot_path

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
    
    def get_all_trainset(self):
        all_info =self.get_all_info()
        all_trainsets = []
        for set_info in all_info:
            if set_info['dataset_type'] == "train":
                all_trainsets.append(set_info['dataset_name'])
        return all_trainsets

    def get_all_testset(self):
        all_info =self.get_all_info()
        all_testsets = []
        for set_info in all_info:
            if set_info['dataset_type'] == "test":
                all_testsets.append(set_info['dataset_name'])
        return all_testsets

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

    def labelme_to_coco(self, dataset_name):
        coco_template = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        class_list = self.get_info_by_name(dataset_name)['class_list']

        class_index = 0
        for class_name in class_list:
            coco_template["categories"].append({"supercategory": class_name,"id": class_index,"name": class_name})
            class_index+=1

        labelme_dir = self.path_handler.get_labelme_annotation_path(dataset_name)
        images_dir = self.path_handler.get_image_path_by_name(dataset_name)
        coco_file_path = self.path_handler.get_coco_annotation_file(dataset_name)

        all_images = os.listdir(images_dir)
        image_id = 0
        box_id = 0
        for image_name in all_images:
            img = cv2.imread(os.path.join(images_dir, image_name))
            img_annot = json.load(open(os.path.join(labelme_dir, image_name.replace(".jpg", ".json"))))
            h, w, _ = img.shape

            coco_template['images'].append({
                "file_name": image_name,
                "height": h,
                "width": w,
                "id": image_id
            })

            for box_info in img_annot['shapes']:
                box = box_info['points']
                x1, y1 = box[0][0], box[0][1]
                box_width = box[1][0] - x1
                box_height = box[1][1] - y1
                class_name = box_info['label']

                coco_template['annotations'].append({
                    "segmentation": [],
                    "area": box_width*box_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x1,y1,box_width,box_height],
                    "category_id": class_list.index(class_name),
                    "id": box_id
                })
                box_id+=1
            image_id+=1
        
        with open(coco_file_path, 'w') as f:
            json.dump(coco_template, f)

if __name__ == "__main__":
    dataset_handler = DatasetHandler()
    dataset_handler.plot_info('test01')

