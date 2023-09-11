from damo_yolo2.tools.demo import InferRunner
from utils.handle_path import PathHandler
from utils.handle_dataset import DatasetHandler
from utils.handle_model import ModelHandler
import os
import json
import numpy as np
from PIL import Image

class PredictDataset():
    def __init__(self, dataset_name, model_name, generate_annotation=False):
        self.path_handler = PathHandler()
        self.dataset_handler = DatasetHandler()
        self.model_handler = ModelHandler()
        
        model_config = self.path_handler.get_config_path_by_name(model_name)
        model_ckpt = self.path_handler.get_ckpt_path_by_name(model_name)
        self.infer_runner = InferRunner(model_config, model_ckpt)
        
        self.image_folder = self.path_handler.get_image_path_by_name(dataset_name)

        if generate_annotation:
            self.labelme_annot_folder = self.path_handler.get_labelme_annotation_path(dataset_name)
            self.coco_annot_file = self.path_handler.get_coco_annotation_file(dataset_name)
        
        self.labelme_template =  {
                "version": "5.2.1",
                "flags": {},
                "shapes":[],
                "imagePath": "",
                "imageData": None,
                "imageHeight": 0,
                "imageWidth": 0
            }
        self.coco_template = {}

    def runs(self):
        all_images = os.listdir(self.image_folder)
        for image_name in all_images:
            labelme_img = self.labelme_template
            image_path = os.path.join(self.image_folder, image_name)
            origin_img = np.asarray(Image.open(image_path).convert('RGB'))
            h, w, _ = origin_img.shape
            labelme_img['imagePath'] = "../{}/{}".format(self.path_handler.general_config.path.image_dir_name, image_name)
            labelme_img['imageHeight'] = h
            labelme_img['imageWidth'] = w

            bboxes, scores, cls_inds = self.infer_runner.predict(origin_img)
            shapes = []
            ['label', 'points', 'group_id', 'description', 'shape_type', 'flags']
            for i in range(len(bboxes)):
                bbox = bboxes[i].tolist()
                scr = scores[i].tolist()
                class_name = self.infer_runner.infer_engine.class_names[int(cls_inds[i].item())]
                box_annot = {
                    'label': '_'.join(class_name.split(" ")),
                    'points': [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    'group_id': None,
                    'description': "",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                shapes.append(box_annot)
            labelme_img['shapes'] = shapes

            json_filename = image_name.replace('.jpg', '.json')
            with open(os.path.join(self.labelme_annot_folder, json_filename), 'w') as f:
                f.write(json.dumps(labelme_img))

if __name__ == "__main__":
    pdt = PredictDataset("test01", "first_demo_model", generate_annotation=True)
    pdt.runs()