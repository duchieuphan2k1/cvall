import numpy as np
import os
import shutil
import json
import cv2
from tqdm import tqdm
from src.segmentation_models.FastSAM2.fastsam import FastSAM, FastSAMPrompt
from src.controller.handle_path import PathHandler

class FastFAM_Infer:
    def __init__(self):
        self.model = FastSAM('data/default_weights/FastSAM2/FastSAM-x.pt')
        self.path_handler = PathHandler()
        self.DEVICE = 'cuda'

    def run_image(self, image, bboxes: list, plot=False, output_path=None):
        everything_results = self.model(image, device=self.DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image, everything_results, device=self.DEVICE)
        # everything prompt
        ann = prompt_process.everything_prompt()
        # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bboxes=bboxes)#.astype(np.uint8)

        if plot and output_path:
            prompt_process.plot(annotations=ann,output_path=output_path, mask_random_color=False)
            return output_path       

        return ann 

    def run_image_path(self, image_path, bboxes: list):
        everything_results = self.model(image_path, device=self.DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.DEVICE)
        # everything prompt
        ann = prompt_process.everything_prompt()
        # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bboxes=bboxes)#.astype(np.uint8)
        return ann

    def run_dataset(self, dataset_name):
        labelme_segmentation_path = self.path_handler.get_labelme_segmentation_path(dataset_name)
        if os.path.exists(labelme_segmentation_path):
            shutil.rmtree(labelme_segmentation_path)
        os.mkdir(labelme_segmentation_path)

        image_dir = self.path_handler.get_image_path_by_name(dataset_name)
        labelme_annotation_path = self.path_handler.get_labelme_annotation_path(dataset_name)

        all_images = os.listdir(image_dir)
        for image_name in tqdm(all_images):
            image_name = image_name.replace(".jpg", "")

            image_path = os.path.join(image_dir, image_name + ".jpg")
            image_annot_path = os.path.join(labelme_annotation_path, image_name + ".json")
            image_segment_path = os.path.join(labelme_segmentation_path, image_name + ".json")
            
            image_annotations = json.load(open(image_annot_path, 'r'))
            object_annots = image_annotations['shapes']

            bboxes = []
            labels = []
            for obj in object_annots:
                points = obj['points']
                bbox = [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]
                bboxes.append(bbox)
                labels.append(obj['label'])

            if len(bboxes) == 0:
                template = image_annotations
                template['shapes'] = []

                with open(image_segment_path, 'w') as f:
                    f.write(json.dumps(template))
                continue
            
            anns = self.run_image_path(image_path, bboxes)

            print(len(anns))
            print(len(labels))
            all_objects = []
            for i in range(len(anns)):
                mask = anns[i]
                label_name = labels[i]
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                p = [c[0].tolist() for c in max(contours, key = cv2.contourArea)]
                p.append(p[0])
                segment_annot = {
                    "label": label_name,
                    "points": p,
                    "group_id": None,
                    'description': "",
                    "shape_type": "polygon",
                    "flags": {}
                }
                all_objects.append(segment_annot)
            
            template = image_annotations
            template['shapes'] = all_objects

            with open(image_segment_path, 'w') as f:
                f.write(json.dumps(template))
        
        return True

    def extract_objects(self, dataset_name):
        segment_ann_path = self.path_handler.get_labelme_segmentation_path(dataset_name)
        object_dir = self.path_handler.get_object_dir(dataset_name)
        
        if os.path.exists(object_dir):
            shutil.rmtree(object_dir)
        os.mkdir(object_dir)

        annot_list = os.listdir(segment_ann_path)
        image_dir = self.path_handler.get_image_path_by_name(dataset_name)

        for annot_file in tqdm(annot_list):
            img_name = annot_file.replace('json', 'jpg')
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)

            annot_path = os.path.join(segment_ann_path, annot_file)
            gt = json.load(open(annot_path, 'r'))
            image_height = gt['imageHeight']
            image_width = gt['imageWidth']
            i = 0
            for obj in gt['shapes']:
                mask = np.zeros((image_height, image_width))
                mask = cv2.fillPoly(mask, pts=[np.array(obj["points"]).astype(int)], color=1)
                obj_name = "{}_{}_{:06d}.png".format(obj['label'], img_name.replace(".jpg", ""), i)
                i+=1
                obj_path = os.path.join(object_dir, obj_name)

                obj_img = img.copy()
                obj_img[mask==0, ...] = 0

                mask = np.expand_dims(mask, axis=2)
                obj_img = np.concatenate([obj_img, mask*255], axis=2).astype(np.uint8)

                x_list, y_list, _ = np.where(obj_img>0)
                xmin, ymin, xmax, ymax = min(x_list), min(y_list), max(x_list), max(y_list)
                obj = obj_img[xmin:xmax, ymin:ymax, ...]
            
                # break
                obj = obj[..., [2,1,0,3]]
                cv2.imwrite(obj_path, cv2.cvtColor(obj, cv2.COLOR_RGBA2BGRA))

        return True
        


