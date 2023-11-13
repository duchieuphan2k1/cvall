from damo_yolo2.tools.demo import InferRunner
from utils.handle_path import PathHandler
from utils.handle_dataset import DatasetHandler
from utils.handle_model import ModelHandler
import os
import shutil
import json
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt

class PredictDataset():
    def __init__(self, dataset_name, model_name, generate_annotation=False):
        self.path_handler = PathHandler()
        self.dataset_handler = DatasetHandler()
        self.model_handler = ModelHandler()

        self.model_name = model_name
        self.class_list = self.dataset_handler.get_info_by_name(dataset_name)['class_list']
        
        model_config = self.path_handler.get_config_path_by_name(model_name)
        model_ckpt = self.path_handler.get_ckpt_path_by_name(model_name)
        self.model_config = model_config
        self.infer_runner = InferRunner(model_config, model_ckpt)
        
        self.image_folder = self.path_handler.get_image_path_by_name(dataset_name)
        self.result_file_path = self.path_handler.get_result_file_path(self.model_name, dataset_name)
        self.precision_plot_dir = self.path_handler.get_precision_plot_dir(self.model_name, dataset_name)
        self.generate_annotation = generate_annotation
        if generate_annotation:
            self.labelme_output_folder = self.path_handler.get_labelme_annotation_path(dataset_name)
        else:
            self.labelme_output_folder = self.path_handler.get_pred_labelme_dir(model_name, dataset_name)
            self.vis_output_folder = self.path_handler.get_pred_vis_dir(model_name, dataset_name)

            self.labelme_annot_folder = self.path_handler.get_labelme_annotation_path(dataset_name)
        
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

    def normalize_classes(self, classes):
        normalized_classes = []
        for class_name in classes:
            normalized_classes.append( '_'.join(class_name.split(" ")))
        return normalized_classes
    
    def runs(self, classes, changed_names):
        if os.path.exists(self.labelme_output_folder):
            shutil.rmtree(self.labelme_output_folder)
        os.mkdir(self.labelme_output_folder)

        classes = self.normalize_classes(classes)

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
            for i in range(len(bboxes)):
                bbox = bboxes[i].tolist()
                scr = scores[i].tolist()
                class_name = self.infer_runner.infer_engine.class_names[int(cls_inds[i].item())]

                if len(classes) != 0:
                    if class_name not in classes:
                        continue

                class_index = classes.index(class_name)
                class_name = changed_names[class_index]
                box_annot = {
                    'label': class_name,
                    'points': [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    'group_id': None,
                    'description': "",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                shapes.append(box_annot)
            labelme_img['shapes'] = shapes

            json_filename = image_name.replace('.jpg', '.json')
            with open(os.path.join(self.labelme_output_folder, json_filename), 'w') as f:
                f.write(json.dumps(labelme_img))

    def run_pred(self, ckpt_path=None):
        if os.path.exists(self.labelme_output_folder):
            shutil.rmtree(self.labelme_output_folder)
        os.mkdir(self.labelme_output_folder)

        if ckpt_path!=None:
            self.infer_runner = InferRunner(self.model_config, ckpt_path)
        # classes = self.model_handler.get_classes_by_name(self.model_name)

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
            bboxes= bboxes.to(torch.int)
            shapes = []
            for i in range(len(bboxes)):
                bbox = bboxes[i].tolist()
                scr = scores[i].tolist()
                class_name = self.infer_runner.infer_engine.class_names[int(cls_inds[i].item())]
                

                # if len(classes) != 0:
                #     if class_name not in classes:
                #         continue

                box_annot = {
                    'label': class_name,
                    'points': [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    'group_id': None,
                    'description': "",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                # cv2.putText(origin_img,"{}: {}".format(class_name, scr),(bbox[0],bbox[1]+10),0,0.3,(0,255,0))
                # cv2.rectangle(origin_img,(bbox[0],bbox[1]),(bbox[2], bbox[3]),(0,255,0),2)
                shapes.append(box_annot)
            labelme_img['shapes'] = shapes
            # if not self.generate_annotation:
            #     cv2.imwrite(os.path.join(self.vis_output_folder, image_name), origin_img)
            json_filename = image_name.replace('.jpg', '.json')
            with open(os.path.join(self.labelme_output_folder, json_filename), 'w') as f:
                f.write(json.dumps(labelme_img))
    
    def cal_iou(self, gt_bb, pred_bb):
        bi = [max(pred_bb[0], gt_bb[0]), max(pred_bb[1], gt_bb[1]), min(pred_bb[2], gt_bb[2]), min(pred_bb[3], gt_bb[3])]
        iw = bi[2] - bi[0] + 1
        ih = bi[3] - bi[1] + 1
        ov = 0
        if iw > 0 and ih > 0:
            # compute IoU = area of intersect / area of union
            ua = (pred_bb[2] - pred_bb[0]  +1) * (pred_bb[3] - pred_bb[1] + 1) + \
                    (gt_bb[2] - gt_bb[0] + 1) * (gt_bb[3] - gt_bb[1] + 1) - iw * ih
            ov = iw * ih / ua
        return ov 
    
    def select_bbox(self, all_gts_bb, pred_bb, thres=0.5):
        ious = []
        for gt_bb in all_gts_bb:
            iou = self.cal_iou(gt_bb, pred_bb)
            if iou>thres:
                ious.append(iou)
        
        if len(ious) == 0:
            return None
        
        return ious.index(max(ious))
    
    def get_bboxes_info(self, data):
        bboxes = []
        labels = []
        for dt in data:
            xmin, ymin = dt['points'][0][0],  dt['points'][0][1]
            xmax, ymax = dt['points'][1][0],  dt['points'][1][1]
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(dt['label'])
        return bboxes, labels
    
    def eval_file(self, annot_data, pred_data, all_results):
        annot_bboxes, annot_labels = self.get_bboxes_info(annot_data)
        pred_bboxes, pred_labels = self.get_bboxes_info(pred_data)
        num_preded = len(pred_bboxes)
        num_annot = len(annot_bboxes)
        for i in range(num_preded):
            bbox = pred_bboxes[i]
            label = pred_labels[i]
            if label not in self.class_list:
                continue
            all_results[label]['num_pred']+=1
            gt_index = self.select_bbox(annot_bboxes, bbox)
            if gt_index is None:
                all_results[label]['total_excessive']+=1

        for i in range(num_annot):
            bbox = annot_bboxes[i]
            label = annot_labels[i]
            all_results[label]['num_gt']+=1
            if label not in self.class_list:
                continue
            pred_index = self.select_bbox(pred_bboxes, bbox)
            if pred_index is None:
                all_results[label]['total_missing']+=1
            elif label!=pred_labels[pred_index]:
                all_results[label]['total_wrong']+=1
            else:
                all_results[label]['num_true']+=1
        
        return all_results

    def get_accuracy(self, all_results:dict):
        precision_list = []
        for class_name in all_results.keys():
            if all_results[class_name]['num_gt'] == 0:
                precision_list.append(1)
                continue

            class_precision = all_results[class_name]['num_true']/all_results[class_name]['num_gt']
            precision_list.append(class_precision)
        return round(np.average(precision_list), 4)

    def eval_results(self, return_accuracy=False, plot=False):
        all_results = {}
        for class_name in self.class_list:
            all_results[class_name] = {
            'num_gt': 0,
            'num_pred': 0,
            'num_true': 0,
            'total_excessive' : 0,
            'total_missing' : 0,
            'total_wrong' : 0}
        for file_name in os.listdir(self.labelme_annot_folder):
            annot_path = os.path.join(self.labelme_annot_folder, file_name)
            pred_path = os.path.join(self.labelme_output_folder, file_name)
            annot_data = json.load(open(annot_path, 'r'))['shapes']
            pred_data = json.load(open(pred_path, 'r'))['shapes']
            all_results = self.eval_file(annot_data, pred_data, all_results)

        print(all_results)
        avg_precision = self.get_accuracy(all_results)
        all_results['avg_precision'] = avg_precision
        with open(self.result_file_path, 'w') as f:
            json.dump(all_results, f)
        if plot:
            self.plot_result(all_results)

        if return_accuracy:
            return avg_precision
        return all_results
    
    def plot_result(self, all_results:dict):
        for class_name in all_results:
            if class_name == 'avg_precision':
                continue
            num_wrong_pred = all_results[class_name]['total_wrong']
            num_excessive = all_results[class_name]['total_excessive']
            num_missing = all_results[class_name]['total_missing']
            num_true = all_results[class_name]['num_true']

            label = ["wrong: {}".format(num_wrong_pred), 
                    "excessive: {}".format(num_excessive), 
                    "missing: {}".format(num_missing),
                    "true: {}".format(num_true)]
            errs = np.array([num_wrong_pred, num_excessive, num_missing, num_true])

            if all_results[class_name]['num_gt'] == 0:
                cls_acc = 1
            else:
                cls_acc = round((all_results[class_name]['num_true'])/(all_results[class_name]['num_gt']), 4)

            if num_wrong_pred!=0 or num_excessive!=0 or num_missing!=0:
                plt.title("Acurracy and Error Analysis of class {}".format(class_name))
                plt.pie(errs, labels = label, autopct='%1.1f%%')
                plt.legend(title = "Total GT: {} objects \nTotal Pred: {} objects\nAccuracy in classification: {}".format(all_results[class_name]['num_gt'], all_results[class_name]['num_pred'], cls_acc), loc='upper left')
                plt.savefig(os.path.join(self.precision_plot_dir, "{}.jpg".format(class_name)))
                plt.close()
        return

if __name__ == "__main__":
    pdt = PredictDataset("test01", "first_demo_model", generate_annotation=True)
    pdt.runs(['car', 'person'])