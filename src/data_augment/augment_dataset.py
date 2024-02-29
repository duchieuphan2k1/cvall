import os
import shutil
import cv2
import json
import random
from PIL import Image, ImageOps, ImageChops
import numpy as np
from utils.handle_path import PathHandler
from utils.handle_dataset import DatasetHandler
from utils.handle_config import ConfigHandler
import imgaug as ia
import imgaug.augmenters as iaa

class DatasetAugment:
    def __init__(self):
        self.path_handler = PathHandler()
        self.dataset_handler = DatasetHandler()
        self.config_handler = ConfigHandler()
        self.augment_config = ConfigHandler().load_default_augment()

    def trim(self, im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def check_overlap(self, bboxA, bboxB, accepted_threshold = 0):
        xA = max(bboxA[0], bboxB[0])
        yA = max(bboxA[1], bboxB[1])
        xB = min(bboxA[2], bboxB[2])
        yB = min(bboxA[3], bboxB[3])
        
        # Compute the area of intersection rectangle
        interArea = (xB - xA)*(yB - yA)
        if interArea <= accepted_threshold:
            return False
        else:
            if xB < xA or yB < yA:
                return False
            return True

    def augment_objects_geometry(self, object_img):
        h, w, channel = object_img.shape
        padded_img = np.zeros((h+h, w+w, channel))
        padded_img[h//2:h//2+h, w//2:w//2+w, :] = object_img

        object_img = np.expand_dims(padded_img, axis=0)

        aug = iaa.Sequential([
            iaa.PerspectiveTransform(scale=self.augment_config.geometry.PerspectiveTransform),
            iaa.Rotate(self.augment_config.geometry.Rotate),
            iaa.ShearX(self.augment_config.geometry.ShearX),
            iaa.ShearY(self.augment_config.geometry.ShearY),
            iaa.PiecewiseAffine(self.augment_config.geometry.PiecewiseAffine)
        ])

        object_img = aug(images=object_img)
        object_img = object_img[0]

        return object_img
    
    def get_size_scale(self, scale, image_size):
        min_scale = image_size*scale[0]//100
        max_scale = image_size*scale[1]//100
        return(min_scale, max_scale)

    def random_resize(self, obj_img, image_size):
        scale = np.random.randint(0, 100)
        small_scale = self.augment_config.resize.small_scale
        medium_scale = self.augment_config.resize.medium_scale
        big_scale = self.augment_config.resize.big_scale

        small_percent = self.augment_config.resize.small_percent
        medium_percent = self.augment_config.resize.medium_percent

        if scale <= small_percent:
            size_scale = self.get_size_scale(small_scale, image_size)

        elif scale <= small_percent + medium_percent:
            size_scale = self.get_size_scale(medium_scale, image_size)
        else:
            size_scale = self.get_size_scale(big_scale, image_size)

        obj_scale = np.random.randint(size_scale[0], size_scale[1])
        w, h  = obj_img.size

        newsize = (obj_scale, int((obj_scale/w)*h)) if w<h else (int((obj_scale/h)*w), obj_scale)

        if newsize[0] < image_size and newsize[1] < image_size:
            obj_img = obj_img.resize(newsize)

        return obj_img

    def augment_objects_color(self, object_img):
        object_img = np.expand_dims(object_img, axis=0)
        seq = iaa.Sequential([
            iaa.MultiplyHueAndSaturation(self.augment_config.color.MultiplyHueAndSaturation, per_channel=False),
            iaa.ChangeColorTemperature(self.augment_config.color.ChangeColorTemperature),
            iaa.ChannelShuffle(self.augment_config.color.ChannelShuffle),
            iaa.Add(self.augment_config.color.Add, per_channel=False),
            iaa.GammaContrast(self.augment_config.color.GammaContrast),
            iaa.MultiplyBrightness(self.augment_config.color.MultiplyBrightness)
        ], random_order=True
        )

        image_aug = seq(images=object_img)
        return image_aug[0]


    def paste_objects(self, bg_path, objects_path:list):
        bg_img = cv2.imread(bg_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGBA)
        
        h, w, _ = bg_img.shape
        bg_img = Image.fromarray(np.uint8(bg_img))
        
        annots = []
        bboxes = []
        for obj in objects_path:
            obj_img = cv2.imread(obj, cv2.IMREAD_UNCHANGED)
            objaug_img = cv2.cvtColor(obj_img, cv2.COLOR_BGRA2RGB)

            if self.augment_config.other.use_color:
                objaug_img = self.augment_objects_color(objaug_img)

            obj_img[:, :, :3] = objaug_img

            if self.augment_config.other.use_geometry:
                obj_img = self.augment_objects_geometry(obj_img)

            obj_img = Image.fromarray(np.uint8(obj_img))
            obj_img = self.trim(obj_img)

            if obj_img == None:
                continue

            if self.augment_config.other.use_resize:
                obj_img = self.random_resize(obj_img, w)

            obj_w, obj_h = obj_img.size
            
            if obj_w >= w or obj_h >= h:
                continue
            
            retry = 30
            overlap = False

            for j in range(retry):
                x = np.random.randint(0, w-obj_w)
                y = np.random.randint(0, h-obj_h)
                bbox = [x, y, x+obj_w, y+obj_h]
                for c_bbox in bboxes:
                    if self.check_overlap(bbox, c_bbox, accepted_threshold=self.augment_config.other.accepted_threshold):
                        overlap = True
                
                if not overlap:
                    bboxes.append(bbox)
                    break

            if overlap:
                continue

            bg_img.paste(obj_img, (x, y), obj_img)
            bg_base_name = os.path.basename(obj)
            class_name = bg_base_name.split("_")[0]
            shape = [[x, y], [x + obj_w, y + obj_h]]

            annots.append({
            "label": class_name,
            "points": shape,
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
            })

        bg_img = bg_img.convert('RGB')

        print(annots)
        return bg_img, annots

    def change_augment_config(self, augment_dataset_name, new_augment_config):
        yaml_path = self.path_handler.get_augment_config_path(augment_dataset_name)
        self.config_handler.dump_config_by_path(yaml_path, new_augment_config)
        return True

    def load_augment_config_by_name(self, augment_dataset_name):
        augment_config_path = self.path_handler.get_augment_config_path(augment_dataset_name)
        if os.path.exists(augment_config_path):
            return self.config_handler.load_config_by_path(augment_config_path)
        else:
            return self.config_handler.load_default_augment()

    def create_augment_dataset(self, origin_dataset_name, augment_dataset_name):
        origin_info = self.dataset_handler.get_info_by_name(origin_dataset_name)
        self.dataset_handler.create_dataset(augment_dataset_name, origin_info['dataset_secarino'], origin_info['dataset_type'], origin_info['dataset_decs'], origin_info['class_list'], augment=1, preparation_progress=4)

    def augment_dataset(self, augment_dataset_name, origin_dataset_name, background_set_name):
        
        background_dir = self.path_handler.get_background_set_by_name(background_set_name)
        object_dir = self.path_handler.get_object_dir(origin_dataset_name)
        augment_image_dir = self.path_handler.get_image_path_by_name(augment_dataset_name)

        yaml_path = self.path_handler.get_augment_config_path(augment_dataset_name)
        print(yaml_path)
        if os.path.exists(yaml_path):
            self.augment_config = self.config_handler.load_config_by_path(yaml_path)
        else:
            self.config_handler.dump_config_by_path(yaml_path, self.config_handler.load_default_augment(dict=True))

        number_augment = self.augment_config.other.number_augment
        nbr_objects_per_image = self.augment_config.other.nbr_objects_per_image
        i = 0
        all_backgrounds = os.listdir(background_dir)
        all_objects = os.listdir(object_dir)

        gray_nb = int(1/self.augment_config.other.grayscale_ratio)
        labelme_annot_dir = self.path_handler.get_labelme_annotation_path(augment_dataset_name)
        while i<number_augment:
            for bg_name in all_backgrounds:
                nbr_objects = random.randint(nbr_objects_per_image[0], nbr_objects_per_image[1])
                bg_path = os.path.join(background_dir, bg_name)
                saved_name = "{:06d}.jpg".format(i)
                labelme_name = saved_name.replace(".jpg", ".json")

                selected_objects = [random.choice(all_objects) for c in range(nbr_objects)]
                selected_objects_path  = [os.path.join(object_dir, s) for s in selected_objects]

                bg_img, annots = self.paste_objects(bg_path, selected_objects_path)
                bg_w, bg_h = bg_img.size
                image_fol = self.path_handler.general_config.path.image_dir_name

                labelme_template =  {
                    "version": "5.2.1",
                    "flags": {},
                    "shapes":annots,
                    "imagePath": "../{}/{}".format(image_fol, saved_name),
                    "imageData": None,
                    "imageHeight": bg_h,
                    "imageWidth": bg_w
                }

                with open(os.path.join(labelme_annot_dir, labelme_name), 'w') as f:
                    f.write(json.dumps(labelme_template))

                if i%gray_nb == 0:
                    bg_img= ImageOps.grayscale(bg_img)

                bg_img.save(os.path.join(augment_image_dir, saved_name))
                i+=1

    