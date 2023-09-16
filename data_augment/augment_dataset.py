import os
import shutil
import cv2
import random
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from utils.handle_path import PathHandler
from utils.handle_dataset import DatasetHandler
import imgaug as ia
import imgaug.augmenters as iaa

class DatasetAugment:
    def __init__(self):
        self.path_handler = PathHandler()
        self.dataset_handler = DatasetHandler()

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
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.Rotate((-45, 45)),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
            iaa.PiecewiseAffine(scale=(0.01, 0.05))
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
        small_scale = [5, 7]
        medium_scale = [7, 25]
        big_scale = [25, 50]

        small_percent = 40
        medium_percent = 50

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
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=False),
            iaa.ChangeColorTemperature((1100, 5000)),
            iaa.ChannelShuffle(0.35),
            iaa.Add((-30, 30), per_channel=False),
            iaa.GammaContrast((0.7, 1.3)),
            iaa.MultiplyBrightness((0.5, 1.5))
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
            objaug_img = self.augment_objects_color(objaug_img)
            obj_img[:, :, :3] = objaug_img

            obj_img = self.augment_objects_geometry(obj_img)
            obj_img = Image.fromarray(np.uint8(obj_img))
            obj_img = self.trim(obj_img)

            if obj_img == None:
                continue

            
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
                    if self.check_overlap(bbox, c_bbox, accepted_threshold=50):
                        overlap = True
                
                if not overlap:
                    bboxes.append(bbox)
                    break

            if overlap:
                continue

            bg_img.paste(obj_img, (x, y), obj_img)
            bg_base_name = os.path.basename(obj)
            class_name = bg_base_name.split("_")[0]
            shape = [(x, y), (x + obj_w, y + obj_h)]

            annots.append([class_name, shape])

        bg_img = bg_img.convert('RGB')
        return bg_img, annots

    def augment_dataset(self, augment_dataset_name, origin_dataset_name, background_set_name):
        origin_info = self.dataset_handler.get_info_by_name(origin_dataset_name)
        self.dataset_handler.create_dataset(augment_dataset_name, origin_info['dataset_secarino'], origin_info['dataset_type'], origin_info['dataset_decs'], origin_info['class_list'], augment=1)
        augment_dir = self.path_handler.get_dataset_path_by_name(augment_dataset_name)
        origin_dir = self.path_handler.get_dataset_path_by_name(origin_dataset_name)
        background_dir = self.path_handler.get_background_set_by_name(background_set_name)

        object_dir = self.path_handler.get_object_dir(origin_dataset_name)
        augment_image_dir = self.path_handler.get_image_path_by_name(augment_dataset_name)

        number_augment = 200
        nbr_objects_per_image = [10, 20]
        i = 0
        all_backgrounds = os.listdir(background_dir)
        all_objects = os.listdir(object_dir)

        while i<number_augment:
            for bg_name in all_backgrounds:
                nbr_objects = random.randint(nbr_objects_per_image[0], nbr_objects_per_image[1])
                bg_path = os.path.join(background_dir, bg_name)
                saved_name = "{:6d}.jpg".format(i)

                selected_objects = [random.choice(all_objects) for c in range(nbr_objects)]
                selected_objects_path  = [os.path.join(object_dir, s) for s in selected_objects]

                bg_img, annots = self.paste_objects(bg_path, selected_objects_path)

                bg_img.save(os.path.join(augment_image_dir, saved_name))
                i+=1

    