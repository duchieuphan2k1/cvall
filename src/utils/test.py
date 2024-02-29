import pandas as pd
import os 
import cv2
import json
class_list = ['cast', 'wrought']
categories = []
for cls_name in class_list:
    categories.append({
        "id": class_list.index(cls_name),
        "name": cls_name,
        "supercategory": ""
        })
img_df = pd.DataFrame(columns=['id', 'width', 'height', 'file_name', 'license', 'flickr_url', 'coco_url', 'date_captured'])
obj_df = pd.DataFrame(columns=['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd', 'attributes'])

# yolo_pred_path = "{}/{}/gt_labels".format(data_fol, set_name)
# image_fol = "{}/{}/grayscale_images".format(data_fol, set_name)
yolo_pred_path = "Dataset4/all_dataset4/gt_labels"
image_fol = "Dataset4/all_dataset4/images"

img_id = 0
obj_id = 0
for img_txt in os.listdir(yolo_pred_path):
    img_h, img_w, _ = cv2.imread(os.path.join(image_fol, img_txt.replace('.txt', '.jpg'))).shape
    img_df.loc[len(img_df.index)] = [img_id, img_w, img_h, img_txt.replace('.txt', '.jpg'), 0, "", "", 0]
    with open(os.path.join(yolo_pred_path, img_txt), 'r') as f:
        obj_dt = f.readlines()

    print(img_txt)
    for obj in obj_dt:
        obj = obj.replace('\n', '')
        obj = obj.split(' ')
        cat_id = class_list.index(obj[0])

        height = int(float(obj[4])) - int(float(obj[2]))
        width = int(float(obj[3])) - int(float(obj[1]))
        area = height*width
        obj_df.loc[len(obj_df.index)] = [obj_id, img_id, cat_id, [], area, [int(float(obj[1])), int(float(obj[2])), width, height], 0, {"occluded": False, "rotation": 0.0}]
        obj_id += 1
    img_id += 1

ann_img = img_df.to_dict('records')
ann_obj = obj_df.to_dict('records')

coco_ann = {
    "licenses": [
        {
            "name": "",
            "id": 0,
            "url": ""
            }
        ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
        },
    "categories": categories,
    "images": ann_img,
    "annotations": ann_obj}

coco_annot_path = "Dataset4/all_dataset4/annotations"
if not os.path.exists(coco_annot_path):
    os.mkdir(coco_annot_path)

coco_json_path = "Dataset4/all_dataset4/annotations/label.json"
with open(coco_json_path, 'w') as f:
    f.write(json.dumps(coco_ann))