import json
import pandas as pd
from tqdm import tqdm

data_path = "Dataset4/all_dataset4"
coco_annot = "Dataset4/all_dataset4/annotations/label.json"

annot = json.load(open(coco_annot, 'r'))
image_df = pd.DataFrame(annot['images'])
box_df = pd.DataFrame(annot['annotations'])

shapes = []
for index, img in tqdm(image_df.iterrows()):
    img_boxes = box_df.loc[box_df.image_id == img.id]
    label = img_boxes.category_id.map(lambda m: annot["categories"][m]["name"])

    points = img_boxes.bbox.map(lambda b: [b[:2], [b[0]+b[2], b[1]+b[3]]])
    img_boxes['label'] = label
    img_boxes['points'] = points
    img_boxes['group_id'] = None
    img_boxes['description'] = ''
    img_boxes['shape_type'] = 'rectangle'
    img_boxes['flags'] = None
    img_boxes = img_boxes[['label', 'points', 'group_id', 'description', 'shape_type', 'flags']]
    img_boxes = img_boxes.to_dict('records')
    print(img_boxes)

    for i in range(len(img_boxes)):
        img_boxes[i]['flags'] = {}

    template = {
        "version": "5.2.1",
        "flags": {},
        "shapes": img_boxes,
        "imagePath": "../images/{}".format(img.file_name),
        "imageData": None,
        "imageHeight": img.height,
        "imageWidth": img.width
        }

    with open('{}/labelme_annotations/{}.json'.format(data_path, img.file_name.replace('.jpg', '')), 'w') as f:
        f.write(json.dumps(template))