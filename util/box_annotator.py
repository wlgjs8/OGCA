import numpy as np
from PIL import Image
from imantics import Mask
import os
import json

COLORS = [
    '(255, 0, 0)',
    '(0, 255, 0)',
    '(0, 0, 255)',
    '(255, 255, 0)',
    '(255, 0, 255)',
    ]

image_root = 'datasets/SOC/TrainSet/Instance gt/Instance name'
file_names = os.listdir(image_root)

for file_name in file_names:
    file_name = '/' + file_name[:-4]
    
    img_path = './datasets/SOC/TrainSet/Imgs' + file_name + '.jpg'
    gt_path = './datasets/SOC/TrainSet/Instance gt/gt' + file_name + '.png'
    gt_class_name = './datasets/SOC/TrainSet/Instance gt/Instance name' + file_name + '.txt'
    img = Image.open(img_path)
    try:
        gt = Image.open(gt_path).convert('RGB')
    except Exception as e:
        gt = Image.open(gt_path[:-3] + 'PNG').convert('RGB')
    width, height = img.size

    anno = {}
    anno['file_name'] = file_name[1:]
    anno['height'] = height
    anno['width'] = width
    anno['objects'] = []

    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = gt.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                sub_masks[pixel_str] = Image.new("1", (width, height))

            sub_masks[pixel_str].putpixel((x, y), 1)

    classes = []
    try:
        with open(gt_class_name, 'r') as f:
            for line in f:
                line = line.strip().split(':')[1]
                classes.append(line)

        for idx, _id in enumerate(COLORS):
            if _id not in sub_masks.keys():
                continue

            mask = Mask(sub_masks[_id])
            area = mask.area()
            area = int(area)
            polygons = mask.polygons()
            bbox = polygons.bbox()

            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            bbox = bbox.tolist()

            segm = [poly for poly in polygons.segmentation if len(poly) % 2 == 0 and len(poly) >= 6]

            obj = {}
            obj['class_name'] = classes[idx]
            obj['area'] = area
            obj['bbox'] = bbox
            obj['polygons'] = segm
            anno['objects'].append(obj)

    except Exception as e:
        print(e)
        continue

    with open('box_annotations/{}.json'.format(file_name[1:]), 'w') as f:
        json.dump(anno, f)