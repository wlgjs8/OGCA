import numpy as np
from PIL import Image
from imantics import Mask
import os
import json
import cv2

# COLORS = [
#     '(255, 0, 0)',
#     '(0, 255, 0)',
#     '(0, 0, 255)',
#     '(255, 255, 0)',
#     '(255, 0, 255)',
#     ]

COLORS = [
    '255',
    ]

def rebuild_gt(json_file_name):
    with open('./box_annotations/{}.json'.format(json_file_name), 'r') as f:
        record = json.load(f)

    height = record["height"]
    width = record["width"]

    file_name = record["file_name"]
    objs = record["objects"]

    print('=== rebuild_gt ===')
    area = objs[0]['area']
    bbox = objs[0]['bbox']
    bbox = np.array(bbox, dtype=np.uint16)
    segm = objs[0]['polygons']

    from PIL import Image, ImageDraw

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(segm[0], outline=1, fill=1)
    mask = np.array(img)
    # mask = polygon.mask()

    # print(polygon.ravel().tolist())
    # print(mask)

    img = cv2.imread('./datasets/DUTS/DUTS-TR/DUTS-TR-Image/' + file_name + '.jpg')
    gt_img = cv2.imread('./datasets/DUTS/DUTS-TR/DUTS-TR-Mask/' + file_name + '.png')

    # img = cv2.resize(img, (384, 384))
    # gt_img = cv2.resize(gt_img, (384, 384))

    cv2.imwrite('check_gt2/img.jpg', img)
    cv2.imwrite('check_gt2/gt_img.png', gt_img)

    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 255), thickness=1)
    cv2.imwrite('check_gt2/bbox.jpg', img)
    cv2.imwrite('check_gt2/mask.jpg', mask)


image_root = 'datasets/DUTS/DUTS-TE/DUTS-TE-Image'
file_names = os.listdir(image_root)
file_names = sorted(file_names)

for idx, file_name in enumerate(file_names):
    # if idx > 0:
    #     break

    print('{} FILENAME => '.format(idx), file_name)
    file_name = '/' + file_name[:-4]
    
    # img_path = './datasets/DUTS/DUTS-TR/DUTS-TR-Image' + file_name + '.jpg'
    gt_path = './datasets/DUTS/DUTS-TE/DUTS-TE-Mask' + file_name + '.png'
    # gt_class_name = './datasets/SOC/TrainSet/Instance gt/Instance name' + file_name + '.txt'
    # img = Image.open(img_path)
    try:
        gt = Image.open(gt_path).convert('RGB')
    except Exception as e:
        gt = Image.open(gt_path[:-3] + 'PNG').convert('RGB')
    width, height = gt.size

    gt = gt.convert('L')
    anno = {}
    anno['file_name'] = file_name[1:]
    anno['height'] = height
    anno['width'] = width
    anno['objects'] = []

    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = gt.getpixel((x,y))
            # pixel = gt.getpixel((x,y))
            # print(pixel)

            if pixel > 127:
                pixel = 255
            else:
                pixel = 0

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                sub_masks[pixel_str] = Image.new("1", (width, height))

            sub_masks[pixel_str].putpixel((x, y), 1)

    classes = []
    try:
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
            # obj['class_name'] = classes[idx]
            obj['class_name'] = 1
            obj['area'] = area
            obj['bbox'] = bbox
            obj['polygons'] = segm
            anno['objects'].append(obj)
        print('Safely End!')

    except Exception as e:
        print('Error => ', e)
        continue

    print('file_name : ', file_name)
    with open('box_annotations/{}.json'.format(file_name[1:]), 'w') as f:
        json.dump(anno, f)

    # rebuild_gt('ILSVRC2012_test_00000004', )