import json
import logging

import os
from detectron2.structures import BoxMode

from PIL import Image
import json
import numpy as np
import cv2

from env import env

from imantics import Mask, Polygons, BBox
from shapely.geometry import Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

COLORS = [
    '(255, 0, 0)',
    '(0, 255, 0)',
    '(0, 0, 255)',
    '(255, 255, 0)',
    '(255, 0, 255)',
    ]

class_names = [
    'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'camera', 'car', 'carrot', 
    'cat', 'cell phone', 'chair', 'clock', 'computer monitor', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'food', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'hat', 'horse',
    'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pillow', 'pizza', 'potted plant', 'quilt', 'refrigerator', 'remote', 'sandwich', 'scissors',
    'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 
    'tv', 'umbrella', 'vase', 'wine glass', 'zebra'
]

def get_soc_instances_dicts(image_root, gt_image_root, instance_gt_image_root, instance_gt_class_root):
    print("\n")
    print("Loading Instances Data ...")

    '''
    image_root :  datasets/SOC/TrainSet/Imgs/
    gt_image_root :  datasets/SOC/TrainSet/gt/
    instance_gt_image_root :  datasets/SOC/TrainSet/Instance gt/
    instance_gt_class_root :  datasets/SOC/TrainSet/Instance gt/Instance name/
    '''

    dataset_dicts = []

    file_names = os.listdir(image_root)
    instance_file_names = os.listdir(instance_gt_class_root)

    for file_name in file_names:
        image_name = file_name[:-4]
        txt_file_name = image_name + '.txt'

        im_path = image_root + file_name
        width, height = Image.open(im_path).size

        gt_path = gt_image_root + file_name[:-3] + 'png'
        instance_gt_path = instance_gt_image_root + file_name[:-3] + 'png'
        instance_gt_class_path = instance_gt_class_root + file_name[:-3] +'txt'

        objs = []
        record = {}
        record["file_name"] = im_path
        record["image_id"] = file_name
        record["height"] = height
        record["width"] = width
        
        record["gt_file_name"] = gt_path
        record["instance_gt_file_name"] = instance_gt_path
        record["annotations"] = objs

        if txt_file_name in instance_file_names:
            try:
                with open('util/box_annotations/{}.json'.format(image_name), 'r') as f:
                    annos = json.load(f)
                for anno in annos['objects']:
                    category_id = [i for i in range(len(class_names)) if anno['class_name'] == class_names[i]]

                    obj = {
                        'area' : anno['area'],
                        'bbox' : anno['bbox'],
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'segmentation' : anno['polygons'],
                        "category_id": category_id[0],
                        "iscrowd": 0,
                    }
                    objs.append(obj)
                dataset_dicts.append(record)

            except Exception as e:
                continue
        else:
            dataset_dicts.append(record)
            continue
        
    print("Loading Instances Data Complete!!!")
    return dataset_dicts

def get_sod_dicts(image_root, gt_image_root):
    print("\n")
    print("Loading Instances Data ...")

    dataset_dicts = []

    file_names = os.listdir(image_root)
    # instance_file_names = os.listdir(instance_gt_class_root)

    for file_name in file_names:
        image_name = file_name[:-4]
        txt_file_name = image_name + '.txt'

        im_path = image_root + file_name
        width, height = Image.open(im_path).size

        gt_path = gt_image_root + file_name[:-3] + 'png'

        objs = []
        record = {}
        record["file_name"] = im_path
        record["image_id"] = file_name
        record["height"] = height
        record["width"] = width
        
        record["gt_file_name"] = gt_path
        # record["instance_gt_file_name"] = instance_gt_path
        record["annotations"] = objs

        try:
            with open('box_annotations/{}.json'.format(image_name), 'r') as f:
                annos = json.load(f)
            for anno in annos['objects']:
                obj = {
                    'area' : anno['area'],
                    'bbox' : anno['bbox'],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation' : anno['polygons'],
                    "category_id": 0,
                    "iscrowd": 0,
                }
                objs.append(obj)
            dataset_dicts.append(record)

        except Exception as e:
            continue
        
    print("Loading Instances Data Complete!!!")
    return dataset_dicts

def refine_boxes(boxes):
    delete_box = []
    final_box = []
    for cur_box in boxes:
        for other_box in boxes:
            if cur_box == other_box:
                continue
            else:
                if cur_box[0] > other_box[0]:
                    continue
                if cur_box[2] < other_box[2]:
                    continue
                if cur_box[1] > other_box[1]:
                    continue
                if cur_box[3] < other_box[3]:
                    continue
                delete_box.append(other_box)
    
    for cur_box in boxes:
        if cur_box not in delete_box:
            final_box.append(cur_box)
    return final_box

def max_area_polygon(multi_polygons):
    max_area = 0
    max_idx = 0
    no_obj_flag = True
    
    for m_pts_idx, m_pts in enumerate(multi_polygons):
        try:
            polygon = Polygon(m_pts)
        except Exception as e:
            '''
            len(poly points) < 3:
            '''
            continue

        if max_area < polygon.area:
            no_obj_flag = False
            max_area = polygon.area
            max_idx = m_pts_idx
            
    if no_obj_flag:
        return []
    else:
        return multi_polygons[max_idx]

def pred2gt(pred_file_names, soc_gt_dir, soc_pred_dir, soc_txt_dir, THRESHOLD=0.5):
    nTP, nFP = 0, 0
    pred_box_dict = {}
    for i, pred_file_name in enumerate(pred_file_names):
        pred_image = Image.open(soc_pred_dir + '/' + pred_file_name)
        width, height = pred_image.size

        pred_arr = Mask(np.array(pred_image, dtype=bool))
        pred_poly = pred_arr.polygons()
        multi_pred_poly = pred_poly.points

        bboxes = []

        for single_poly in multi_pred_poly:
            single_poly = Polygons(single_poly)
            bbox = single_poly.bbox()
            if bbox.area() < MIN_BOX_AREA:
                continue
            if bbox.area() > MAX_BOX_AREA:
                continue
            bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            bbox = bbox.tolist()
            bboxes.append(bbox)
        bboxes = refine_boxes(bboxes)
        pred_box_dict[pred_file_name] = bboxes
        
    for i, pred_file_name in enumerate(pred_file_names):
        pred_image = Image.open(soc_pred_dir + '/' + pred_file_name)
        gt_image = Image.open(soc_gt_dir + '/' + pred_file_name)

        try:
            bboxes = pred_box_dict[pred_file_name]
        except KeyError:
            continue

        TP, FP = 0, 0
        for bbox in bboxes:
            pred_crop_image = pred_image.crop(bbox)
            gt_crop_image = gt_image.crop(bbox)

            pred_crop_arr = Mask(np.array(pred_crop_image, dtype=bool))
            pred_crop_poly = pred_crop_arr.polygons()
            multi_pred_poly_points = pred_crop_poly.points
            single_pred_poly = max_area_polygon(multi_pred_poly_points)
            single_pred_poly = Polygon(single_pred_poly)

            gt_crop_arr = np.array(gt_crop_image, dtype=bool)
            if True not in gt_crop_arr :
                FP += 1
                continue
            gt_crop_arr = Mask(gt_crop_arr)

            gt_crop_poly = gt_crop_arr.polygons()
            multi_gt_poly_points = gt_crop_poly.points
            single_gt_poly = max_area_polygon(multi_gt_poly_points)
            single_gt_poly = Polygon(single_gt_poly)

            single_gt_poly = make_valid(single_gt_poly)
            single_pred_poly = make_valid(single_pred_poly)

            intersect = single_gt_poly.intersection(single_pred_poly)
            intersect = intersect.area
            union = single_gt_poly.union(single_pred_poly)
            union = union.area
            iou = intersect / union

            if iou > THRESHOLD:
                TP += 1
            else:
                FP += 1

        txt_file = soc_txt_dir + '/' + pred_file_name[:-3] + 'txt'
        with open(txt_file, 'r') as r:
            line = r.readlines()
            LimitTP = len(line)
        if TP > LimitTP:
            TP = LimitTP

        nTP += TP
        nFP += FP

    Pred_nTP = nTP
    Pred_nFP = nFP

    return Pred_nTP, Pred_nFP


def gt2pred(gt_file_names, soc_gt_dir, soc_pred_dir, gt_box_dict, THRESHOLD=0.5):
    nTP, nFN = 0, 0

    for i, gt_file_name in enumerate(gt_file_names):
        gt_image = Image.open(soc_gt_dir + '/' + gt_file_name)
        pred_image = Image.open(soc_pred_dir + '/' + gt_file_name)

        bboxes = gt_box_dict[gt_file_name]
        TP, FN = 0, 0
        for bbox in bboxes:
            gt_crop_image = gt_image.crop(bbox)
            pred_crop_image = pred_image.crop(bbox)

            gt_crop_arr = Mask(np.array(gt_crop_image, dtype=bool))
            gt_crop_poly = gt_crop_arr.polygons()
            multi_gt_poly_points = gt_crop_poly.points
            single_gt_poly = max_area_polygon(multi_gt_poly_points)
            single_gt_poly = Polygon(single_gt_poly)

            pred_crop_arr = Mask(np.array(pred_crop_image, dtype=bool))
            pred_crop_poly = pred_crop_arr.polygons()
            multi_pred_poly_points = pred_crop_poly.points
            if len(multi_pred_poly_points) == 0:
                FN += 1
                continue
            single_pred_poly = max_area_polygon(multi_pred_poly_points)
            if len(single_pred_poly) == 0:
                FN += 1
                continue

            single_pred_poly = Polygon(single_pred_poly)

            single_gt_poly = make_valid(single_gt_poly)
            single_pred_poly = make_valid(single_pred_poly)

            intersect = single_gt_poly.intersection(single_pred_poly)
            intersect = intersect.area
            union = single_gt_poly.union(single_pred_poly)
            union = union.area
            iou = intersect / union

            if iou > THRESHOLD:
                TP += 1
            else:
                FN += 1

        nTP += TP
        nFN += FN

    return nTP, nFN

def get_gt_dict(soc_gt_dir, gt_file_names):
    gt_box_dict = {}
    for i, gt_file_name in enumerate(gt_file_names):
        gt_image = Image.open(soc_gt_dir + '/' + gt_file_name).convert('RGB')
        width, height = gt_image.size

        sub_masks = {}
        for x in range(width):
            for y in range(height):
                pixel = gt_image.getpixel((x,y))[:3]

                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    sub_masks[pixel_str] = Image.new("1", (width, height))

                sub_masks[pixel_str].putpixel((x, y), 1)

        bboxes = []
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
            bboxes.append(bbox)
        gt_box_dict[gt_file_name] = bboxes
    return gt_box_dict

def set_env(args):
    global MIN_BOX_AREA, MAX_BOX_AREA
    MIN_BOX_AREA = env.get_MIN_BOX_AREA()
    MAX_BOX_AREA = env.get_MAX_BOX_AREA()

def binarized(soc_pred_dir, pred_file_names, binarized_output_dir):
    for pred_file_name in pred_file_names:
        pred_path = os.path.join(soc_pred_dir, pred_file_name)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        pred = _prepare_data(pred)
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        output = pred >= adaptive_threshold
        output = (output * 255).astype(np.uint8)
        cv2.imwrite(binarized_output_dir + '/' + pred_file_name, output)


def _prepare_data(pred: np.ndarray):
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())

    adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
    output = pred >= adaptive_threshold
    output = (output * 255).astype(np.uint8)
    return output

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)