import os
import cv2
import numpy as np
import argparse

import util.utils as utils

def main(args):
    utils.set_env(args)
    dataset_root = './datasets/SOC/'

    soc_pred_dir = args.prediction_dir
    pred_file_names = sorted(os.listdir(soc_pred_dir))

    binarized_output_dir = soc_pred_dir + '_binarized'
    if not os.path.exists(binarized_output_dir):
        os.makedirs(binarized_output_dir)

    utils.binarized(soc_pred_dir, pred_file_names, binarized_output_dir)
    soc_pred_dir = binarized_output_dir

    soc_gt_dir = os.path.join(dataset_root, 'ValSet/Instance gt/gt/')
    soc_txt_dir = os.path.join(dataset_root, 'ValSet/Instance gt/Instance name/')
    file_names = sorted(os.listdir(soc_gt_dir))

    print('=== Extracting Backtracked Boxes from GT ===')
    gt_box_dict = utils.get_gt_dict(soc_gt_dir, file_names)

    print('=== Mapping Prediction''s Backtracked Boxes to GT Region ===')
    PTP, PFP = utils.pred2gt(file_names, soc_gt_dir, soc_pred_dir, soc_txt_dir)
    Precision = PTP / (PTP + PFP)
    print('Precision : ', Precision)
    
    print('=== Mapping GT''s Backtracked Boxes to Prediction Region ===')
    GTP, GFN = utils.gt2pred(file_names, soc_gt_dir, soc_pred_dir, gt_box_dict)
    Recall = GTP / (GTP + GFN)
    print('Recall : ', Recall)
    Sal = 2 * (Precision * Recall) / (Precision + Recall)
    print('Sal Measure performance : ', Sal)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", default='output')
    parser.add_argument("--MIN_BOX_AREA", default=3072, help='0.01 ratio of SOC Image Resolution (480 x 640 x 0.01)')

    args = parser.parse_args()
    main(args)