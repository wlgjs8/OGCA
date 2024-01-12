import os
import sys
import cv2
from tqdm import tqdm
import util.metrics as M
import json
import argparse

def evaluation(args):
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    AFM = M.AvgFmeasure()

    method = args.method
    dataset = args.dataset
    attr = args.attr

    gt_root = './datasets/SOC/ValSet/gt/'
    pred_root = args.save + '/'
    gt_name_list = sorted(os.listdir(pred_root))

    em_dict = dict()

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        if '.png' not in gt_name:
            continue

        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)
        AFM.step(pred=pred, gt=gt)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    afm = AFM.get_results()['avg_fm']

    print(
        'Method:', args.method, ',',
        'Dataset:', args.dataset, ',',
        'Attribute:', args.attr, '||',
        'Smeasure:', sm.round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'maxFm:', fm['curve'].max().round(3), '; ',
        'avgFm:', afm['avg_fm'].mean().round(3),
        sep=''
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='OGCA')
    parser.add_argument("--dataset", default='SOC')
    parser.add_argument("--attr", default='SOC-1200')
    main()
