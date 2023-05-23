import os
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from OGCA.ogca import build_model

def inference(file_name, option=None):
    # Inference with model
    cfg = get_cfg()
    cfg.merge_from_file('config/OGCA.yaml')
    weight_file_name = 'model_final.pth'
    cfg.MODEL.WEIGHTS = './saved_models/' + weight_file_name

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 86
    
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 48

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64], [128], [256], [512]]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6']
    cfg.MODEL.RPN.IN_FEATURES = ['p3', 'p4', 'p5', 'p6']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    image_size = 384
    cfg.INPUT.MIN_SIZE_TRAIN = image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size

    predictor = DefaultPredictor(cfg)

    im_path = file_name
    im = cv2.imread(im_path)
    height, width, _ = im.shape
    im = cv2.resize(im, (image_size, image_size))

    predictions = predictor(im)
    predictions = predictions['instances']
    pred_boxes = predictions.pred_boxes.tensor
    pred_boxes = pred_boxes.cpu().data.numpy()

    pred_masks = predictions.pred_masks
    pred_masks = pred_masks.cpu().data.numpy()

    sal_mask = np.zeros(shape=(image_size, image_size), dtype=bool)

    for pred_mask in pred_masks:
        sal_mask = sal_mask | pred_mask

    sal_mask = np.where(sal_mask == True, 255.0, 0.0)
    sal_mask = np.expand_dims(sal_mask, axis = 2)

    sal_mask = np.repeat(sal_mask, 3, axis = 2)
    original_sal_mask = cv2.resize(sal_mask, dsize=(width, height))

    single_channel_mask = original_sal_mask[:,:,0].copy()

    if option == 'save':
        out_path = './result.png'
        cv2.imwrite(out_path, original_sal_mask)

    return single_channel_mask