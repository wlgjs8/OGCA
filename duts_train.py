import os
import argparse
import torch
from torch import autograd

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from OGCA.ogca import OGCA
from OGCA.Trainer import SaliencyTrainer
from datasets.dataset import init_dataset

setup_logger()

def main(args):
    torch.cuda.empty_cache()

    image_size = args.image_size
    epoch = args.epoch
    iteration = args.iteration

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='polygon'
    cfg.merge_from_file('config/OGCA.yaml')
    # We only support for batch size 1
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 48

    cfg.OUTPUT_DIR = './saved_models/'
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64], [128], [256], [512]]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6']
    cfg.MODEL.RPN.IN_FEATURES = ['p3', 'p4', 'p5', 'p6']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.INPUT.MIN_SIZE_TRAIN = image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    START_ITER = 1000000
    cfg.SOLVER.MAX_ITER = START_ITER + epoch * iteration

    weight_file_name = 'model_init.pth'
    cfg.MODEL.WEIGHTS = './pretrained/' + weight_file_name

    cfg.SOLVER.CHECKPOINT_PERIOD = iteration
    cfg.TEST.EVAL_PERIOD = iteration

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    init_dataset(args)

    cfg.DATASETS.TRAIN = ('coco_train_DUTS',)
    cfg.DATASETS.TEST = ("coco_val_DUTS", )
    cfg.DATASETS.VAL = ("coco_val_DUTS", )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    # print(cfg)
    
    trainer = SaliencyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # print(trainer.model)

    # for name, param in trainer.model.named_parameters():
    #     print('>> : ', name)
    #     print(param)

    # for name, layer in trainer.model.named_modules():
    # freeze_lists = [
    #     'roi_heads.box_head.fc',
    #     'roi_heads.box_predictor'
    # ]
    # for name, param in trainer.model.named_parameters():
    #     if name in freeze_lists:
    #         print('>> : ', name)
    #         param.requires_grad = False

    # trainer.model.roi_heads.box_head.fc1.weight.requires_grad = False
    # trainer.model.roi_heads.box_head.fc1.bias.requires_grad = False
    # trainer.model.roi_heads.box_head.fc2.weight.requires_grad = False
    # trainer.model.roi_heads.box_head.fc2.bias.requires_grad = False
    # trainer.model.roi_heads.box_head.fc3.weight.requires_grad = False
    # trainer.model.roi_heads.box_head.fc3.bias.requires_grad = False
    # trainer.model.roi_heads.box_predictor.cls_score.weight.requires_grad = False
    # trainer.model.roi_heads.box_predictor.cls_score.bias.requires_grad = False
    # trainer.model.roi_heads.box_predictor.bbox_pred.weight.requires_grad = False
    # trainer.model.roi_heads.box_predictor.bbox_pred.bias.requires_grad = False

    for name, param in trainer.model.named_parameters():
        print('name : ', name, ' => ', param.requires_grad)
        # print(param.requires_grad)
        # print(name.param)
        # print()
        # print(name, param.requires_grad)

    with autograd.detect_anomaly():
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='duts', help='training dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--image_size', type=int, default=384, help='input size')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/imagenet/cvt/cvt-13-384x384.yaml',
                        type=str)

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)

    parser.add_argument('--num_classes', type=float, default=1, help='DUTS have single classes')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--iteration', type=int, default=5000, help='number of salient images in SOC')
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")

    args = parser.parse_args()
    main(args)