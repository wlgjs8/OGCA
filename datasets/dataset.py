import util.utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog

class_names = [
    'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'camera', 'car', 'carrot', 
    'cat', 'cell phone', 'chair', 'clock', 'computer monitor', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'food', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'hat', 'horse',
    'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pillow', 'pizza', 'potted plant', 'quilt', 'refrigerator', 'remote', 'sandwich', 'scissors',
    'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 
    'tv', 'umbrella', 'vase', 'wine glass', 'zebra'
]

def register_dataset(name, metadata, image_root, gt_image_root, instance_gt_image_root, instance_gt_class_root):
    DatasetCatalog.register(
        name,
        # lambda: utils.get_soc_instances_dicts(image_root, gt_image_root, instance_gt_image_root, instance_gt_class_root)
        lambda: utils.get_sod_dicts(image_root, gt_image_root)
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        gt_image_root=gt_image_root,
        instance_gt_image_root=instance_gt_image_root,
        instance_gt_class_root=instance_gt_class_root,
        thing_classes=class_names,
        **metadata
    )
    print("Dataset Registered!!!")

def init_dataset(args):
    dataset_root = 'datasets/'

    if args.data == 'soc':
        args.data = 'SOC'
        image_root = dataset_root + args.data + '/TrainSet' + '/Imgs/'
        gt_image_root = dataset_root + args.data + '/TrainSet' + '/gt/'
        instance_gt_image_root = dataset_root + args.data + '/TrainSet' + '/Instance gt' + '/gt/'
        instance_gt_class_root = dataset_root + args.data + '/TrainSet' + '/Instance gt' + '/Instance name/'

        val_image_root = dataset_root + args.data + '/ValSet' + '/Imgs/'
        val_gt_image_root = dataset_root + args.data + '/ValSet' + '/gt/'
        val_instance_gt_image_root = dataset_root + args.data + '/ValSet' + '/Instance gt' + '/gt/'
        val_instance_gt_class_root = dataset_root + args.data + '/ValSet' + '/Instance gt' + '/Instance name/'

        train_name = 'coco_train_SOC'
        val_name = 'coco_val_SOC'
        metadata = {}

        register_dataset(train_name, metadata, image_root, gt_image_root, instance_gt_image_root, instance_gt_class_root)
        register_dataset(val_name, metadata, val_image_root, val_gt_image_root, val_instance_gt_image_root, val_instance_gt_class_root)

    elif args.data == 'duts':
        args.data = 'DUTS'
        image_root = dataset_root + args.data + '/DUTS-TR' + '/DUTS-TR-Image/'
        gt_image_root = dataset_root + args.data + '/DUTS-TR' + '/DUTS-TR-Mask/'

        val_image_root = dataset_root + args.data + '/DUTS-TE' + '/DUTS-TE-Image/'
        val_gt_image_root = dataset_root + args.data + '/DUTS-TE' + '/DUTS-TE-Mask/'

        train_name = 'coco_train_DUTS'
        val_name = 'coco_val_DUTS'
        metadata = {}

        register_dataset(train_name, metadata, image_root, gt_image_root, None, None)
        register_dataset(val_name, metadata, val_image_root, val_gt_image_root, None, None)
        