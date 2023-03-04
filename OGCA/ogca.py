import torch
from torch import nn

from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList

from OGCA.ROIHeads import build_saliency_roi_heads
from OGCA.cvt import build_backbone, build_multi_scale_aggregator

__all__ = ["OGCA"]

@META_ARCH_REGISTRY.register()
class OGCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        assert self.num_classes is not None

        self.backbone = build_backbone(self.num_classes)
        self.backbone.size_divisibility = 32

        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.context_generator = build_multi_scale_aggregator(cfg)
        self.roi_heads = build_saliency_roi_heads(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
    
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        context_features = self.context_generator(features)
        instance_results, detector_losses = self.roi_heads(images, features, proposals, gt_instances, context_features)

        if self.training:
            losses = {}
            losses.update({k: v for k, v in detector_losses.items()})
            losses.update(proposal_losses)

            return losses

        else:
            processed_results = generate_bbox(instance_results, batched_inputs)
            return processed_results

def generate_bbox(instance_results, batched_inputs, threshold=0.5):
    results = []
    for instance_result, input_per_image in zip(instance_results,
                                          batched_inputs):
        height = input_per_image.get("height")
        width = input_per_image.get("width")

        detector_r = detector_postprocess(instance_result, height, width)
        detector_r = {'instances': detector_r}

        results.append(detector_r)
        return results

def build_model(cfg):
    return OGCA(cfg)