import torch
import torch.nn as nn

def build_add_features(cfg):
    return AddMaskHead(cfg)

class AddMaskHead(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        self.mask_upsample = nn.Upsample(size=(cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION, cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION), mode='bilinear')

        self.add_model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, features, proposal_boxes, mask_features):
        concat_feats = self.mask_upsample(features[0])
        proposal_boxes = [boxes.tensor * 0.125 for boxes in proposal_boxes]

        batch_crop_features = None
        for batch_idx, batch_proposal_boxes in enumerate(proposal_boxes):
            if batch_proposal_boxes.shape[0] != 0:
                for idx, boxes in enumerate(batch_proposal_boxes):
                    x1, y1, x2, y2 = boxes
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    if x2-x1 < 1:
                        if x2 != self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION:
                            x2 = x2 + 1
                        else:
                            x1 = x1 - 1
                    
                    if y2-y1 < 1:
                        if y2 != self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION:
                            y2 = y2 + 1
                        else:
                            y1 = y1 - 1

                    crop_feature = concat_feats[batch_idx, :, y1:y2, x1:x2]
                    crop_feature = self.mask_upsample(crop_feature.unsqueeze(0))

                    if idx == 0:
                        crop_features = crop_feature
                    else:
                        crop_features = torch.cat([crop_features, crop_feature], dim=0)
            
                if batch_crop_features == None:
                    batch_crop_features = crop_features
                else:
                    batch_crop_features = torch.cat([batch_crop_features, crop_features], dim=0)
        
        if batch_crop_features == None:
            new_mask_features = mask_features
        else:
            if batch_crop_features.shape[0] != 0:
                new_mask_features = torch.cat([mask_features, batch_crop_features], dim=1)
                new_mask_features = self.add_model(new_mask_features)
            else:
                new_mask_features = mask_features
        
        return new_mask_features