import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from OGCA.SaliencyModule import build_saliency_module

def build_saliency_box_head(cfg):
    return SaliencyFastRCNNConvFCHead(cfg)

class SaliencyFastRCNNConvFCHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_fc = 3
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self._output_size = (256, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION)

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        self.salient_attention = build_saliency_module(embed_dim=1024, num_heads=8)
        self.out_size = fc_dim

    def forward(self, x, context_features=None):
        
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)

            for l_idx, layer in enumerate(self.fcs):
                if l_idx == 1:
                    x = self.salient_attention(x, context_features)
                    
                x = F.relu(layer(x))

        return x

    @property
    def output_size(self):
        return self.out_size
