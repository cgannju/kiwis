import os
import os.path as osp
import sys
sys.path.extend(['.', osp.join('.', '.'), osp.join('.', '.', '.'), osp.join('.', '.', '.', '.')])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from monai.networks.nets.classifier import Classifier
from monai.networks.blocks import SimpleASPP

from cls.model.segment_anything.modeling.image_encoder import ImageEncoderViT

class ViTRes(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.backbone = ImageEncoderViT(**self.config.backbone)
        self.fusion = SimpleASPP(**self.config.fusion)
        self.head = Classifier(**self.config.head)
        
    def forward(self, x):
        features = self.backbone(x)
        fusion = self.fusion(features)
        logits = self.head(fusion)
        
        return logits
        
                


if __name__=="__main__":
    model = ViTRes(config=None)
    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(out.shape)
    summary(model, input_size=(1, 3, 1024, 1024))
    
    