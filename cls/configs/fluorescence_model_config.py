from easydict import EasyDict
from functools import partial
import torch

config = EasyDict({})


config.model_config = EasyDict({})
config.model_config.backbone = EasyDict({})
config.model_config.backbone.img_size = 1024
config.model_config.backbone.patch_size = 16
config.model_config.backbone.in_chans = 3
config.model_config.backbone.embed_dim = 768
config.model_config.backbone.depth = 12
config.model_config.backbone.num_heads = 12
config.model_config.backbone.mlp_ratio = 4.0
config.model_config.backbone.out_chans = 256
config.model_config.backbone.qkv_bias = True
config.model_config.backbone.norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
config.model_config.backbone.act_layer = torch.nn.GELU
config.model_config.backbone.use_abs_pos = False
config.model_config.backbone.use_rel_pos = True
config.model_config.backbone.rel_pos_zero_init = True
config.model_config.backbone.global_attn_indexes = [2, 5, 8, 11]
config.model_config.backbone.window_size = 14
config.model_config.backbone.out_chans = 256

config.model_config.fusion = EasyDict({})
config.model_config.fusion.spatial_dims = 2
config.model_config.fusion.in_channels = 256
config.model_config.fusion.conv_out_channels = 256

config.model_config.head = EasyDict({})
config.model_config.head.in_shape = (1024, 64, 64)
config.model_config.head.channels = (256,)
config.model_config.head.classes = 4
config.model_config.head.strides = (1, 1)
config.model_config.head.dropout = 0.1