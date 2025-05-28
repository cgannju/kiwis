import sys
import os
import os.path as osp
sys.path.extend(['.', osp.join('.', '..'), osp.join('.', '..', '..')])
cuda_idx = sys.argv[-1]
if cuda_idx != str(-1):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
sys.path.extend(['.', osp.join('.', '..'), osp.join('.', '..', '..')])
from glob import glob
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse
import cv2
import numpy as np
from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchshow as ts
from torchinfo import summary

from pprint import pprint
import json

from albumentations import Normalize, Resize, Compose
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor 
import pandas as pd

from cls.model.vit.sam import ViTRes
from cls.configs.type_model_config import config as type_model_config
from cls.configs.appearance_model_config import config as appearance_model_config
from cls.configs.distribution_model_config import config as distribution_model_config
from cls.configs.fluorescence_model_config import config as fluorescence_model_config
from cls.configs.location_model_config import config as location_model_config
from cls.configs.posneg_model_config import config as posneg_model_config
from cls.configs.t_model_config import config as t_model_config

parser = argparse.ArgumentParser(description='classification')
parser.add_argument('--type_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/type.ckpt", help='model checkpoint')
parser.add_argument('--appearance_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/appearance.ckpt", help='model checkpoint')
parser.add_argument('--distribution_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/distribution.ckpt", help='model checkpoint')
parser.add_argument('--fluorescence_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/fluorescence.ckpt", help='model checkpoint')
parser.add_argument('--location_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/location.ckpt", help='model checkpoint')
parser.add_argument('--posneg_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/posneg.ckpt", help='model checkpoint')
parser.add_argument('--t_ckpt_path', type=str, default=r"/xxx/KIWIS/cls/cls_checkpoints/t.ckpt", help='model checkpoint')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--image_dir_path', type=str, default=r"/xxx/KIWIS/data", help='image path')
parser.add_argument('--mask_dir_path', type=str, default=r"/xxx/KIWIS/seg/inference_outs", help='mask path')
parser.add_argument('--save_dir_path', type=str, default=r"/xxx/KIWIS/cls/results", help='mask path')

args = parser.parse_args()

def load_checkpoint(model, model_ckpt_path):
    model_dict_tmp = torch.load(model_ckpt_path, map_location='cpu')
    model_dict = {}

    for param_name, param_value in model_dict_tmp.items():
        if "image_encoder" in param_name:
            model_dict[param_name.replace("image_encoder.", "")] = param_value
        else:
            continue

    msg = model.backbone.load_state_dict(model_dict, strict=False)
    pprint(msg)
    
    for param_name, param_value in model.backbone.named_parameters():
        param_value.requires_grad = False
    
    return model    

def extract_foreground_components(img_path: str, msk_path: str, padding: int = 50, min_pixels: int = 150) -> List[np.ndarray]:
    try:
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError("unable to load")
    except Exception as e:
        print(f"load error: {str(e)}")
        return []

    if img.shape[:2] != msk.shape:
        print("size mismatch")
        return []

    _, binary_mask = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    result = []
    boxes = []
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < min_pixels:
            continue

        img_h, img_w = img.shape[:2]
        
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(img_w, x + w + padding)
        y_max = min(img_h, y + h + padding)

        cropped = img[y_min:y_max, x_min:x_max]
        box = [x_min, y_min, x_max, y_max]

        result.append(cropped)
        boxes.append(box)

    return result, boxes

if __name__ == '__main__':
    
    type_model = ViTRes(type_model_config.model_config)
    appearance_model = ViTRes(appearance_model_config.model_config)
    distribution_model = ViTRes(distribution_model_config.model_config)
    fluorescence_model = ViTRes(fluorescence_model_config.model_config)
    location_model = ViTRes(location_model_config.model_config)
    posneg_model = ViTRes(posneg_model_config.model_config)
    t_model = ViTRes(t_model_config.model_config)

    summary(type_model, input_size=(1, 3, 1024, 1024))
    summary(appearance_model, input_size=(1, 3, 1024, 1024))
    summary(distribution_model, input_size=(1, 3, 1024, 1024))
    summary(fluorescence_model, input_size=(1, 3, 1024, 1024))
    summary(location_model, input_size=(1, 3, 1024, 1024))
    summary(posneg_model, input_size=(1, 3, 1024, 1024))
    summary(t_model, input_size=(1, 3, 1024, 1024))
    
    type_model = load_checkpoint(type_model, args.type_ckpt_path)
    appearance_model = load_checkpoint(appearance_model, args.appearance_ckpt_path)
    distribution_model = load_checkpoint(distribution_model, args.distribution_ckpt_path)
    fluorescence_model = load_checkpoint(fluorescence_model, args.fluorescence_ckpt_path)
    location_model = load_checkpoint(location_model, args.location_ckpt_path)
    posneg_model = load_checkpoint(posneg_model, args.posneg_ckpt_path)
    t_model = load_checkpoint(t_model, args.t_ckpt_path)
    
    size = [1024, 1024]
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]    
    
    list_transforms = []
    list_transforms.extend(
    [
        Resize(height=size[0], width=size[1]),
        Normalize(mean=mean, std=std, p=1),
        ToTensor()
    ]
    )
    transform_func = Compose(list_transforms)
    
    img_filepaths = glob(osp.join(args.image_dir_path, '**', '*.png'), recursive=True)
    msk_filepaths = [osp.join(args.mask_dir_path, osp.basename(item).replace('.png', '_predmask.png')) for item in img_filepaths]
    
    results = []
    
    for img_ix, (image_path, mask_path) in tqdm(enumerate(zip(img_filepaths, msk_filepaths)), total=len(img_filepaths)):
        template = {}
        template['image_path'] = osp.basename(image_path)

        image_obj = Image.open(image_path)
        mask_obj = Image.open(mask_path)
    
        margin = 50
        min_area_pixels = 150
    
        components, boxes = extract_foreground_components(image_path, mask_path, padding=margin, min_pixels=min_area_pixels)

        for comp_ix, (comp, box) in enumerate(zip(components, boxes), len(boxes)):
            template['bbox'] = '_'.join([str(item) for item in box])

            img_ary = comp
            msk_ary = np.zeros_like(img_ary)[:, :, 0]
            
            transform_img_tensor = transform_func(image=img_ary, mask=msk_ary)
        
            img_tsr = transform_img_tensor['image'].float().cuda().unsqueeze(0)
            
            type_model = type_model.cuda()
            type_logits = type_model(img_tsr)
            type_model = type_model.cpu()
            type_pred = torch.argmax(type_logits, dim=1).item()
            type_attr = {0: 'Glomeruli', 1: 'Tubule', 2: 'Vessel'}[type_pred]
            template['RenalStructure'] = type_attr
            
            if type_attr == 'Glomeruli':
            
                appearance_model = appearance_model.cuda()
                appearance_logits = appearance_model(img_tsr)
                appearance_model = appearance_model.cpu()
                appearance_pred = torch.argmax(appearance_logits, dim=1).item()
                appearance_attr = {0: "Granular", 1: "Clumpy"}[appearance_pred]
                template['AppearanceOfGlomeruli'] = appearance_attr
                
                distribution_model = distribution_model.cuda()
                distribution_logits = distribution_model(img_tsr)
                distribution_model = distribution_model.cpu()
                distribution_pred = torch.argmax(distribution_logits, dim=1).item()
                distribution_attr = {0: "Global", 1: "Segmental"}[distribution_pred]
                template['DistributionOfGlomeruli'] = distribution_attr

                fluorescence_model = fluorescence_model.cuda()
                fluorescence_logits = fluorescence_model(img_tsr)
                fluorescence_model = fluorescence_model.cpu()
                fluorescence_pred = torch.argmax(fluorescence_logits, dim=1).item()
                fluorescence_attr = {0: "1+", 1: "2+", 2: "3+", 3: "Trace"}[fluorescence_pred]
                template['FluorescenceOfGlomeruli'] = fluorescence_attr

                location_model = location_model.cuda()
                location_logits = location_model(img_tsr)
                location_model = location_model.cpu()
                location_pred = torch.argmax(location_logits, dim=1).item()
                location_attr = {0: "Capillary loop", 1: "Mesangial area"}[location_pred]
                template['LocationOfGlomeruli'] = location_attr
            
                posneg_model = posneg_model.cuda()
                posneg_logits = posneg_model(img_tsr)
                posneg_model = posneg_model.cpu()
                posneg_pred = torch.argmax(posneg_logits, dim=1).item()
                posneg_attr = {0: 'Negativity', 1: 'Positivity'}[posneg_pred]
                template['PositiveOrNegative'] = posneg_attr
            
            elif type_attr == 'Tubule':
            
                t_model = t_model.cuda()
                t_logits = t_model(img_tsr)
                t_model = t_model.cpu()
                t_pred = torch.argmax(t_logits, dim=1).item()
                t_attr = {0: 'Epithelial cell cytoplasm', 1: 'Basal membrane', 2: 'Protein cast'}[t_pred]
                template['SubtypeOfTubule'] = t_attr
                
            elif type_attr == 'Vessel': 
            
                posneg_model = posneg_model.cuda()
                posneg_logits = posneg_model(img_tsr)
                posneg_model = posneg_model.cpu()
                posneg_pred = torch.argmax(posneg_logits, dim=1).item()
                posneg_attr = {0: 'Negativity', 1: 'Positivity'}[posneg_pred]
                template['PositiveOrNegative'] = posneg_attr
                
            results.append(deepcopy(template))
    df = pd.DataFrame(results)
    os.makedirs(args.save_dir_path, exist_ok=True)
    df.to_excel(osp.join(args.save_dir_path, 'attributes.xlsx'), index=False)
    
            
        