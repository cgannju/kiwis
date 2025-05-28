import argparse
import os
import os.path as osp
import sys

cuda_idx = sys.argv[-1]
if cuda_idx != str(-1):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
sys.path.extend(['.', osp.join('.', '..'), osp.join('.', '..', '..')])

import random
import time
import numpy as np
from datetime import datetime
import nibabel as nib
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from seg.models import sam_seg_model_registry, sam_feat_seg_model_registry

from pprint import pprint
from tqdm import tqdm

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description='segmentation')

parser.add_argument('--model_type', type=str, default="vit_b", help='path to splits file')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--save_dir', type=str, default=r"/xxx/KIWIS/seg/inference_outs")
parser.add_argument('--saved_model_path', type=str, default=r"/xxx/KIWIS/seg/checkpoints/checkpoint.pth.tar")
parser.add_argument('--instance_norm', type=bool, default=False, help='whether apply z-score norm during preprocess')
parser.add_argument('--scale', type=float, default=0.25)
parser.add_argument('--domain', type=str, default='test')
parser.add_argument('--strides', type=int, default=256)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=6)
parser.add_argument('--test_data_dir_path', type=str, default=r"/xxx/KIWIS/data")

def main():
    args = parser.parse_args()
    args.distributed = False

    if args.model_type == 'vit_b':
        model_checkpoint = r'/xxx/KIWIS/seg/checkpoints/sam_vit_b_01ec64.pth'
    else:
        raise NotImplementedError

    model = sam_feat_seg_model_registry[args.model_type](num_classes=args.num_classes, checkpoint=model_checkpoint, decoder_checkpoint=args.saved_model_path)

    if args.cuda != -1:
        model.image_encoder = model.image_encoder.cuda() 
        model.mask_decoder = model.mask_decoder.cuda()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    fold = None
    split_name = None
    imgs = None
    gts = None
    strides = [args.strides, args.strides]
    
    if args.domain == 'test':
        split_name = 'test'
        fold = -1
        imgs = glob(osp.join(args.test_data_dir_path, "**", "*.png"), recursive=True)

    crop_size = [1024, 1024]
    scale = args.scale
    instance_norm = args.instance_norm
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    test_on_whole_images(model, args, imgs, gts, 
                         scale=args.scale, 
                         crop_size=tuple(crop_size), 
                         strides=tuple(strides), 
                         instance_norm=args.instance_norm, 
                         save_dir=save_dir)


def norm_img(img_tensor):
    mask = img_tensor.ne(0.0)
    desired = img_tensor[mask]
    mean_val, std_val = desired.mean(), desired.std()
    img_tensor = (img_tensor - mean_val) / std_val
    return img_tensor


def test_on_whole_images(model, args, imgs, gts, scale=1.0, crop_size=(1024, 1024), strides=(256, 256), instance_norm=True, save_dir=None):
    print("===================================INFERENCE ON IMAGES!=======================================")
    print(f"[INFO]: images num = {len(imgs)}")

    model.eval()
    
    with torch.no_grad():
        for img_path in tqdm(imgs, total=len(imgs)):
            print(img_path)
            save_path = osp.join(save_dir, osp.basename(img_path).replace('.png', '_predmask.png'))
            if os.path.exists(save_path):
                continue
            
            img_obj = Image.open(img_path)

            width, height = img_obj.size

            new_width = round(width * scale)
            new_height = round(height * scale)
            
            resz_img_obj = img_obj.resize((new_width, new_height), Image.ANTIALIAS)
            img_ary = np.array(resz_img_obj)

            new_width = round(width * scale)
            new_height = round(height * scale)
            

            img_tsr = torch.FloatTensor(img_ary).unsqueeze(0).permute(0, 3, 1, 2)
            
            if args.cuda != -1:
                img_tsr = img_tsr.cuda()

            h_crop, w_crop = crop_size
            h_stride, w_stride = strides

            bs, h_img, w_img = None, None, None
            pad_h, pad_w = None, None
            pad_flag = False
            batch_crop = []
            batch_box_id = []
            batchsize = args.batchsize

            bs, _, h_img, w_img = img_tsr.size()

            if h_img < h_crop or w_img < w_crop:
                pad_h = max(0, h_crop - h_img)
                pad_w = max(0, w_crop - w_img)

                img_tsr = F.pad(img_tsr, (0, pad_w, 0, pad_h), mode='constant', value=0)

                pad_flag = True

            bs, _, new_h_img, new_w_img = img_tsr.size()

            h_grids = max(new_h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(new_w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            print(f'{h_grids=}, {w_grids=}')

            crop_boxes = []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, new_h_img)
                    x2 = min(x1 + w_crop, new_w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_boxes.append([y1, y2, x1, x2])
            
            
            pred_logits = torch.zeros(1, args.num_classes, new_h_img, new_w_img)
            cnt_logits = torch.zeros(1, new_h_img, new_w_img)


            for box_id, crop_box in enumerate(crop_boxes, 0):
                y1, y2, x1, x2 = crop_box
                crop_img = img_tsr[:, :, y1:y2, x1:x2].squeeze(0)
                

                if instance_norm:
                    crop_img = norm_img(crop_img)
                
                crop_img = crop_img.unsqueeze(0)

                batch_crop.append(crop_img)
                batch_box_id.append(box_id)

                if len(batch_crop) < batchsize: 
                    continue

                else:
                    crops_tensor = torch.cat(batch_crop)
                    
                    masks = model(crops_tensor)
                    masks = F.softmax(masks, dim=1)

                    for mask_id, mask in enumerate(masks):
                        y1, y2, x1, x2 = crop_boxes[batch_box_id[mask_id]]
                        pred_logits[:, :, y1:y2, x1:x2] += mask.detach().cpu()
                        del mask
                        cnt_logits[:, y1:y2, x1:x2] += 1

                    batch_crop = []
                    batch_box_id = []

            if len(batch_crop) != 0:
                crops_tensor = torch.cat(batch_crop)
                masks = model(crops_tensor)
                masks = F.softmax(masks, dim=1)

                for mask_id, mask in enumerate(masks):
                    y1, y2, x1, x2 = crop_boxes[batch_box_id[mask_id]]
                    pred_logits[:, :, y1:y2, x1:x2] += mask.detach().cpu()
                    del mask
                    cnt_logits[:, y1:y2, x1:x2] += 1

                batch_crop = []
                batch_box_id = []
            

            if pad_flag:
                if pad_h > 0:
                    pred_logits = pred_logits[:, :, :-pad_h, :]
                    cnt_logits = cnt_logits[:, :-pad_h, :]
                if pad_w > 0:
                    pred_logits = pred_logits[:, :, :, :-pad_w]
                    cnt_logits = cnt_logits[:, :, :-pad_w]

            masks = pred_logits / cnt_logits
            binary_masks = torch.argmax(F.softmax(masks, dim=1), dim=1, keepdim=True)


            mask = binary_masks.long()
            mask[mask > 0] = 255
            mask = F.interpolate(mask.detach().cpu().float(), (height, width), mode='nearest').long()

            mask_array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
            mask_obj = Image.fromarray(mask_array)
            mask_obj.save(save_path)

if __name__ == '__main__':
    main()