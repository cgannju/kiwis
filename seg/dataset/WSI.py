import pickle
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
from pprint import pprint
import time
import torch
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize, adjust_brightness

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
import torchshow as ts


join = os.path.join

import json
from glob import glob
import os
import os.path as osp

import random

class WSIDataset(Dataset):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.patch_size = (args.img_size, args.img_size)
        self.mode = mode

        split = None
        with open(args.split_file, 'r') as f:
            split = json.load(f)
        self.fold = args.fold
        self.instance_norm = args.instance_norm

        if self.mode == "train" or self.mode == "val":
            self.imgs = [item[0] for item in split[self.mode][str(self.fold)]]
            self.msks = [item[1] for item in split[self.mode][str(self.fold)]]
        elif self.mode == "test":
            self.imgs = [item[0] for item in split[self.mode]]
            self.msks = [item[1] for item in split[self.mode]]
        else:
            raise NotImplementedError("mode is not supported:", self.mode)
        
        print(f'dataset length: {len(self.imgs)}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        
        img_path, msk_path = self.imgs[index], self.msks[index]

        img = Image.open(self.imgs[index])
        label = Image.open(self.msks[index])

        label = np.array(label)
        label[label == 255] = 1
        label[label > 1] = 0

        img = np.array(img).astype(np.float32).transpose([2, 0, 1])
        if self.instance_norm:
            img = (img - img.min()) / (img.max() - img.min())

        if self.mode == 'contrast':
            img1, img2 = self.transform_contrast(img)
            return img1, img2
        else:
            img, label = self.transform(img, label)
            return img, label, img_path, msk_path

    def transform_contrast(self, img):
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None]}
        tr_transforms = [  # CenterCropTransform(crop_size=target_size),
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            ResizeTransform(target_size=self.patch_size, order=1),  # resize
            MirrorTransform(axes=(1,)),
            SpatialTransform(patch_size=self.patch_size, random_crop=False,
                             patch_center_dist_from_border=self.patch_size[0] // 2,
                             do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                             do_rotation=True, p_rot_per_sample=0.5,
                             angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                             scale=(0.5, 1.9), p_scale_per_sample=0.5,
                             border_mode_data="nearest", border_mode_seg="nearest"),
        ]

        train_transform = Compose(tr_transforms)
        data_dict = train_transform(**data_dict)
        img1 = data_dict.get('data')[0]
        data_dict = train_transform(**data_dict)
        img2 = data_dict.get('data')[0]
        return img1, img2

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                 patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label


class WSIDatasetPrefetch(Dataset):
    def __init__(self, args, mode='train', use_aux_data=False, shift=False):
        super().__init__()
        self.args = args
        self.patch_size = (args.img_size, args.img_size)
        self.mode = mode
        self.use_aux_data = use_aux_data
        self.shift = shift
        self.roundbox = args.roundbox
        self.segObj = args.segObj
        self.data = args.data
        print(f'[warning]:{self.patch_size=}, {args.img_size=}, {args.roundbox=}, {args.shift=}, {args.segObj=}')
        
        self.save_dir = osp.join(args.save_dir, "cached_vis", "FromWSIDatasetPrefetch", self.mode)
        os.makedirs(self.save_dir, exist_ok=True)

        sub_grid = None
        if args.roundbox:
            sub_grid = '_2dgrid_' + \
                str(self.patch_size[0]) + \
                'x' + \
                str(self.patch_size[1]) + \
                '_fold_' + \
                str(args.fold) + \
                f'_{args.per_image_sample}_per_sample' + \
                f'_scale_{args.scale}' + \
                f'_instance_norm_{args.instance_norm}' + \
                f'_min_area_{args.min_area}_roundbox_{args.roundbox}'            
        else:
            sub_grid = '_2dgrid_' + \
                str(self.patch_size[0]) + \
                'x' + \
                str(self.patch_size[1]) + \
                '_fold_' + \
                str(args.fold) + \
                f'_{args.per_image_sample}_per_sample' + \
                f'_scale_{args.scale}' + \
                f'_instance_norm_{args.instance_norm}' + \
                f'_min_area_{args.min_area}'

        self.sub_vol_path = osp.join(
            r'/home/gc/workbench/wsiworkbench/MedicalZooPytorch/datasets',
            # 'PrivateWSI_898/generated/' + mode + sub_grid + '/'
            # 'PrivateWSI_898_Omit_Negative/generated/' + mode + sub_grid + '/'
            # 'PrivateWSI_893_AdjustBrightness/generated/' + mode + sub_grid + '/'
            # 'PrivateWSI_893_SplitionBasedOnNaRu/generated/' + mode + sub_grid + '/'
            # 'PrivateWSI_893_SplitionBasedOnNaRu_only_A/generated/' + mode + sub_grid + '/'
            f'{self.data}/generated/' + mode + sub_grid + '/'
            # 'PrivateWSI_893_SplitionBasedOnNaRu_10_percent/generated/' + mode + sub_grid + '/'
        )

        print(f"[warning]:{self.sub_vol_path}...")

        image_paths = glob(osp.join(self.sub_vol_path, "**input.npy"))
        label_paths = glob(osp.join(self.sub_vol_path, "**label.npy"))
        print(f"[info]:{len(image_paths)=}, {len(label_paths)=}")
        
        if self.use_aux_data:
            print(f"{self.mode=}, {len(image_paths)=}")
            aux_image_paths = glob(osp.join(self.sub_vol_path.replace('PrivateWSI_898', 'PrivateWSI_129'), "**input.npy"))
            aux_label_paths = glob(osp.join(self.sub_vol_path.replace('PrivateWSI_898', 'PrivateWSI_129'), "**label.npy"))
            
            image_paths += aux_image_paths
            label_paths += aux_label_paths
            print(f"{self.mode=}, {len(image_paths)=}")
            print("="*40)

        image_paths.sort()
        label_paths.sort()

        self.imgs = image_paths
        self.msks = label_paths
        
        print(f'dataset length: {len(self.imgs)}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        t_d = {}
        
        t0 = time.time()
        img_path, msk_path = self.imgs[index], self.msks[index]

        msk_ary = np.load(msk_path)
        t1 = time.time()
        t_d[str(1)] = t1-t0
        
        
        if self.segObj == 'all':
            msk_ary[msk_ary > 0] = 1
        elif self.segObj == 'G（小球）':
            msk_ary[msk_ary != 255] = 0
            msk_ary[msk_ary == 255] = 1
        elif self.segObj == 'T（小管）':
            msk_ary[msk_ary != 192] = 0
            msk_ary[msk_ary == 192] = 1
        elif self.segObj == 'A（动脉）':
            msk_ary[msk_ary != 128] = 0
            msk_ary[msk_ary == 128] = 1
        else:
            raise NotImplementedError(f"segObj is not supported: {self.segObj}")
        try:
            if msk_ary.sum() == 0:
                return self.__getitem__(index+1)
        except Exception:
            pass
        t2 = time.time()
        t_d[str(2)] = t2-t1
        
            
    
        img_ary = np.load(img_path)
        
        t3 = time.time()
        t_d[str(3)] = t3-t2
        
        
        
        # if self.shift:
        alpha = np.random.rand()
        if alpha < self.shift:
            threshs = [25, 50, 75]
            thresh = random.choice(threshs)
            bool_mask = (img_ary > thresh).astype(np.float32)
            
            supp_img_ary = img_ary.copy()
            supp_img_ary -= thresh
            supp_img_ary /= (255.0 - thresh)
            supp_img_ary *= bool_mask
            supp_img_ary *= 50
            
            img_ary = img_ary * (1 - bool_mask) + supp_img_ary
        
        t4 = time.time()
        t_d[str(4)] = t4-t3
        
        
        # print(f"[BEFORE TRANS]: {img_ary.shape}, {msk_ary.shape}")
        if self.roundbox:
            img, label = self.roundbox_transform(img_ary, msk_ary)
        else:
            img, label = self.transform(img_ary, msk_ary)
        
        # print(f"[AFTER TRANS]: {img.shape}, {label.shape}")
        t5 = time.time()
        t_d[str(5)] = t5-t4
        
        
        # from torchvision.transforms.functional import adjust_brightness
        # import torchshow as ts
        
        # coin = np.random.rand()
        # if coin <= 0.05:
        
        #     ts.save(torch.FloatTensor(msk_ary).unsqueeze(0).unsqueeze(0), osp.join(self.save_dir, osp.basename(img_path).split('.')[0] + '_' + "msk_ary.png"))
        #     ts.save(adjust_brightness(torch.FloatTensor(img_ary).unsqueeze(0), brightness_factor=0.15), osp.join(self.save_dir, osp.basename(msk_path).split('.')[0] + '_' + "img_ary_adjust_brightness.png"))
        #     ts.save(torch.FloatTensor(img_ary).unsqueeze(0), osp.join(self.save_dir, osp.basename(msk_path).split('.')[0] + '_' + "img_ary.png"))
            

            # ts.save(label.unsqueeze(0), "_torchshow/msk_trans_tensor.png")
            # ts.save(adjust_brightness(img.unsqueeze(0), brightness_factor=2), "_torchshow/img_trans_tensor.png")
        # pprint(t_d)
        return img, label, img_path, msk_path

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                # ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=self.args.random_crop,
                                #  patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label

    def roundbox_transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            if self.args.spatial_aug and self.args.bright_aug:
                aug_list = [  # CenterCropTransform(crop_size=target_size),
                    BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                    GammaTransform(p_per_sample=0.5),
                    GaussianNoiseTransform(p_per_sample=0.5),
                    MirrorTransform(axes=(1,)),
                    SpatialTransform(patch_size=self.patch_size, random_crop=self.args.random_crop,
                                    do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                    do_rotation=True, p_rot_per_sample=0.5,
                                    angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                    scale=(0.95, 1.05), p_scale_per_sample=0.5,
                                    border_mode_data="nearest", border_mode_seg="nearest"),
                    NumpyToTensor(),
                ]
            elif self.args.spatial_aug:
                aug_list = [  # CenterCropTransform(crop_size=target_size),
                    BrightnessTransform(mu=1, sigma=1, p_per_sample=0.0),
                    GammaTransform(p_per_sample=0.0),
                    GaussianNoiseTransform(p_per_sample=0.0),
                    MirrorTransform(axes=(1,)),
                    SpatialTransform(patch_size=self.patch_size, random_crop=self.args.random_crop,
                                    do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                    do_rotation=True, p_rot_per_sample=0.5,
                                    angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                    scale=(0.95, 1.05), p_scale_per_sample=0.5,
                                    border_mode_data="nearest", border_mode_seg="nearest"),
                    NumpyToTensor(),
                ]
            elif self.args.bright_aug:
                aug_list = [  # CenterCropTransform(crop_size=target_size),
                    BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                    GammaTransform(p_per_sample=0.5),
                    GaussianNoiseTransform(p_per_sample=0.5),
                    MirrorTransform(axes=(1,)),
                    SpatialTransform(patch_size=self.patch_size, random_crop=self.args.random_crop,
                                    do_elastic_deform=False, p_el_per_sample=0.0,
                                    do_rotation=False, p_rot_per_sample=0.0,
                                    do_scale=False, p_scale_per_sample=0.0,        
                                    border_mode_data="nearest", border_mode_seg="nearest"),
                    NumpyToTensor(),
                ]
            else:
                aug_list = [  # CenterCropTransform(crop_size=target_size),
                    MirrorTransform(axes=(1,)),
                    SpatialTransform(patch_size=self.patch_size, random_crop=self.args.random_crop,
                                    do_elastic_deform=False, p_el_per_sample=0.0,
                                    do_rotation=False, p_rot_per_sample=0.0,
                                    do_scale=False, p_scale_per_sample=0.0,        
                                    border_mode_data="nearest", border_mode_seg="nearest"),
                ]
            aug = Compose(aug_list)
        else:
            aug_list = [
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                    do_elastic_deform=False, p_el_per_sample=0.0,
                                    do_rotation=False, p_rot_per_sample=0.0,
                                    do_scale=False, p_scale_per_sample=0.0,        
                                    border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label


class WSIDatasetPrefetchCropOmitNegative(Dataset):
    def __init__(self, args, mode='train', use_aux_data=False):
        super().__init__()
        self.patch_size = (args.img_size, args.img_size)
        self.mode = mode
        self.use_aux_data = use_aux_data

        # sub_grid = '_2dgrid_' + \
        #     str(self.patch_size[0]) + \
        #     'x' + \
        #     str(self.patch_size[1]) + \
        #     '_fold_' + \
        #     str(args.fold) + \
        #     f'_{args.per_image_sample}_per_sample' + \
        #     f'_scale_{args.scale}' + \
        #     f'_instance_norm_{args.instance_norm}' + \
        #     f'_min_area_{args.min_area}'

        # self.sub_vol_path = osp.join(
        #     r'/home/gc/workbench/wsiworkbench/MedicalZooPytorch/datasets',
        #     'PrivateWSI_898/generated/' + mode + sub_grid + '/'
        # )

        # print(f"[warning]:{self.sub_vol_path}...")
        
        anno_dict = None
        with open(args.split_file, 'r') as f:
            anno_dict = json.load(f)
            
        samples = anno_dict[str('0')][self.mode]

        image_paths = [item[0] for item in samples]
        label_paths = [item[1] for item in samples]
        
        # if self.use_aux_data:
        #     print(f"{self.mode=}, {len(image_paths)=}")
        #     aux_image_paths = glob(osp.join(self.sub_vol_path.replace('PrivateWSI_898', 'PrivateWSI_129'), "**input.npy"))
        #     aux_label_paths = glob(osp.join(self.sub_vol_path.replace('PrivateWSI_898', 'PrivateWSI_129'), "**label.npy"))
            
        #     image_paths += aux_image_paths
        #     label_paths += aux_label_paths
        #     print(f"{self.mode=}, {len(image_paths)=}")
        #     print("="*40)

        # image_paths.sort()
        # label_paths.sort()

        self.imgs = image_paths
        self.msks = label_paths
        
        print(f'dataset length: {len(self.imgs)}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        
        img_path, msk_path = self.imgs[index], self.msks[index]

        # img_ary = np.load(img_path)
        # msk_ary = np.load(msk_path)
        
        img_obj = Image.open(img_path)
        msk_obj = Image.open(msk_path)

        img_ary = np.array(img_obj)
        msk_ary = np.array(msk_obj)

        img, label = self.transform(img_ary, msk_ary)

        from torchvision.transforms.functional import adjust_brightness
        import torchshow as ts
        # ts.save(torch.FloatTensor(msk_ary).unsqueeze(0).unsqueeze(0), "_torchshow/msk_ary.png")
        # ts.save(adjust_brightness(torch.FloatTensor(img_ary).unsqueeze(0), brightness_factor=2), "_torchshow/img_ary.png")

        # ts.save(label.unsqueeze(0), "_torchshow/msk_trans_tensor.png")
        # ts.save(adjust_brightness(img.unsqueeze(0), brightness_factor=2), "_torchshow/img_trans_tensor.png")

        return img, label, img_path, msk_path

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=True,
                                #  patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)

        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label
