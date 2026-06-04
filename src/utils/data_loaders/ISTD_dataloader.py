import os
import cv2
import torch
import random
import numpy as np
from einops import rearrange
from tqdm import tqdm

from typing import List, Union, Optional

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

import torchvision.transforms as T

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class ISTDDataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(ISTDDataLoaderTrain, self).__init__()
        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps
        inp_path = self.inp_filenames[index_]
        mask_path = self.mask_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        mask_img = Image.open(mask_path)
        tar_img = Image.open(tar_path)
        inp_img = inp_img.convert('HSV')
        tar_img = tar_img.convert('HSV')

        inp_h, inp_s, inp_v = inp_img.split()
        tar_h, tar_s, tar_v = tar_img.split()

        h_offset = random.randint(-8, 8)
        s_scale = random.uniform(0.9, 1.1)
        v_scale = random.uniform(0.9, 1.1)

        inp_h = inp_h.point(lambda p: (p + h_offset) % 256)
        inp_s = inp_s.point(lambda p: min(255, max(0, int(p * s_scale))))
        inp_v = inp_v.point(lambda p: min(255, max(0, int(p * v_scale))))

        tar_h = tar_h.point(lambda p: (p + h_offset) % 256)
        tar_s = tar_s.point(lambda p: min(255, max(0, int(p * s_scale))))
        tar_v = tar_v.point(lambda p: min(255, max(0, int(p * v_scale))))

        inp_img = Image.merge('HSV', (inp_h, inp_s, inp_v)).convert('RGB')
        tar_img = Image.merge('HSV', (tar_h, tar_s, tar_v)).convert('RGB')

        if random.random() < 0.3:
            blur = T.GaussianBlur(kernel_size=3)
            inp_img = blur(inp_img)
            tar_img = blur(tar_img)

        if random.random() < 0.5:
            mask_np = np.array(mask_img.convert("L"))
            inp_np = np.array(inp_img)
            ys, xs = np.where(mask_np < 128)
            if len(xs) > 0:
                idx = random.randint(0, len(xs) - 1)
                cx, cy = xs[idx], ys[idx]

                erase_size = 10
                x1, y1 = max(0, cx - erase_size), max(0, cy - erase_size)
                x2, y2 = min(cx + erase_size, inp_np.shape[1]), min(cy + erase_size, inp_np.shape[0])

                inp_np[y1:y2, x1:x2] = 0

                inp_img = Image.fromarray(inp_np)

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            mask_img = TF.pad(mask_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        mask_img = TF.to_tensor(mask_img)
        tar_img = TF.to_tensor(tar_img)
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        mask_img = mask_img[:, rr:rr + ps, cc:cc + ps]  # Crop mask image
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        if aug == 1:
            inp_img = inp_img.flip(1)
            mask_img = mask_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            mask_img = mask_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            mask_img = torch.rot90(mask_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            mask_img = torch.rot90(mask_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            mask_img = torch.rot90(mask_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, mask_img, filename

class ISTDDataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(ISTDDataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if
                               is_image_file(x)]  # Load mask files
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.img_options = img_options
        self.sizex = len(self.tar_filenames)
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps
        inp_path = self.inp_filenames[index_]
        mask_path = self.mask_filenames[index_]
        tar_path = self.tar_filenames[index_]
        inp_img = Image.open(inp_path)
        mask_img = Image.open(mask_path)
        tar_img = Image.open(tar_path)

        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            mask_img = TF.center_crop(mask_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))


        inp_img = TF.to_tensor(inp_img)
        mask_img = TF.to_tensor(mask_img)
        tar_img = TF.to_tensor(tar_img)  # shape: [3, H, W]

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img, mask_img, filename

class RandomDataLoader(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(RandomDataLoader, self).__init__()
        self.img_options = img_options
        self.len = 100
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        tar_img = torch.ones((3, self.ps, self.ps))
        return tar_img, tar_img, ''
