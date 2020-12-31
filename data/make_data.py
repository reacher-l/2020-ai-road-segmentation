import cv2
import pdb
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from PIL import Image, ImageOps, ImageFilter
import random
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from .edge_utils import *
class GaofenTrain(data.Dataset):
    def __init__(self, root, list_path,  crop_size=(640, 640),
                 scale=True, mirror=True,rotation=True, bright=False, ignore_label=1, use_aug=True, network='resnet101'):
        self.root = root
        self.src_h = 1024
        self.src_w = 1024
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.bright = bright
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.rotation = rotation
        self.use_aug = use_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.network = network
        for item in self.img_ids:
            image_path = 'images/'+item.split('_')[0]+'/'+item
            label_path = 'labels/'+item.split('_')[0]+'/'+item
            name = item
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def random_brightness(self, img):
        if random.random() < 0.5:
            return img
        self.shift_value = 10 #取自HRNet
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0 # [0.5, 1.5]
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"],0)
        #旋转90/180/270
        if self.rotation and random.random() > 0.5:
            angel = np.random.randint(1,4)
            M = cv2.getRotationMatrix2D(((self.src_h - 1) / 2., (self.src_w - 1) / 2.), 90*angel, 1)
            image = cv2.warpAffine(image, M, (self.src_h, self.src_w), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, M, (self.src_h, self.src_w), flags=cv2.INTER_NEAREST, borderValue=self.ignore_label)
        # 旋转-30-30
        if self.rotation and random.random() > 0.5:
            angel = np.random.randint(-30,30)
            M = cv2.getRotationMatrix2D(((self.src_h - 1) / 2., (self.src_w - 1) / 2.), angel, 1)
            image = cv2.warpAffine(image, M, (self.src_h, self.src_w), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, M, (self.src_h, self.src_w), flags=cv2.INTER_NEAREST, borderValue=self.ignore_label)
        size = image.shape
        if self.scale: #尺度变化
            image, label = self.generate_scale_label(image, label)
        if self.bright: #亮度变化
            image = self.random_brightness(image)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        mean = (0.355403, 0.383969, 0.359276)
        std = (0.206617, 0.202157, 0.210082)
        image /= 255.
        image -= mean
        image /= std

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))  #边界填充的是ignore
        else:
            img_pad, label_pad = image, label
            
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1)) # 3XHXW

        if self.is_mirror: #水平/垂直翻转
            flip1 = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip1]
            label = label[:, ::flip1]
            flip2 = np.random.choice(2) * 2 - 1
            image = image[:,::flip2, :]
            label = label[::flip2,:]
        oneHot_label = mask_to_onehot(label,2) #edge=255,background=0
        edge = onehot_to_binary_edges(oneHot_label,2,2)
        # 消去图像边缘
        edge[:2, :] = 0
        edge[-2:, :] = 0
        edge[:, :2] = 0
        edge[:, -2:] = 0
        return image.copy(), label.copy(), edge,np.array(size), datafiles

class GaofenVal(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                 scale=False, mirror=False, ignore_label=255, use_aug=True, network="renset101"):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.network = network
        for item in self.img_ids:
            image_path = 'images/'+item.split('_')[0]+'/'+item
            label_path = 'labels/'+item.split('_')[0]+'/'+item
            name = item
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {}

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"],0)

        size = image.shape

        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        mean = (0.355403, 0.383969, 0.359276)
        std = (0.206617, 0.202157, 0.210082)
        image /= 255.
        image -= mean
        image /= std

        img_h, img_w = label.shape

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        image = image.transpose((2, 0, 1)) # 3XHXW
        return image.copy(), label.copy(),label.copy(), np.array(size), datafiles



class GaofenSubmit(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                scale=False, mirror=False, ignore_label=255, use_aug=True, network="renset101"):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.network = network
        for item in self.img_ids:
            image_path = 'images/'+item
            label_path = 'labels/'+item[:-4]+'.png'
            name = item
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {}

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        mean = (0.355403, 0.383969, 0.359276)
        std = (0.206617, 0.202157, 0.210082)
        image /= 255.
        image -= mean
        image /= std

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1)) # 3XHXW
        return image.copy(), np.array(size), name

