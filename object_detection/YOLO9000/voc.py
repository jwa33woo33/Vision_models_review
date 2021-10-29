import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VOCDataset(Dataset):
    def __init__(self, data_list, mode='train', transforms=None):
        self.data_list = data_list
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        record = self.data_list[idx]
        img_id = record['image_id']
        bboxs = record['bboxs']
        labels = record['labels']

        img = cv2.imread(image_f.format(self.mode, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            sample = self.transforms(image=img, bboxes=bboxs, category_ids=labels)
            image = sample['image']
            bboxs = np.asarray(sample['bboxes'])
            labels = np.asarray(sample['category_ids'])

        if self.mode=='train':
            target = np.c_[bboxs, labels].astype(np.float32)
            return image, target
        else: 
            return image

def get_train_transforms():
    return A.Compose([
        A.Resize(*image_size, always_apply=True, p=1),
        A.ColorJitter(),
        A.Flip(),
        ToTensor(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))
# transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])

def get_test_transforms():
    return A.Compose([
        A.Resize(*image_size, always_apply=True, p=1),
        ToTensor(),
    ])