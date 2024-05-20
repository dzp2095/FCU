# encoding: utf-8

"""
CXR14
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import logging

import albumentations
from albumentations.pytorch import ToTensorV2

class_names =  ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

class_name_to_id = {}
for i, each_label in enumerate(class_names):
    class_id = i  
    class_name = each_label
    class_name_to_id[class_name] = class_id


class ISICDataset(Dataset):
    def __init__(self, mode, cfg):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ISICDataset, self).__init__()
        csv_file = cfg['dataset'][mode]

        self.img_names, self.labels = self.__load_imgs__(csv_file)
        if mode == 'train':
            self.transform = albumentations.Compose([
                albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.OneOf([
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85,),
                albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
                ToTensorV2()
                ])
            
        elif mode == 'test' or mode == 'val' or mode == 'forgotten' or mode == 'remembered':
            self.transform = albumentations.Compose([
                albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
                albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
                ToTensorV2()
            ])
        else:
            raise ValueError('mode should be train, test, val, forgotten or remembered')
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        path = self.img_names[index]
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
        label = self.labels[index]
        image = self.transform(image=image)['image']
        return index, image, label

    def __len__(self):
        return len(self.img_names)

    def __load_imgs__(self, csv_path):
        data = pd.read_csv(csv_path)
        imgs = data['path'].values
        # convert label to one-hot
        onehots = []
        for i, row in data.iterrows():
            onehot = np.zeros(len(class_name_to_id), dtype=np.float32)
            for label in class_names:
                if row[label]:
                    onehot[class_name_to_id[label]] = 1
            onehots.append(onehot)
        labels = onehots
        logging.info(f'Total # images:{len(imgs)}, labels:{len(labels)}')
        return imgs, labels