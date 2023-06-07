'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import argparse

import albumentations as A
import imgaug.augmenters as iaa
import cv2 as cv
import torch
import torch.nn as nn
import cv2
from trainer import Trainer
from utils import (Transform, build_criterion, build_model, make_dataset,
                   make_loader, make_weights, read_py_config)
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
import random

#add augment new


def random_crop(img):
    # add argument random rop
    image = Image.fromarray(img.copy())

    pct_focusx = random.uniform(0,0.15)
    pct_focusy = random.uniform(0,0.15)
    x, y = image.size
    image = image.crop((x*pct_focusx, y*pct_focusy, x*(1-pct_focusx), y*(1-pct_focusy)))
    image = np.array(image)
    return image

def noisy(img):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    image=img.copy() 
    mean=0
    st=0.7
    gauss = np.random.normal(mean,st,image.shape)
    gauss = gauss.astype('uint8')
    image = cv2.add(image,gauss)
    return image

def Blur(img):
    image=img.copy()
    fsize = 5
    return cv2.GaussianBlur(image, (fsize, fsize), 0)

def hight_blur(img):
    ksize = random.randint(5, 20)
    # Using cv2.blur() method 
    image = cv2.blur(img, (ksize, ksize), cv2.BORDER_DEFAULT) 
    return image

def scale(img):
    image = img.copy()
    scale = random.randint(1, 3)
    w, h = img.shape[0] , img.shape[1]
    dim = (int(h/scale), int(w/scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def colorjitter(img, cj_type="b"):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 0, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 0,  30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(10, 60)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

class FASDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, is_train=True, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        # self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.data = pd.read_csv( csv_file)

        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        img_path = os.path.join(self.root_dir, img_name)

        #add augmetn data

        img = cv2.imread(img_path)

        #///////update augment imgs///////

        # random bright, saturation, contrast    blur
        i = random.uniform(0,1)
        if i > 0.7:
            img = hight_blur(img)

        i = random.uniform(0,1)
        if i > 0.5:
            img = colorjitter(img, cj_type = 'b')
        if i > 0.7:
            img = colorjitter(img, cj_type = 's')
        if i > 0.8:
            img = colorjitter(img, cj_type = 'c')

        i = random.uniform(0,1)
        if i > 0.7:
            img = Blur(img)
        i = random.uniform(0,1)
        if i > 0.5:
            img = random_crop(img)

        i = random.uniform(0,1)
        if i > 0.6:
            img = scale(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = int(label.astype(np.float32))
        img1 = self.transform( image = img)['image']
        img1 = np.transpose(img1, (2, 0, 1)).astype(np.float32)

        return torch.tensor(img1), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

class FASDataset_tester(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, is_train=True, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        # self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.data = pd.read_csv(csv_file)

        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        # img_name = os.path.join(self.root_dir, "images", img_name
        img_path = os.path.join(self.root_dir, img_name)
        # print("check ====img" , img_name)
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)

        label = int(label.astype(np.float32))
        # label = np.expand_dims(label, axis=0)
        # print("label====", torch.tensor(label, dtype=torch.long))
        # if self.transform:
        img1 = self.transform( image = img)['image']
        # print("img1", img1)

        img1 = np.transpose(img1, (2, 0, 1)).astype(np.float32)
        # print("img1", img1.shape)
        return torch.tensor(img1), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    parser.add_argument('--config', type=str, default="./configs/config.py", required=True,
                        help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='if you want to train model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # manage device, arguments, reading config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    if config.data_parallel.use_parallel:
        device = f'cuda:{config.data_parallel.parallel_params.output_device}'
    # if config.multi_task_learning and config.dataset != 'celeba_spoof':
    #     raise NotImplementedError(
    #         'Note, that multi task learning is avaliable for celeba_spoof only. '
    #         'Please, switch it off in config file'
    #         )
    # launch training, validation, testing
    train(config, device, args.save_checkpoint)

def train(config, device='cuda:0', save_chkpt=True):
    ''' procedure launching all main functions of training,
        validation and testing pipelines'''
    
    # for pipeline testing purposes
    save_chkpt = False if config.test_steps else True

    # preprocessing data
    normalize = A.Normalize(**config.img_norm_cfg)
    train_transform_real = A.Compose([
                        A.Rotate(limit = 30, p = 0.5),
                        A.HorizontalFlip(p=0.5),
                        A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.3),

                        A.augmentations.transforms.CLAHE(clip_limit=2.0, tile_grid_size=(6, 6), always_apply=False, p=0.15),
                        # A.augmentations.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3, hue=0.3, always_apply=False, p=0.2),
                        A.augmentations.transforms.FancyPCA(alpha=0.1, always_apply=False, p=0.15),
                        A.augmentations.transforms.ISONoise(color_shift=(0.01,0.2),
                                                            intensity=(0.1, 0.4), p=0.2),
                        A.augmentations.transforms.JpegCompression (quality_lower=50, quality_upper=100, always_apply=False, p=0.2),
                        A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.3,
                                                                            contrast_limit=0.3,
                                                                            brightness_by_max=True,
                                                                            always_apply=False, p=0.2),
                    
                        A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),

                        normalize
                        ])

    val_transform_real = A.Compose([
                A.Rotate(limit = 25, p = 0.5),
                A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.3),

                A.augmentations.transforms.ISONoise(color_shift=(0.01,0.1),
                                                                intensity=(0.1, 0.2), p=0.2),
                A.augmentations.transforms.MotionBlur(blur_limit=3, p=0.3),
                A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,
                                                                    contrast_limit=0.2,
                                                                    brightness_by_max=True,
                                                                    always_apply=False, p=0.3),

                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                normalize
                ])

    # load data
    sampler = config.data.sampler
    if sampler:
        num_instances, weights = make_weights(config)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_instances, replacement=True)

    path_data = config.datasets_train_turning['folder_train']
    label_data = config.datasets_train_turning['path_train']

    path_data_test = config.datasets_train_turning['folder_val']
    label_test = config.datasets_train_turning['path_val']

    trainset = FASDataset(
        root_dir= path_data,
        transform=train_transform_real,
        csv_file= label_data,
        is_train=True
    )
    valset = FASDataset_tester(
        root_dir= path_data_test,
        transform=val_transform_real,
        csv_file= label_test,
        is_train=True
    )

    batsize = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size= batsize,
        shuffle=True,
        pin_memory = True,
        num_workers = 8
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size= batsize,
        shuffle=True,
        pin_memory = True,

        num_workers=8
    )
    test_loader = val_loader
    print("ok done data")
 

    # build model and put it to cuda and if it needed then wrap model to data parallel
    model = build_model(config, device=device, strict=False, mode='train')

    #///////fine tune model
    '''i = 0
    for param in model.parameters():
        i += 1
        if 10 < i < 159:
            param.requires_grad = False'''
    # for name, param in model.named_parameters():
    #     i+= 1
    # # if param.requires_grad:
    #     print(i, name)

    #fine tune nodel done =======================

    model.to(device)
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)

    # build a criterion
    softmax = build_criterion(config, device, task='main').to(device)
    cross_entropy = build_criterion(config, device, task='rest').to(device)
    bce = nn.BCELoss().to(device)
    criterion = (softmax, cross_entropy, bce) if config.multi_task_learning else softmax

    # build optimizer and scheduler for it
    optimizer = torch.optim.SGD(model.parameters(), **config.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.scheduler)

    # create Trainer object and get experiment information
    trainer = Trainer(model, criterion, optimizer, device, config, train_loader, val_loader, test_loader)
    trainer.get_exp_info()

    # learning epochs
    for epoch in range(config.epochs.start_epoch, config.epochs.max_epoch):
        if epoch != config.epochs.start_epoch:
            scheduler.step()

        # train model for one epoch
        train_loss, train_accuracy = trainer.train(epoch)
        print(f'epoch: {epoch}  train loss: {train_loss}   train accuracy: {train_accuracy}')
        trainer.save_model(epoch)

        # validate your model
        accuracy = trainer.validate()

        # eval metrics such as AUC, APCER, BPCER, ACER on val and test dataset according to rule
        trainer.eval(epoch, accuracy, save_chkpt=save_chkpt)
        # for testing purposes
        if config.test_steps:
            exit()

    # evaluate in the end of training
    # if config.evaluation:
    #     file_name = 'tests.txt'
    #     trainer.test(file_name=file_name)


if __name__=='__main__':
    main()
