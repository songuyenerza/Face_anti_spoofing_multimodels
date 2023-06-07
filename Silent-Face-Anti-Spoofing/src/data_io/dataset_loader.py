# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT, DatasetFolderFT_val
from src.data_io import transform as trans

def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.7, 1.2)),
        
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.3, hue=0.1),
        trans.RandomRotation(25),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    # root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    root_path = conf.train_root_path
    trainset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return train_loader

def get_val_loader(conf, root_path):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.7, 1.2),ratio=(4. / 5., 5. / 4.)),
        
        trans.ColorJitter(brightness=0.2,
                          contrast=0.2, saturation=0.2, hue=0.1),
        trans.RandomRotation(15),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    # root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    # print(conf)
    # print( "root_path" , root_path)
    trainset = DatasetFolderFT_val(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return train_loader
