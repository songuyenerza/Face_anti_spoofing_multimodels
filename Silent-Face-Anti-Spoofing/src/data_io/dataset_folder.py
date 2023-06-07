# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
from torchvision import datasets
import numpy as np
import random
from PIL import Image

# add augment new

def colorjitter(img, cj_type="b"):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-20, -40, -30, 0, 30, 40, 20]))
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
        value = np.random.choice(np.array([-20, -40, -30, 0,  30, 40, 20]))
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
        contrast = random.randint(10, 50)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

#noise
def noisy(image):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    i = random.uniform(0,1)
    if i > 0.5:
        return cv2.GaussianBlur(image, (3, 3), 0)
    else:
        return cv2.GaussianBlur(image, (5, 5), 0)

def Blur(image):
    i = random.uniform(0,1)

    if i > 0.5:
        return cv2.GaussianBlur(image, (3, 3), 0)
    else:
        return cv2.GaussianBlur(image, (5, 5), 0)

def scale(img):
    image = img.copy()
    scale = random.randint(1, 2)
    w, h = img.shape[0] , img.shape[1]
    dim = (int(h/scale), int(w/scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# def sharpen(img):
#     image = img.copy

def random_crop(img):
    image = Image.fromarray(img.copy())

    pct_focusx = random.uniform(0,0.1)
    pct_focusy = random.uniform(0,0.1)
    x, y = image.size
    image = image.crop((x*pct_focusx, y*pct_focusy, x*(1-pct_focusx), y*(1-pct_focusy)))
    image = np.array(image)
    return image

def equalize(img): #input bgr output = bgr

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])

    image = cv2.cvtColor(img_hsv, cv2.COLOR_YCrCb2BGR)
    return image

def hight_blur(img):
    ksize = random.randint(10, 30)
    # Using cv2.blur() method 
    image = cv2.blur(img, (ksize, ksize), cv2.BORDER_DEFAULT) 
    return image

def opencv_loader(path):
    img = cv2.imread(path)
    #///////update augment imgs///////
    i = random.uniform(0,1)
    if i > 0.7:
        img = hight_blur(img)
    
    i = random.uniform(0,1)
    if i > 0.7:
        img = Blur(img)
    i = random.uniform(0,1)
    if i > 0.8:
        img = scale(img)

    i = random.uniform(0,1)
    if i > 0.5:
        img = random_crop(img)
    if i > 0.9:
        img = equalize(img)

    # random bright, saturation, contrast    
    i = random.uniform(0,1)
    if i > 0.6:
        img = colorjitter(img, cj_type = 'b')
    if i > 0.7:
        img = colorjitter(img, cj_type = 's')
    if i > 0.85:
        img = colorjitter(img, cj_type = 'c')

    return img

def opencv_loader_df(path):

    i = random.uniform(0,1)

    img = cv2.imread(path)
    if i > 0.7:
        img = Blur(img)
    if i > 0.8:
        img = scale(img)

    # random bright, saturation, contrast    
    i = random.uniform(0,1)
    if i > 0.6:
        img = colorjitter(img, cj_type = 'b')
    if i > 0.7:
        img = colorjitter(img, cj_type = 's')
    if i > 0.85:
        img = colorjitter(img, cj_type = 'c')
    # img = cv2.resize(img, (80,80))

    return img

class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=80, ft_height=80, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        # print("check path", path)
        sample = self.loader(path)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, sample, target

class DatasetFolderFT_val(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=80, ft_height=80, loader=opencv_loader_df):
        super(DatasetFolderFT_val, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured ================ : %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg