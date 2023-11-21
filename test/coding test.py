import pandas as pd
import numpy as np
import cv2
from easydict import EasyDict
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import CrossEntropyLoss, NLLLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import matplotlib.pyplot as plt
# %matplotlib inline

train = pd.read_csv("D:/Jupyter Notebook/DeepLearning/Data/digit-recognizer/train.csv")
test = pd.read_csv("D:/Jupyter Notebook/DeepLearning/Data/digit-recognizer/test.csv")
submission = pd.read_csv("D:/Jupyter Notebook/DeepLearning/Data/digit-recognizer/sample_submission.csv")

train_images = train.iloc[:,1:].values.reshape(-1,28,28)
train_labels = train.iloc[:,0].values
test_images = test.values.reshape(-1,28,28)



# 图片增强
def get_transform(image_size, train=True):
    # 训练阶段使用
    if train:
        return A.Compose([
            #                     A.RandomCrop(width=22, height=22), # 随即裁剪 可能是版本和opencv的版本冲突
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.VerticalFlip(p=0.5),  # 垂直翻转
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),  # 形状统一必不可少
            A.Normalize(0.1310, 0.3085),  # 标准化，
            ToTensorV2()  # 把数据转化为Pytorch格式
        ])
    # 测试阶段使用
    else:
        return A.Compose([
            A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),  # 取决于测试集的图片大小，如果和训练集一样大，可以不要
            A.Normalize(0.1310, 0.3085),  # 标准化
            ToTensorV2()  # 把数据转化为Pytorch格式
        ])
train_transform = get_transform((28,28),True) # 方法初始化,此时train_transform为一个图像变换的操作

# train_images[0].shape = (28,28)
train_transform(image = train_images[0]) # 赋值，传入变量,此处仅传入一个变量



# Dataset
class MiniDataSet(Dataset):

    # 传入数据
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    # 获取长度
    def __len__(self):
        return len(self.images)

    # 提取列表数据
    def __getitem__(self, idx):
        ret = {}  # 返回是一个字典
        img = self.images[idx]

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        ret["image"] = img

        # 假如输入是有label的
        if self.labels is not None:
            ret["label"] = self.labels[idx]
        return ret

ds = MiniDataSet(train_images, train_labels, get_transform((28, 28), True))
a = ds[0]["label"] # 取字典中的量