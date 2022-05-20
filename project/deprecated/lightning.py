#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -q --user albumentations
# !pip3 install pydicom


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import cv2
import numpy as np
import torch # 1.9
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os
import torch.optim as optim
import time
import ipywidgets as widgets
import pickle


# 導入dicom套件
from pydicom import dcmread
from pydicom.data import get_testdata_files


# In[4]:


from utils.dataset import getAllDataPath, CustomImageDataset, show_image
from utils.unet import UNet, ResUnet, AttUnet
from utils.loss import DiceLoss, FocalLoss



# In[5]:


BATCH_SIZE = 4
WIDTH = 256
HEIGHT = 256
# device = 'cuda:0'


# In[6]:


# https://albumentations.ai/docs/getting_started/mask_augmentation/

transform = A.Compose([
    A.Resize(WIDTH, HEIGHT),
])

target_transform = A.Compose([                       
    A.Resize(WIDTH, HEIGHT),
])


# ## 資料整理與處理

# In[17]:


root = './data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/CT/'
CT_data = getAllDataPath(root, test_split_size=0.2)
root = './data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/MRI/MRI_Label/'
MRI_data = getAllDataPath(root, test_split_size=0.2)
root = './data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/MRI/MRI_nonLabel/'
MRI_nlb_data = getAllDataPath(root, test_split_size=None, imgOnly=True)

for data in ['CT_data', 'MRI_data', 'MRI_nlb_data']:
    i = eval(data)
    for k in i.keys():
        print(data,k, np.shape(i[k]))


dataset_train = CustomImageDataset(MRI_data['train'], transform=transform, pseudo_label=False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dataset_test = CustomImageDataset(MRI_data['test'], transform=target_transform, pseudo_label=False) 
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

dataset_nlb_train = CustomImageDataset(MRI_nlb_data['train'], transform=transform, pseudo_label=False)
dataloader_nlb_train = torch.utils.data.DataLoader(dataset_nlb_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

CT_dataset_train = CustomImageDataset(CT_data['train'], transform=transform, pseudo_label=False)
CT_dataloader_train = torch.utils.data.DataLoader(CT_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

CT_dataset_test = CustomImageDataset(CT_data['test'], transform=target_transform)
CT_dataloader_test = torch.utils.data.DataLoader(CT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)


# In[18]:


'''
Abs: test transform on dataloader_train.
---
take first image in every batch.
'''
for data in dataloader_train:
    for x, y in zip(*data): 
        print(x.shape, y.shape)
        print(np.histogram(x.numpy()), y.unique())
    
        show_image(x.squeeze(0).numpy(), y.squeeze(0).numpy())
        break
    break


# ## process1: share weight

# In[19]:

# In[21]:


'''
abs: training model
---
- 下面的程序會在3種條件下儲存當前的model(只包含weight)
    - 當前模型的loss是目前以來最低
    - 當前epoch數是20的倍數
    - 完成一個epoch的訓練
'''
from torchvision.ops import sigmoid_focal_loss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from lightning_utils import unetModel, discModel, unetWithDiscModel
from models import DomainClassifier

# time 
from datetime import datetime, timezone, timedelta
def now():
    # 設定為 +8 時區
    tz = timezone(timedelta(hours=+8))

    now = datetime.now(tz)
    date_time = now.strftime("%m%d-%H%M")
    return date_time

# In[17]:


'''
title: create model
---
補充:
- 要在建立optimizer之前就把model的參數移到gpu裡面(也就是在把參數託管給optim以前)
ref: 
- https://pytorch.org/docs/stable/optim.html 
- Road Extraction by Deep Residual U-Net, 2017
- U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015
- Attention U-Net: Learning Where to Look for the Pancreas, 2018
'''

model_name = 'UNet' 
# model_name = 'ResUnet'  # Note: Sigmoid activation, Dice loss or focal loss
# model_name = 'AttUnet'  # better ResUnet 
model = eval(model_name)
model = model((WIDTH, HEIGHT), in_ch=1, out_ch=1, activation=None)#.to(device)
discrimator = DomainClassifier()
# optimizer = optim.Adam(model.parameters(), lr = 1e-1)


# In[18]:


'''
torch-lightning
''' 
save_root = './data/save_weights/'
os.makedirs(save_root, exist_ok=True)
# model = unetModel(model)
model = discModel(model, discrimator, early_stop=0.6)
# model = unetWithDiscModel(model, discrimator, lamb=0.1)
# train model
# try:
checkpoint_callback = ModelCheckpoint(monitor='train_loss',
            dirpath=save_root,
            filename=now()+'_{epoch:03d}_{train_loss:.4f}_model',
            every_n_epochs = 20,
            save_on_train_epoch_end = True,
            mode = 'min',
            save_weights_only = True,
            save_top_k = -1,
            )


# disc_model = discModel(model, discrimator, early_stop=0.6)
# unet_model = unetWithDiscModel(model, discrimator, lamb=0.1)
# for i in range(10):
    
#     trainer = pl.Trainer(devices=1, accelerator="gpu", strategy='ddp', callbacks=[checkpoint_callback], max_epochs=10)
#     trainer.fit(model=model, train_dataloaders=[CT_dataloader_train, dataloader_train])
#     print(model.early_stop)
# early_stop_callback = EarlyStopping(monitor="train_loss", 
#                                     min_delta=0.00,
#                                     patience=3,
#                                     verbose=False,
#                                     stopping_threshold=0.5,
#                                     mode="min")
# trainer = pl.Trainer(devices=1, accelerator="gpu", strategy='ddp', callbacks=[checkpoint_callback, early_stop_callback], max_epochs=2)
# trainer.fit(model=unet_disc_model, train_dataloaders=(CT_dataloader_train, dataloader_train))



trainer = pl.Trainer(devices=1, accelerator="gpu", strategy='ddp', callbacks=[checkpoint_callback], max_epochs=10)
trainer.fit(model=model, train_dataloaders=[CT_dataloader_train, dataloader_train])
# print(model.early_stop)