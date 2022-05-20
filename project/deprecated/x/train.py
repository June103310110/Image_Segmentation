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
device = 'cuda:0'


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


class conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.cell=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        return self.cell(x)
    
class DomainClassifier(nn.Module): # input B, 1024, 16*16
    def __init__(self,ch_in=1):
        super().__init__()
        ftrs_ch = 32
        self.blocks = nn.Sequential(
          conv(ch_in,ftrs_ch),
          nn.AdaptiveAvgPool2d((1,1))
        )
        self.output = nn.Sequential(
          nn.Flatten(),
          nn.Linear(ftrs_ch, 128),
          nn.ReLU(),
          nn.Linear(128, 1), 
        )
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.output(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, out_sz, out_channels=3, activation=None, multi_level=0):
        super().__init__()
        self.FeatureExtractor = FeatureExtractor(enc_chs=(3*2,64,128,256))
        LP = LabelPredictor(out_sz=out_sz, dec_chs=(256, 128, 64),
                            activation=activation, multi_level=multi_level)
        LP.head = nn.Conv2d(64//SCALE, out_channels, 1)
        self.LabelPredictor = LP
        
    def forward(self, x, domain_label):
        x = torch.cat([x, domain_label], dim=1)

        x = self.FeatureExtractor(x)
        x, _ = self.LabelPredictor(x)
        
        return x, _


# In[21]:


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
 
model = UNet
model = model((WIDTH, HEIGHT), in_ch=1, out_ch=1, activation=None).to(device)
domain_classifier = DomainClassifier().to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-1)
optimizer_disc = optim.Adam(domain_classifier.parameters(), lr = 1e-1)

domain_criterion = nn.BCEWithLogitsLoss()
class_criterion = DiceLoss()


# In[24]:


'''
abs: training model
---
- 下面的程序會在3種條件下儲存當前的model(只包含weight)
    - 當前模型的loss是目前以來最低
    - 當前epoch數是20的倍數
    - 完成一個epoch的訓練
'''
from torchvision.ops import sigmoid_focal_loss
EPOCHS = 20
DISC_LOOPS = 4
lamb = 0.1
min_target_loss_value = float("inf") 
save_root = './data/save_weights/'
os.makedirs(save_root, exist_ok=True)


# In[25]:

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchvision.ops import sigmoid_focal_loss

# class unetModel(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.encoder = model
# #         self.decoder = decoder

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         sx, sy = batch[0]
#         tx, ty = batch[1]
#         outputs = self.encoder(torch.cat([sx,tx]))

#         loss = sigmoid_focal_loss(outputs, torch.cat([tx,ty]), reduction='sum')
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
    
#     def forward(self, x):
#         return self.encoder(x)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_utils import unetModel

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

optimizer = optim.Adam(model.parameters(), lr = 1e-1)


# In[18]:


'''
torch-lightning
''' 
save_root = './data/save_weights/'
model = unetModel(model)

# train model
# try:
checkpoint_callback = ModelCheckpoint(monitor='train_loss',
            dirpath=save_root,
            filename='{epoch}_{train_loss:.6f}_model',
            every_n_epochs = 20,
            save_on_train_epoch_end = True,
            mode = 'min',
            save_weights_only = True,

            )

trainer = pl.Trainer(devices=2, accelerator="gpu", strategy='ddp', callbacks=[checkpoint_callback], max_epochs=2)
trainer.fit(model=model, train_dataloaders=[dataloader_train,dataloader_train])
