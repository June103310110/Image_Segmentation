#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import torch # 1.9
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A


# In[2]:


from dataset import getAllDataPath, CustomImageDataset, show_image
from unet import UNet
from loss import DiceLoss, FocalLoss


# In[3]:


BATCH_SIZE = 8
WIDTH = 256
HEIGHT = 256
device = 'cuda:0'


# In[4]:


# https://albumentations.ai/docs/getting_started/mask_augmentation/
transform = A.Compose([
#     A.CenterCrop(300, 900, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
#     A.Rotate((-10, 10), interpolation=0),  

    A.Resize(WIDTH, HEIGHT),
])

target_transform = A.Compose([
    A.Resize(WIDTH, HEIGHT),
])


# In[5]:


path = os.getcwd()
img_dir = f'{path}/data/capsule/test/scratch/'
anno_dir = f'{path}/data/capsule/ground_truth/scratch/'

defective_number = [i.split('.')[0] for i in os.listdir(img_dir)]
print('defective_number: ',defective_number)
data = getAllDataPath(img_dir, anno_dir, test_split_size=0.4)
print(data.keys())
data['train'] = data['train']


# 在這邊會強制對所有不滿BATCH_SIZE的訓練資料做數量上的匹配(單純把路徑複製貼上直到滿足16筆資料)，接著透過CustomImageDataset的transform對16筆資料做資料擴增
if len(data['train']) < 16: 
    lis = data['train']
    lis = [lis[i%len(lis)] for i in range(BATCH_SIZE)]
    data['train'] = lis

dataset_train = CustomImageDataset(data['train'], transform=transform, pseudo_label=True)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, drop_last=True,
                                               shuffle=True, pin_memory=True,
                                              )

dataset_test = CustomImageDataset(data['test'], transform=target_transform, pseudo_label=True) # **如果要正式使用要記得把這裡換成X_test
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, drop_last=False,
                                              shuffle=False, pin_memory=True)


# In[6]:


'''
Abs: test transform on dataloader_train.
---
take first image in every batch.
'''
for data in dataloader_train:
    for x, y in zip(*data): 
        print(x.shape, y.shape)
        print(x.unique().max(), y.unique())
        show_image(x.permute(1,2,0).numpy(), y.squeeze(0).numpy())
#         break 


# In[7]:


'''
title: create model
---
補充:
- 要在建立optimizer之前就把model的參數移到gpu裡面(也就是在把參數託管給optim以前)
ref: https://pytorch.org/docs/stable/optim.html 
'''

model = UNet
model = model((WIDTH, HEIGHT), in_ch=3, out_ch=3, activation=None).to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)


# In[ ]:


'''
abs: training model
---
- 下面的程序會在3種條件下儲存當前的model(只包含weight)
    - 當前模型的loss是目前以來最低
    - 當前epoch數是20的倍數
    - 完成一個epoch的訓練
'''
from torchvision.ops import sigmoid_focal_loss

EPOCHS = 399
min_target_loss_value = 100
save_root = './data/save_weights/'
os.makedirs(save_root, exist_ok=True)

for epoch in range(EPOCHS):  
    class_loss_value = 0.0

    for i, (source_data, source_label) in enumerate(dataloader_train):
        # zero the parameter gradients
        '''
        abs: zero the parameter gradients
        ---
        這兩種方法都能夠清除variable內的gradient:
        方法1
        param in model.parameters():
        param.grad = None
        方法2 借助optimizer尋找關聯的variable並清除gradient
        optimizer.zero_grad()
        '''
        optimizer.zero_grad()
    
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        
        outputs = model(source_data)
        '''
        abs: single channel'
        ---
        Note:
        - If DiceLloss, outputs = F.sigmoid(outputs)
        - If BCEWithLogitsLoss, outputs不須activation
        '''
#         
#         loss = DiceLoss()(outputs, source_label)
#         loss = torch.nn.MSELoss()(outputs, source_label)
#         loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([100]).to(device))(outputs, source_label)
#         loss = sigmoid_focal_loss(outputs, source_label, reduction='sum')
        
        '''
        title: multi channel
        ---
        Note:
        - If CrossEntropyLoss, source_label should be as channel as outputs. ex: (B, C, W, H)
        - if FocalLoss, source_label should be 1 channel, ex: (B, 1, W, H)
        '''
#         loss = nn.CrossEntropyLoss()(outputs, source_label.long())
        loss = FocalLoss(alpha=[0.1,0.1,1], gamma = 1, size_average=False)(outputs, source_label.long())
        class_loss_value += loss.item()

        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(i, end='\r')
        del source_data, source_label, outputs
        torch.cuda.empty_cache()

    class_loss_value /= (i+1)   
    testing_loss_value = class_loss_value

    print(f'epoch: {epoch}, class_loss_value:{class_loss_value}')
    if testing_loss_value < min_target_loss_value:
        min_target_loss_value = testing_loss_value
        print('save best model')
        torch.save(model.state_dict(), f'{save_root}best_model.bin')
    else:
        if epoch%50==49:
            
            torch.save(model.state_dict(), f'{save_root}E{epoch}_model.bin')
        torch.save(model.state_dict(), f'{save_root}model.bin')
    


# In[ ]:


model = UNet
model = model(HEIGHT, in_ch=3, out_ch=3, activation=None).to(device)
save_root = './data/save_weights/'
filepath = f'{save_root}model.bin'
model.load_state_dict(torch.load(filepath)) 


# In[ ]:


'''
abs: testing model
---
'''

for i, data in enumerate(dataloader_train, 1):
    image, mask = data
    print(len(image), image.shape, mask.shape)
    with torch.no_grad():
        image = image.to(device)
        mask = mask.to(device)
        outputs = model(image)
#     outputs = outputs[:,1,:,:]
        print(outputs.shape)
        outputs = torch.argmax(outputs, dim=1)
        outputs = outputs.unsqueeze(1)
    print(outputs.shape)
#     loss = DiceLoss()(outputs, mask)
#     print(outputs.shape)
#     threshold = 0
#     outputs[outputs>=threshold] = 1.
#     outputs[outputs!=1] = 0.

    img_process = lambda image:image.permute(0,2,3,1).cpu().numpy()
    mask_process = lambda mask:mask.squeeze(1).cpu().numpy()
    
    for x, m, outputs in zip(img_process(image), mask_process(mask), mask_process(outputs)):
        show_image(x, m, outputs)
    del outputs, image, mask
#         break


# In[ ]:


import os
from IPython import get_ipython
if __name__ == '__main__':
    if get_ipython().__class__.__name__ =='ZMQInteractiveShell':
        os.system('jupyter nbconvert train_unet.ipynb --to python')


# In[ ]:




