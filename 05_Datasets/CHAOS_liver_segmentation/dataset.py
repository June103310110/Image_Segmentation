#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BATCH_SIZE = 8 # 8 for 256x256/ 16 for 128x128
# NUM_LABELS = 1
# WIDTH = 256
# HEIGHT = 256 
# MULTI_CHANNELS = False
# device = 'cuda:0'


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import cv2
import numpy as np
import torch # 1.9
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os

from sklearn.model_selection import train_test_split
# import torch.optim as optim
# import time
# import ipywidgets as widgets
# import pickle

# # 從repo裡面導入套件
# from utils import show_image_mask, mask_CutMix#, patience
# from unet import UNet


# 導入dicom套件
from pydicom import dcmread
from pydicom.data import get_testdata_files


# In[4]:


def show_image(*img_):
    for i in img_:
        assert i.__class__.__name__ == 'ndarray', 'imput data type should be ndarray'

    plt.figure(figsize=(10,3))
    for i, img in enumerate(list(img_), 1):
        plt.subplot(1,len(img_),i)

        if len(np.shape(img)) == 2 or np.shape(img)[-1] == 1:
            plt.imshow(img, cmap='gray')
        elif len(np.shape(img)) == 3:
            plt.imshow(img)
    plt.show()
    plt.close()


# ### Build torch dataset

# In[5]:


# def getImg(path):
#     if path.__contains__('.dcm'):  
#       # pydcm read image
#         ds = dcmread(path)
#         file = ds.pixel_array
#         file = file.astype('uint8') # 調整格式以配合albumentation套件需求
#     elif path.__contains__('.png'):
#         file = cv2.imread(path)[...,0]
#         file = file.astype('float32') # 調整格式以配合albumentation套件需求
        
#         if 'MRI' in path:
#             file[file!=63] = 0
#             file[file!=0] = 1
#         elif 'CT' in path:
#             file /= 255
#         else:
#             raise ValueError('Non-support dtype')
#     else:
#         raise ValueError(f'img format: {path} unknown')
#     return file


# In[6]:


def getAllDataPath(dir_path, imgOnly=False, test_split_size=None):
    
    images = []
    labels = []
    
    for root, dirs, files in os.walk(os.path.abspath(dir_path)):
        for file in sorted(files):
            if '.dcm' in file:
                images.append(os.path.join(root, file))
            elif '.png' in file:
                labels.append(os.path.join(root, file))
    if imgOnly:
        data_list = images
    else:
        data_list = list(zip(images, labels))

    if test_split_size:
        assert type(test_split_size)==float, 'set float to split test set size'
        train, test = train_test_split(data_list,
                         test_size = test_split_size)
        return {'train':train, 'test':test}
    else:
        return {'train':data_list}
        


# In[7]:


root = '/home/jovyan/DA/DATA/ST_data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/CT/'
CT_data = getAllDataPath(root, test_split_size=0.2)
root = '/home/jovyan/DA/DATA/ST_data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/MRI/MRI_Label/'
MRI_data = getAllDataPath(root, test_split_size=0.2)
root = '/home/jovyan/DA/DATA/ST_data/CHAOS_AIAdatasets/2_Domain_Adaptation_dataset/MRI/MRI_nonLabel/'
MRI_imgOnly_data = getAllDataPath(root, imgOnly=True)

for data in ['CT_data', 'MRI_data', 'MRI_imgOnly_data']:
    i = eval(data)
    for k in i.keys():
        print(data,k, np.shape(i[k]))


# In[8]:


#  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CTMRI_ImageDataset(Dataset):
    def __init__(self, imgs_anno_path_list,
                 #dtype, 
#                  dir_path,
                 transform=None):
        self.imgs_anno_path_list = imgs_anno_path_list
        self.transform = transform

#   
    def __len__(self):
        return len(self.imgs_anno_path_list)
    
    def __getitem__(self, idx):
        # now = time.time()
        imgOnly = False
        img_anno_path = self.imgs_anno_path_list[idx]

        if type(img_anno_path)==tuple:
#             img_anno_path = [i for i in img_anno_path]
            image = self.getImg(img_anno_path[0])
            mask = self.getImg(img_anno_path[1])
        else:
            image = self.getImg(img_anno_path)
            imgOnly = True
    
        
        if imgOnly:
            if self.transform:        
                transformed = self.transform(image=image)
                image = transformed['image']
            image = np.expand_dims(image, axis=0)
#             image = np.concatenate((image, image, image), axis=0)
            image = torch.Tensor(image)
            return image
        else:
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
#                 print('2', image.max())
            image = np.expand_dims(image, axis=0)
#             image = np.concatenate((image, image, image), axis=0)
            image = torch.Tensor(image)

            mask = torch.Tensor(mask) 
            mask = mask.unsqueeze(0)
            return image, mask
    
    def getImg(self, path):
        if path.__contains__('.dcm'):  
          # pydcm read image
            ds = dcmread(path)
            file = ds.pixel_array
            # image process
            file = cv2.medianBlur(file, 5)
            file = cv2.normalize(file, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#             file = file.astype('float32')
        elif path.__contains__('.png'):
            file = cv2.imread(path)[...,0]
            file = file.astype('float32') # 調整格式以配合albumentation套件需求

            if 'MRI' in path:
#                 pass
                file[file!=63] = 0
                file[file!=0] = 1
            elif 'CT' in path:
                file /= 255
            else:
                raise ValueError('Non-support dtype')
        else:
            raise ValueError(f'img format: {path} unknown')
        return file
     


# #### 使用albumentations進行資料擴增

# In[9]:


# # https://albumentations.ai/docs/getting_started/mask_augmentation/

# BATCH_SIZE = 8
# WIDTH, HEIGHT = (256,256)

# transform = A.Compose([
# #     A.HorizontalFlip(p=0.5),
# #     A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
# #     A.Rotate((-30, 30), interpolation=0), 
# #     A.RandomContrast(limit=0.2, p=1), 
# #     A.MedianBlur(always_apply=True, blur_limit=(3, 5)),

# #     A.Normalize(p=1, mean=(0.485), std=(0.229)),
# #     A.ToFloat(always_apply=True),
#     A.Resize(WIDTH, HEIGHT),
# ])

# target_transform = A.Compose([
# #     A.Normalize(p=1, mean=(0.485), std=(0.229)),     
# #     A.MedianBlur(always_apply=True, blur_limit=(3, 5)),
# #     A.ToFloat(always_apply=True),
#     A.Resize(WIDTH, HEIGHT),
# ])


# ### 建立DataLoader

# In[10]:


if '__main__' == __name__:
# 建議同時間只有8個(256,256)的sample進行計算 (Total = BATCH_SIZE*MULTIPLE_BATCH)

    dataset_train = CTMRI_ImageDataset(MRI_data['train'], transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    dataset_test = CTMRI_ImageDataset(MRI_data['test'], transform=target_transform) # **如果要正式使用要記得把這裡換成X_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    CT_dataset_train = CTMRI_ImageDataset(CT_data['train'], transform=transform)
    CT_dataloader_train = torch.utils.data.DataLoader(CT_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    CT_dataset_test = CTMRI_ImageDataset(CT_data['test'], transform=target_transform)
    CT_dataloader_test = torch.utils.data.DataLoader(CT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)


#     a = iter(dataloader_train)
#     x, y = a.next()
    all_y = torch.Tensor([])
    all_x = 0
    for batch in CT_dataset_train:
        x, y = batch
        all_y = torch.cat([all_y, y])
        
#         all_x += len(x[x>1])
#         assert len(x[x>1])>0
#         print(x.max().item())
#         all_x.append(x.max().item())
#     print(all_y.unique())
#     print(all_x)
#     print(max(all_x))

# MRI_imgOnly_dataset_train = ImageOnly_Dataset(MRI_imgOnly_data['train'], transform=transform)
# MRI_image_dataloader = torch.utils.data.DataLoader(MRI_imgOnly_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# In[ ]:


import os
if '__main__' == __name__:
    try:
        if get_ipython().__class__.__name__=='ZMQInteractiveShell':
            os.system('jupyter nbconvert dataset.ipynb --to python')
    except NameError:
        pass


# In[ ]:




