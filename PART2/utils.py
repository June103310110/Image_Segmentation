#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations import DualTransform
import cv2
import os
import random
import torch


# In[2]:


def show_image_mask(*img_, split=False):
    plt.figure(figsize=(10,3))
    for i, img in enumerate(list(img_), 1):
#         print(np.shape(img))
        plt.subplot(1,len(img_),i)
    
            
        if type(img) == torch.Tensor:
            if len(img.shape)==4:
                if img.shape[1] == 3:
                    img =  img.flatten(0,1).permute(1,2,0).int().detach().numpy()
                else:
                    img =  img.flatten(0,2).int().detach().numpy()
            elif len(img.shape)==2:
                img = img.int().detach().numpy()
            
 
        
        img = img - img.min()
        if len(np.shape(img)) == 2 or np.shape(img)[-1] == 1:
            
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
    plt.show()
    plt.close()
    


# In[8]:


def find_objects_contours(mask):
    print(mask.shape)
    thresh = mask.astype(np.uint8)
    contours, hier =         cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    np.shape(contours)

    arr = np.array(contours)[-1].reshape(-1,2)
    arr = arr.mean(axis=0)
    return arr


# In[9]:


if __name__ == '__main__':
    pass


# In[10]:


def center_to_4point(mask, arr, side_width, pad=25):
    limit = len(mask)
    points = [0]*4
    
    if not pad:
        pad = 0
    value = side_width/2+pad    
    for i in arr:
        if side_width+2*pad > limit:
            print(side_width+2*pad)
            raise ValueError('not enough')
        if i > limit:
            raise ValueError('not include')
            
    for i in range(len(points)):
        if i in [0,1]:
            if arr[i%2] - value < 0:
                points[i] = 0
                points[i+2] += np.abs(arr[i%2] - value)
            else:
                points[i] = arr[i%2]-value
        if i in [2,3]:
            if arr[i%2]+value > limit:
                print(arr[i%2]+value)
                points[i] = len(mask)
                points[i-2] -= np.abs(limit - arr[i%2] - value)
            else:
                points[i] = arr[i%2]+value
    
    return np.round(points).astype(int)


# In[11]:


class mask_CutMix(DualTransform):
    def __init__(self,img_mask_list, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.img_lis, self.mask_lis = zip(*img_mask_list)
        self.choice = np.random.choice(range(len(self.img_lis)),size=1, replace=False)
        self.seed = 1000
        
    def apply(self, img, **params):
        a = self.choice[0]
#         a = choice[0]
#         b = choice[1]
#         print(a,b)
        source_center = self.find_objects_contours(self.mask_lis[a])
        points, _ = self.center_to_4point(self.mask_lis[a], source_center, 256)
        
        target_image = img
        if len(np.shape(img)) == 2:
            source_image = self.mask_lis[a]
        else:
            source_image = self.img_lis[a]
            self.seed = np.random.choice(range(10000),size=1)[0]
        
    
        x_min, y_min, x_max, y_max = points
        target_image = target_image.copy()
        piece = source_image[y_min:y_max, x_min:x_max]
        
        
        transform = A.Compose([
                A.Rotate((-30, 30), p=1), 
                A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
                A.HorizontalFlip(p=0.5),
            ])
        random.seed(self.seed)
        transformed  = transform(image=piece)

        
        target_image[y_min:y_max, x_min:x_max] = transformed['image']
        return target_image
        
    def find_objects_contours(self, mask):
        thresh = mask
        contours, hier =             cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        np.shape(contours)

        center = np.array(contours).reshape(-1,2)
        center = center.mean(axis=0)
        return center

    def center_to_4point(self, mask, arr, side_width, pad=None):
        limit = len(mask)
        points = [0]*4

        if not pad:
            pad = 0
        value = side_width/2+pad    
        for i in arr:
            if side_width+2*pad > limit:
                print(side_width+2*pad)
                raise ValueError('not enough')
            if i > limit:
                raise ValueError('not include')

        for i in range(len(points)):
            if i in [0,1]:
                if arr[i%2] - value < 0:
                    points[i] = 0
                    points[i+2] += np.abs(arr[i%2] - value)
                else:
                    points[i] = arr[i%2]-value
            if i in [2,3]:
                if arr[i%2]+value > limit:
                    print(arr[i%2]+value)
                    points[i] = len(mask)
                    points[i-2] -= np.abs(limit - arr[i%2] - value)
                else:
                    points[i] = arr[i%2]+value
        points = np.round(points).astype(int) 
        x_min, y_min, x_max, y_max = points
        return points, mask[y_min:y_max, x_min:x_max]


# In[13]:


if __name__ == '__main__':
    if get_ipython().__class__.__name__ =='ZMQInteractiveShell':
        os.system('jupyter nbconvert utils.ipynb --to python')


# In[ ]:





# In[ ]:




