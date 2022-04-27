#!/usr/bin/env python
# coding: utf-8

# In[27]:


BATCH_SIZE = 16
NUM_LABELS = 1
WIDTH = 128
HEIGHT = 128


# In[28]:


import os
import cv2
import numpy as np
import torch # 1.9
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A


# In[29]:


# os.system('python demo_set.py')


# In[30]:

def show_image(*img_):
    for i in img_:
        assert i.__class__.__name__ == 'ndarray', 'input data type should be ndarray'

    plt.figure(figsize=(10,3))
    for i, img in enumerate(list(img_), 1):
        plt.subplot(1,len(img_),i)

        if len(np.shape(img)) == 2 or np.shape(img)[-1] == 1:
            plt.imshow(img, cmap='gray')
        elif len(np.shape(img)) == 3:
            plt.imshow(img)
    plt.show()
    plt.close()



# In[31]:


def getAllDataPath(*dirs, test_split_size=None):
    all_file = []
    for img_dir in dirs:
        images = []
        for root, dirs, files in os.walk(os.path.abspath(img_dir)):
            for file in sorted(files):
                assert file.__contains__('.png'), 'not png'
                images.append(os.path.join(root, file))
        all_file.append(images)
    
    data_list = all_file[0] if len(all_file)==1 else list(zip(*all_file))

    if not test_split_size or test_split_size==0:
        return {'train':data_list}
    else:
        assert type(test_split_size)==float, 'set float to split test set size'
        train, test = train_test_split(data_list,
                         test_size = test_split_size)
        return {'train':train, 'test':test}
    
    
# In[47]:


#  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, imgs_anno_path_list, transform=None):
        self.imgs_anno_path_list = imgs_anno_path_list
        assert isinstance(self.imgs_anno_path_list, list), 'Need Input a list'
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs_anno_path_list)

    def __getitem__(self, idx):
        data = self.imgs_anno_path_list[idx]
        img_path = data[0]
        anno_path = data[1]
        
        # image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (1000, 1000, 3)
#         image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # mask
        mask = cv2.imread(anno_path,  cv2.IMREAD_GRAYSCALE) # (1000, 1000) 
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        image = torch.Tensor(image)
        image = image.permute(2,0,1)
        image = image.float()/255.
        
        mask = torch.Tensor(mask) 
        mask = mask.unsqueeze(0)
        mask = mask.float()/255.

        return image, mask
    


# In[55]:


# https://albumentations.ai/docs/getting_started/mask_augmentation/
if '__main__' == __name__:
    transform = A.Compose([
        A.CenterCrop(300, 900, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
        A.Rotate((-30, 30), interpolation=0),  

        A.ToFloat(always_apply=True),
        A.Resize(WIDTH, HEIGHT),
    ])

    target_transform = A.Compose([
        A.ToFloat(always_apply=True),
        A.Resize(WIDTH, HEIGHT),
    ])


# In[56]:


if '__main__' == __name__:
    path = os.getcwd()
    img_dir = f'{path}/data/capsule/test/scratch/'
    anno_dir = f'{path}/data/capsule/ground_truth/scratch/'

    defective_number = [i.split('.')[0] for i in os.listdir(img_dir)]
    print('defective_number: ',defective_number)
    data = getAllDataPath(img_dir, anno_dir, test_split_size=0.2)
    data.keys()


# In[57]:
    # 在這邊會強制對所有不滿BATCH_SIZE的訓練資料做數量上的匹配(單純把路徑複製貼上直到滿足16筆資料)，接著透過CustomImageDataset的transform對16筆資料做資料擴增
    if len(data['train']) < 16: 
        lis = data['train']
        lis = [lis[i%len(lis)] for i in range(BATCH_SIZE)]
        data['train'] = lis

    dataset_train = CustomImageDataset(data['train'], transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, drop_last=True,
                                                   shuffle=True, pin_memory=True,
                                                  )

    dataset_test = CustomImageDataset(data['test'], transform=target_transform) # **如果要正式使用要記得把這裡換成X_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, drop_last=False,
                                                  shuffle=False, pin_memory=True)


# In[ ]:




