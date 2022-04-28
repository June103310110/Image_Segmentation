#!/usr/bin/env python
# coding: utf-8

# In[9]:


BATCH_SIZE = 8 # 8 for 256x256/ 16 for 128x128
WIDTH = 256
HEIGHT = 256 
device = 'cuda:0'


# In[10]:


import numpy as np
import albumentations as A
import torch # 1.9
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from unet import UNet
from dataset import getAllDataPath, CTMRI_ImageDataset
from criterion import DiceLoss, FocalLoss


# In[11]:


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


# In[12]:


# https://albumentations.ai/docs/getting_started/mask_augmentation/

transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
#     A.Rotate((-30, 30), interpolation=0), 
#     A.RandomContrast(limit=0.2, p=1), 

#     A.Normalize(p=1, mean=(0.485), std=(0.229)),
#     A.ToFloat(always_apply=True),
    A.Resize(WIDTH, HEIGHT),
])

target_transform = A.Compose([
#     A.Normalize(p=1, mean=(0.485), std=(0.229)),                         
#     A.ToFloat(always_apply=True),
    A.Resize(WIDTH, HEIGHT),
])


# In[13]:


dataset_train = CTMRI_ImageDataset(MRI_data['train'], transform=transform)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dataset_test = CTMRI_ImageDataset(MRI_data['test'], transform=target_transform) # **如果要正式使用要記得把這裡換成X_test
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

CT_dataset_train = CTMRI_ImageDataset(CT_data['train'], transform=transform)
CT_dataloader_train = torch.utils.data.DataLoader(CT_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

CT_dataset_test = CTMRI_ImageDataset(CT_data['test'], transform=target_transform)
CT_dataloader_test = torch.utils.data.DataLoader(CT_dataset_test, batch_size=BATCH_SIZE, shuffle=False)


# In[14]:


print('BATCH_SIZE:', BATCH_SIZE)
for i in ['dataloader_train', 'dataloader_test', 'CT_dataloader_train', 'CT_dataloader_test']:
    loader = eval(i)
    print('---')
    print(i, loader.__len__())
    print([i.shape for i in iter(loader).next()])
    print([(i.max(), i.min()) for i in iter(loader).next()])
    


# In[15]:


def train_label_unet(backward=True):
    # part1 
    for model in [model_MRI]:
        for param in model.parameters():
            param.requires_grad = True
    mriOptim.zero_grad()
    'y pred'
    MRI_pred = model_MRI(target_data)

    '''
    class loss
    '''
#     MRI_class_loss = class_criterion(MRI_pred, target_label)
#     print(MRI_pred.shape, target_label.shape)
#     print(MRI_pred.min())
#     print(target_label.unique())
    MRI_class_loss = sigmoid_focal_loss(MRI_pred, target_label, alpha = 0.25, gamma = 2, reduction = 'mean')
    
#     if MRI_pred.data.size()[1]>1:
#         MRI_pred = F.softmax(MRI_pred, dim=1)
#         MRI_pred = torch.argmax(MRI_pred, dim=1)
    
#     MRI_dice_loss = dice_criterion(MRI_pred, target_label)
#         MRI_dice_loss += dice_criterion(MRI_pred[:,0,:,:], target_label)
#         MRI_dice_loss /= 2
    
    
    
    
#     lamb = WIDTH*HEIGHT/MRI_pred.data.size()[0]
    lamb = 0
    MRI_dice_loss = 0.0
#     MRI_loss = MRI_class_loss# + lamb*MRI_dice_loss
  
    if backward:
        mriOptim.zero_grad()
        MRI_class_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model_MRI.parameters(), max_norm = 10)
        mriOptim.step()
        
    del MRI_pred
    
    return MRI_class_loss, lamb*MRI_dice_loss, 0#, csis_loss/2, cycle_loss/2


# In[16]:


from torchvision.ops import sigmoid_focal_loss
save_root = 'saved/0420-1/'
dice_criterion = DiceLoss()
class_criterion = FocalLoss(gamma=2, alpha=[0.25, 0.75], size_average=False)
# class_criterion = sigmoid_focal_loss#(gamma=2, alpha=[0.25, 0.75])
# domain_criterion = nn.BCEWithLogitsLoss()
# consist_criterion = nn.L1Loss()
# device = 'cuda:0'


# - 測試看看torch vision focalloss bce (輸出1ch)
# - softmax head
# - sigmoid head

# In[17]:


source_dataloader, target_dataloader = CT_dataloader_train, dataloader_train
test_dataloader = dataloader_test # CT_dataloader_test, dataloader_test

model_MRI =  UNet(out_sz=(HEIGHT, WIDTH), num_class=1, enc_chs=(1,64,128,256,512,1024),
                  activation=None, # nn.Softmax(dim=1)
                 ).to(device) 
mriOptim = optim.Adam(model_MRI.parameters(), lr=1e-1)

EPOCHS = 300
min_target_loss_value = 100
source_domain_label = 1
target_domain_label = 0

for epoch in range(EPOCHS):  
    class_loss_value = 0.0
    dice_loss_value = 0.0
    cycle_loss_value = 0.0
    testing_loss_value = 0.0
    warmup = 5
    

    
    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(source_dataloader,
                                                                                       target_dataloader)):
#         source_data = source_data.to(device)
#         source_label = source_label.to(device)
        target_data = target_data.to(device, dtype = torch.float32)
        target_label = target_label.to(device, dtype = torch.float32)

        a,b,c = train_label_unet()
        class_loss_value += a
        dice_loss_value += b
        
        print(i, end='\r')
        del source_data, source_label, target_data, a, b, c
        torch.cuda.empty_cache()
        
    class_loss_value /= (i+1)   
    dice_loss_value /= (i+1)   

    testing_loss_value = class_loss_value

    print(f'epoch: {epoch}, class_loss_value:{class_loss_value:9.6f}, dice_loss_value: {dice_loss_value:9.6f}, cycle_loss_value:{cycle_loss_value}')
    if testing_loss_value < min_target_loss_value:
        min_target_loss_value = testing_loss_value
        print('save best model')
        torch.save(model_MRI.state_dict(), f'{save_root}best_model_MRI.bin')
    else:
        if epoch%20==0:
            torch.save(model_MRI.state_dict(), f'{save_root}E{epoch}_model_MRI.bin')
        torch.save(model_MRI.state_dict(), f'{save_root}model_MRI.bin')
        


# In[18]:


# # loader 
# test_dataloader = dataloader_test
# device = 'cpu'
# dice_loss_value = 0.0

# with torch.no_grad():
#     for i, (target_data, target_label) in enumerate(test_dataloader):
#         #         source_data = source_data.to(device)
#         #         source_label = source_label.to(device)
#         target_data = target_data.to(device)
#         target_label = target_label.to(device)
#         model_MRI = model_MRI.to(device)

#         outputs = model_MRI(target_data)
# #         outputs = F.sigmoid(outputs)

#         if True:
#             threshold = 0
#             outputs[outputs>threshold] = 1.
#             outputs[outputs!=1.] = 0.
#             outputs = outputs#.detach().cpu().numpy()
#     #     print(outputs.max())
#     #     outputs = F.log_softmax(outputs, dim=1)*-1
#     #     for o, t in zip(outputs, target_label):
#     #         z = o[1]
#     #         t = t.squeeze(0)
#     # #         print(z.shape, t.shape)
#     #         print('ch1 t=1', z[t==1])
#     #         print('ch1 t=0', z[t==0])
#     #         z = o[1]
#     #         t = t.squeeze(0)
#     #         print('ch1', z[t==1])
#     #     outputs = F.softmax(outputs, dim=1)
#     #     outputs = torch.argmax(outputs, dim=1)

#         loss = DiceLoss()(outputs, target_label)
#         dice_loss_value += loss

#         print(i, end='\r')
#         del target_data, target_label
#         torch.cuda.empty_cache()

#         break
#     dice_loss_value /= (i+1)   

# dice_loss_value


# In[19]:


import os
try:
    if get_ipython().__class__.__name__=='ZMQInteractiveShell':
        os.system('jupyter nbconvert train.ipynb --to python')
except NameError:
    pass


# In[ ]:




