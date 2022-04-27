#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch # 1.9
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import os


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu' # for debug建議使用cpu作為torch的運行背景
device


# ## Chapter1 : UNet網路構建

# ### ConvBlock
# - 加入Instance Norm.
# - <img src="https://miro.medium.com/max/983/1*p84Hsn4-e60_nZPllkxGZQ.png" 512="50%">
# 
# > 上圖為一整個batch的feature-map。輸入6張圖片，輸入6chs, 輸出也是6chs(C方向看進去是channel, N方向看進去是圖片)

# In[4]:


# # 原始版本
# class convBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.conv1(x))))


# In[5]:


## 加入instance normalization
class convBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding = 0):
        super().__init__()
        kernel_size = kernel_size
        pad_same_value = lambda kernel_size:(kernel_size-1)//2
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.INorm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        
    def forward(self, x):
        x = self.INorm(self.conv1(x))
        x = self.relu(x)
        x = self.INorm(self.conv2(x))
        x = self.relu(x)
        return x


# In[6]:


if __name__ == '__main__':
    block = convBlock(1, 64, 3, padding=1)
    WIDTH, HEIGHT = (512, 512)
    x = torch.randn(1, 1, WIDTH, HEIGHT)
    print(block(x).shape)


# ## Encoder(DownStream)
# 將影像進行編碼，過程中解析度會縮小(maxpooling、convolution)

# In[7]:


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, padding=True):
        super().__init__()
        pad = 1 if (padding==True or padding=='same') else 0

        self.conv = convBlock(in_ch, out_ch, 3, pad)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


# In[8]:


if __name__ == '__main__':
    down = Down(3, 64)
    WIDTH, HEIGHT = (512, 512)
    
    x = torch.randn(1, 3, WIDTH, HEIGHT)
    x = down(x)
    print(x.shape)


# ## Decoder(UpStream)
# 將編碼還原成影像，過程中解析度會放大直到回復成輸入影像解析度(transposed Convolution)。
# - 將編碼還原成影像是因為影像分割是pixel-wise的精度進行預測，解析度被還原後，就可以知道指定pixel位置所對應的類別
# - 類別資訊通常用feature-map的channels(chs)去劃分，一個channel代表一個class
# - 有許多UNet模型架構會有輸入576x576，但輸出只有388x388的情況，是因為他們沒有對卷積過程做padding，導致解析度自然下降。最後只要把mask resize到388x388就能繼續計算loss。

# ### Transposed Conv and UpsampleConv
# <img src="https://i.imgur.com/eIIJxre.png" alt="drawing" 512="300"/>
# <img src="https://i.imgur.com/uLo7icF.png" alt="drawing" 512="300"/>
# 
# Transposed Conv 
# - 透過上面的操作做轉置卷積，feature-map上的數值會作為常數與kernel相乘
# 
# UpsampleConv
# - 先做上採樣(Upsample/ Unpooling)
# - 然後作卷積(padding = same)
# <!-- #### 替代方案 UpSampling(Unpooling)+Convolution -->
# 

# In[9]:


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, padding=True):
        super().__init__()
        if bilinear:
            # normal convolutions to reduce the number of channels
            self.up = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                                    nn.Conv2d(in_ch, (in_ch // 2), 3, padding=1, bias=True))
        else:
            self.up = nn.ConvTranspose2d(in_ch, (in_ch // 2), kernel_size = 2, stride = 2)

        pad = 1 if (padding==True or padding=='same') else 0
        self.conv = convBlock(in_ch, out_ch, 3, padding=pad)
    
    def forward(self, x, enc_ftrs):
        x = self.up(x)
        enc_ftrs = self.crop(enc_ftrs, x)
        x = torch.cat([x, enc_ftrs], dim=1)
        x = self.conv(x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
            


# In[10]:


if __name__ == '__main__':
    enc_ftrs = torch.randn(1, 256, 32, 32)
    x = torch.randn(1, 512, 28, 28)
    x = Up(512, 256, bilinear=True, padding=True)(x, enc_ftrs)
    print(x.shape)


# ## Unet構建
# 結合encoder和decoder組成Unet。
# - 在輸出層如果用softmax做多元分類問題預測的話，類別數量要+1(num_classes+background)

# In[11]:


class UNet(nn.Module):
    def __init__(self, out_sz, in_ch, out_ch, bilinear=True, activation=None):
        super().__init__()     
        if isinstance(out_sz,(int)): self.out_sz = (out_sz, out_sz)
        if isinstance(out_sz,(tuple,list)): self.out_sz = tuple(out_sz)
        
        chs = (64, 128, 256, 512, 1024)
        
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.activation = activation
        self.out_sz = out_sz
        
        '''
        Unet with nn.ModuleList
        '''
        self.input = convBlock(in_ch, chs[0], 3, padding=1)
        self.down_list = nn.ModuleList([Down(chs[i], chs[i+1], padding='same')for i in range(len(chs)-1)]) 
        chs = chs[::-1]
        self.up_list = nn.ModuleList([Up(chs[i], chs[i+1], padding='same')for i in range(len(chs)-1)]) 
        
        '''
        Unet with simple code
        ---
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        '''
        

        
    def forward(self, x):
        down_layer_0 = self.input(x)
        
        'Unet with nn.ModuleList'
        enc_ftrs = [down_layer_0]
        for idx in range(len(self.down_list)):
            outputs = self.down_list[idx](enc_ftrs[idx])
            enc_ftrs.append(outputs)
        enc_ftrs = enc_ftrs[::-1]
        
        tmp_ftr = enc_ftrs[0]
        for idx in range(len(self.up_list)):
            tmp_ftr = self.up_list[idx](tmp_ftr, enc_ftrs[idx+1])
        
        '''
        Unet with simple code
        ---
        down_layer_1 = self.down1(down_layer_0)
        down_layer_2 = self.down2(down_layer_1)
        down_layer_3 = self.down3(down_layer_2)
        down_layer_4 = self.down4(down_layer_3)
        up_layer_1 = self.up1(down_layer_4, down_layer_3)
        up_layer_2 = self.up2(up_layer_1, down_layer_2)
        up_layer_3 = self.up3(up_layer_2, down_layer_1)
        tmp_ftr = self.up4(up_layer_3, down_layer_0)
        '''


        logits = self.head(tmp_ftr)
        
        # interpolate 
        _, _, H, W = logits.shape
        if (H,W)==self.out_sz: pass
        else:
            logits = F.interpolate(logits, self.out_sz)
        
        # add activation (not necessary)
        if self.activation:
            logits = self.activation(logits)
        
        return logits


# In[12]:


if __name__ == '__main__':
    HEIGHT, WIDTH,  = (512, 512)
    unet = UNet(HEIGHT, 3, 1, activation=None)
    
    x    = torch.randn(1, 3, WIDTH, HEIGHT)#.to(device)
    y_pred = unet(x)
    print(y_pred.shape)


# In[2]:


import os
if __name__ == '__main__':
    if get_ipython().__class__.__name__ =='ZMQInteractiveShell':
        os.system('jupyter nbconvert unet.ipynb --to python')


# In[ ]:




