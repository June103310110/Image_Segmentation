#!/usr/bin/env python
# coding: utf-8

# # Unet
# source: https://amaarora.github.io/2020/09/13/unet.html
# 
# <img src="https://i.imgur.com/LQORH9i.png" alt="drawing" width="500"/>
# 

# In[1]:


BATCH_SIZE = 32
NUM_LABELS = 1
WIDTH = 512
HEIGHT = 512


# In[2]:


import cv2
import torch # 1.9
import torch.nn as nn
import numpy as np
import os
import torchvision
from torch.nn import functional as F


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu' # for debug建議使用cpu作為torch的運行背景
device


# ## Chapter1 : UNet網路構建

# ### ConvBlock
# - 加入Instance Norm.
# - <img src="https://miro.medium.com/max/983/1*p84Hsn4-e60_nZPllkxGZQ.png" width="50%">
# 
# > 上圖為一整個batch的feature-map。輸入6張圖片，輸入6chs, 輸出也是6chs(C方向看進去是channel, N方向看進去是圖片)

# In[5]:


# # 原始版本
# class convBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.conv1(x))))


# In[6]:


## 加入instance normalization
class convBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 'same', kernel_size=3):
        super().__init__()
        kernel_size = kernel_size
        pad_size = lambda kernel_size:(kernel_size-1)//2
        if padding=='same':
            self.padding = pad_size(kernel_size)
        else:
            self.padding = padding
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=self.padding, bias=False)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=self.padding, bias=False)
        self.INorm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        
    def forward(self, x):
        x = self.INorm(self.conv1(x))
        x = self.relu(x)
        x = self.INorm(self.conv2(x))
        x = self.relu(x)
        return x


# In[7]:


if __name__ == '__main__':
    block = convBlock(1, 64)
    x = torch.randn(1, 1, WIDTH, HEIGHT)
    block(x).shape


# ## Encoder(DownStream)
# 將影像進行編碼，過程中解析度會縮小(maxpooling、convolution)

# In[8]:


class Encoder(nn.Module):
    def __init__(self, chs=(3,32,64,128,256,512), padding='same'):
        super().__init__()
        self.FPN_enc_ftrs = nn.ModuleList([convBlock(chs[i], chs[i+1], padding) for i in range(len(chs)-1)])
#         self.pool = nn.MaxPool2d(2)
        self.pool = torch.max_pool2d
        
    def forward(self, x):
        features = []
        
        for block in self.FPN_enc_ftrs:
            x = block(x)
            features.append(x)
#             print(x.shape)
            x = self.pool(x, kernel_size=2)
        return features


# In[9]:


if __name__ == '__main__':
    encoder = Encoder()
    x = torch.randn(1, 3, WIDTH, HEIGHT)
    features = encoder(x)
    for f in features:
        print(f.shape)


# ## Decoder(UpStream)
# 將編碼還原成影像，過程中解析度會放大直到回復成輸入影像解析度(transposed Convolution)。
# - 將編碼還原成影像是因為影像分割是pixel-wise的精度進行預測，解析度被還原後，就可以知道指定pixel位置所對應的類別
# - 類別資訊通常用feature-map的channels(chs)去劃分，一個channel代表一個class
# - 有許多UNet模型架構會有輸入576x576，但輸出只有388x388的情況，是因為他們沒有對卷積過程做padding，導致解析度自然下降。最後只要把mask resize到388x388就能繼續計算loss。

# ### Transposed Conv and UpsampleConv
# <img src="https://i.imgur.com/eIIJxre.png" alt="drawing" width="300"/>
# <img src="https://i.imgur.com/uLo7icF.png" alt="drawing" width="300"/>
# 
# Transposed Conv 
# - 透過上面的操作做轉置卷積，feature-map上的數值會作為常數與kernel相乘
# 
# UpsampleConv
# - 先做上採樣(Upsample/ Unpooling)
# - 然後作卷積(padding = same)
# <!-- #### 替代方案 UpSampling(Unpooling)+Convolution -->
# 

# In[10]:


# ConvTranspose2d透過設定k=2, s=2, output_padding=0可以讓影像從28x28變成56x56
if __name__ == '__main__':
    x = torch.randn(1, 3, 28, 28)
    x = nn.ConvTranspose2d(3, 30, kernel_size=2, stride=2, output_padding=0)(x)
    x.shape


# In[11]:


class UpSampleConvs(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu  = nn.ReLU()
        self.upSample = nn.Upsample(scale_factor=2)
        self.INorm = torch.nn.InstanceNorm2d(out_ch)
        
    def forward(self, x):
        x = self.upSample(x)
#         x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.INorm(x)
#         return self.relu(self.conv2(self.relu(self.upSample(x))))
        return x


# In[12]:


if __name__ == '__main__':
    x = torch.randn(1, 3, 28, 28)
    x = UpSampleConvs(3,30)(x)
    x.shape


# ### decoder(上採樣) module

# In[13]:


class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32), padding='same'):
        super().__init__()

        self.chs = chs
        self.padding = padding
#         self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])  # 轉置卷積
        self.upconvs = nn.ModuleList([UpSampleConvs(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # 上採樣後卷積
        self.FPN_dec_ftrs = nn.ModuleList([convBlock(chs[i], chs[i+1], padding=padding) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            enc_ftrs = encoder_features[i]
            
            x = self.upconvs[i](x)
                
#             if self.padding == 0:
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.FPN_dec_ftrs[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


# In[14]:


if __name__ == '__main__':
    for i in features:
        print(i.shape)


# In[15]:


if __name__ == '__main__':
    decoder = Decoder()
    decoder
    x = torch.randn(1, 512, WIDTH//16, HEIGHT//16)
    decoder(x, features[::-1][1:]).shape 


# nn.Tanh## Unet構建
# 結合encoder和decoder組成Unet。
# - 在輸出層如果用softmax做多元分類問題預測的話，類別數量要+1(num_classes+background)

# In[16]:


class UNet(nn.Module):
    def __init__(self, out_sz, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, padding='same', activation=None):
        super().__init__()
        self.encoder     = Encoder(enc_chs, padding=padding)
        self.decoder     = Decoder(dec_chs, padding=padding)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz
        self.activation = activation

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:]) # 把不同尺度的所有featuremap都輸入decoder，我們在decoder需要做featuremap的拼接
        out      = self.head(out)
        if self.activation:
            out = self.activation(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


# In[17]:


if __name__ == '__main__':
    unet = UNet(num_class=1, padding = 'same', out_sz=(WIDTH,HEIGHT), retain_dim=False)
    unet#.to(device)
    x    = torch.randn(1, 3, WIDTH, HEIGHT)#.to(device)
    y_pred = unet(x)
    y_pred.shape


# In[18]:


if __name__ == '__main__':
    if get_ipython().__class__.__name__ =='ZMQInteractiveShell':
        os.system('jupyter nbconvert unet.ipynb --to python')


# In[ ]:




