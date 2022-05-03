# Image_Segmentation 影像分割，入門到大師之路
## 章節介紹
- 01_Segmentation_basic
  - 影像分割的深度學習基礎
- 02_Augmentation_methods
  - 影像分割用的影像增強方法，特別是特定資料集的操作(TDB)
- 03_Models
  - 影像分割的模型們
- 04_Instance_segmentation
  - 實例分割範例
- 05_Datasets
  - 資料集與Demo
- 06_Domain_adapation
  - 影像分割結合領域遷移(include semi/self-supervised)

### 參考文件
- [Unet家族的演進](https://github.com/June103310110/Image_Segmentation/blob/main/PART2/README.md)

## Intro of segmentation 常見的影像分割架構設計
- Albumentation 對標註也做影像增強/客製化方法 [01_custom_augmentation.ipynb](https://colab.research.google.com/drive/1_2T0IFvjgj6kUb6UCIPDe0uLYu1seTSq?usp=sharing)
- Scaling 上採樣與下採樣 [02_Scaling.ipynb](https://colab.research.google.com/drive/1wU7gQeKBfhrYSPKKh8KyQwzpoJwb3Jix?usp=sharing)
- Dilated convolution介紹與應用(DeepLab) [03_Dilated_conv.ipynb](https://colab.research.google.com/drive/13WQ_UJQSu1ePM3w_p1z6Gw53Ac_ZulId?usp=sharing)

## Semantic Segmentation FCN/FPN以及語意分割模型
![image](https://user-images.githubusercontent.com/32012425/157360181-0dd63a80-05ca-4437-823d-5ced6b291620.png)
- create_unet
  - https://colab.research.google.com/drive/1jswxT5G4Dd3x6Wq0HmaAeGp7xJj7bwMz?usp=sharing
- training Unet
  - https://colab.research.google.com/drive/1CyhnCdrCxXvndX_QNSMzZDb4Pc9rys7x?usp=sharing
- 補充: ResUnet
  - [create_train_ResUnet.ipynb](https://colab.research.google.com/drive/1SUKf7uI9Ezl1fAKJEPiOtf1nEfFTk9k5?usp=sharing)

## Instance Segmentation SOLO實例分割技巧
![](https://i.imgur.com/vbmbcWS.png)
- 轉換資料集(基於labelme)
  - https://colab.research.google.com/drive/1XmIuhu80d80IVjipxVtkxAuvWQmBjAhe?usp=sharing
- 訓練與推論
  - https://colab.research.google.com/drive/1Dsfhnj2mNkTtqjXgqUOXrMOI2u395Uhp?usp=sharing
- 補充: maskRCNN
  - https://colab.research.google.com/drive/1dBHrjzDjAk8HShQ49wu1jPd65WXXKS4a#scrollTo=EtP9NahKiTWX
  - (torch ver. 3rd. ithub) https://github.com/multimodallearning/pytorch-mask-rcnn
