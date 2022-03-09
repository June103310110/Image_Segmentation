# Unet training (pytorch)
![image](https://user-images.githubusercontent.com/32012425/156869013-0b7a9373-e5ec-4ba9-8a27-eadb7b23a453.png)

### reference:
- dataset https://www.mvtec.com/company/research/datasets/mvtec-ad


## Colab example:
- create_unet 
  - https://colab.research.google.com/drive/1jswxT5G4Dd3x6Wq0HmaAeGp7xJj7bwMz?usp=sharing
- training Unet 
  - https://colab.research.google.com/drive/1CyhnCdrCxXvndX_QNSMzZDb4Pc9rys7x?usp=sharing
 


## Unet家族的演進
最開始的Unet借鑑於[FCN(Fully convolutional networks for semantic segmentation, 2014)](https://arxiv.org/pdf/1411.4038.pdf)的發展，FCN的網路中只有用到卷積層，並捨棄全連階層。在最後一個layer做上採樣將解析度縮放到原始影像的解析度，通道數則依據類別數而定，最後用pixel-wise的softmax來預測類別。

![](https://i.imgur.com/6OUbDS6.png)

FCN結構在輸出時會將前級的隱藏層(ex: pool4)，傳遞到後級(pool5)，並在pool5的位置進行上採樣。pool5要恢復到原始尺寸需要放大32倍，而pool4由於少經過pool，所以只需要放大16倍。將恢復到原始尺寸的兩組特徵向量相加後，才會進行類別預測的softmax計算。

如果只從pool5放大並輸出，稱為FCN32s，從pool4,5輸出稱為FCN16s，從pool3,4,5輸出則稱為FCN8s，FCN8s有最好的預測能力。

![](https://i.imgur.com/9dUOZCS.png)

#### [Unet, 2015](https://arxiv.org/pdf/1505.04597.pdf)對FCN做的改進在於
- FCN會把「最後3層」的特徵向量進行「相加」運算並輸出，Unet則會把每一個convolution block(由連續兩個卷積層和激勵函數組成)的特徵向量都向後傳遞。
- FCN的特徵向量在組合時是透過相加，Unet則是疊加(concat)。
- FCN進入上採樣階段時，通道數已經被限制在類別數(ex:21)不會增加維度，Unet則在整個採樣階段都保留大量的通道，這讓不同尺寸的特徵更能保留下來，並有更好的還原度。
- FCN會把完整的特徵向量向後傳遞並放大，但Unet的作者們則發現在卷積的過程中，邊緣的像素一定會產生縮減(除非使用padding，但padding也會影響邊緣的還原)。於是Unet的設計上向後傳遞時，會把特徵向量的邊緣去除掉，只保留影像正中間的部分。

> 所以Unet在設計時，你會發現最後一層的特徵向量的解析度與輸入不同，以原始論文為例，原本572x572的輸入會變成384x384的輸出，這不是你的上採樣模組壞掉，而是卷積過程的自然損失。通常我們在最後輸出之前直接做二次插值將影像resize到匹配尺寸。你可以取消這個動作並設定卷積層的padding來觀察邊緣的重建能力。

![](https://i.imgur.com/gd7JDiC.png)

#### [Vnet, 2016](https://arxiv.org/abs/1606.04797)對Unet的改進在於
- Dice Loss的創新導入，迄今Dice Loss也是Class Level的經典損失函數
- 另外雖然提不上改進，但Vnet在3D醫學影像上實現了語意分割。

![](https://i.imgur.com/30CKMXu.png)

#### [ResUnet, 2017](https://arxiv.org/abs/1711.10684)對Unet的改進在於
- 把convolution block從兩個卷積換成包含Batch Normalization和Skip Connections的Residual convolution Block。
- 下採樣的機制取消pooling，而是在Residual convolution Block的第一個卷積層採用stride=2的參數，實現1/2的下採樣。
- 取消Unet的Crop機制。(卷積的padding設定為same)

![](https://i.imgur.com/XAIkKTl.png)

在ResUnet之後還有出現[Inception-Unet(2020)](https://dl.acm.org/doi/abs/10.1145/3376922)，大體上就是把Residual convolution Block換成Inception-like的Block。




#### 最後是![](https://latex.codecogs.com/svg.latex?\\U^2net)[Link](https://arxiv.org/pdf/2005.09007.pdf)
大體上，針對Convolution Block做的這些演進與更改，主要的目的提取更多細節，但實際上每一層的Block能夠提取到的細節是受限於當前特徵向量的解析度的，也就是當前卷積核在當前的解析度下，能獲取的感受野是小的，也就是local feature，這對於尋找到顯著的、用以進行識別(recognize)的特徵是有幫助的，但對於輪廓、形狀甚至材質等特徵，我們會希望萃取non-local(global) features，這也是影像分割需要的。

於是有些研究就開始思考如何在單一層(單一個block)中同時萃取local feature和non-local(global) features，實際上Inception-Unet就是一個開端，而U<sup>2</sup>-Net則將它發揚光大。

![](https://i.imgur.com/RXQyIBm.png)

![](https://latex.codecogs.com/svg.latex?\\U^2net)的Block稱為`residual U-block, RSU`，也就是上圖最右邊的Block，相對於Inception-Unet，雖然兩者都是為了能在高解析度的影像就獲取Global information(with non-local features)，但Inception-block使用的多層空洞卷積會造成計算量以及記憶體的負擔，為了解決這一點，RSU-block希望使用傳統的pooling方法(max-pooling)來增加感受野而不是使用空洞卷積，並同時在block內對特徵向量進行連續的池化和卷積。

但如果在block內對特徵向量進行連續的池化和卷積，解析度還是會損失的。於是為了在block內萃取local/non-local feature的同時不損失特徵向量解析度，作者將block設計成一個小型的Unet，其中使用到的卷積的padding設定為same，下採樣階段使用maxpooling，上採樣則是二次線性差值。

然後用這個RSU block取代convolution block，就完成了。
> ref https://github.com/xuebinqin/U-2-Net

如下圖所示，可以看到基於RSU block，他的算力需求相較INC block(Inception Unet)有了大幅下降。

![](https://i.imgur.com/tJDsMJp.png)


