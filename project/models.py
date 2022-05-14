import torch # 1.9
import torch.nn as nn
from torch.nn import functional as F

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