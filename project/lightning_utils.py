import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.ops import sigmoid_focal_loss
import torch

class unetModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.encoder = model
#         self.discrimator = discrimator

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        sx, sy = batch[0]
        tx, ty = batch[1]
        outputs = self.encoder(torch.cat([sx,tx]))
        
        loss = sigmoid_focal_loss(outputs, torch.cat([sy,ty]), reduction='sum')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.encoder(x)


class discModel(pl.LightningModule):
    def __init__(self, model, discrimator, early_stop=None):
        super().__init__()
        self.encoder = model
        self.discrimator = discrimator
        if isinstance(early_stop, float):
            self.early_stop = 0.5
        else:
            self.early_stop = float('-inf')

    def training_step(self, batch, batch_idx):
        if self.early_stop == 'stop_training':
            self.trainer.should_stop = True
            return None
        else:
            # training_step defines the train loop.
            sx, sy = batch[0]
            tx, ty = batch[1]
            with torch.no_grad():
                outputs = self.encoder(torch.cat([sx,tx]))
            domain_label = torch.zeros([outputs.shape[0], 1], device=outputs.device)
            domain_label[:sx.shape[0]] = 1

            # 不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
            domain_logits = self.discrimator(outputs) # unet輸出層

            loss = torch.nn.BCEWithLogitsLoss()(domain_logits, domain_label)
            self.log("train_loss", loss)
            if loss < self.early_stop:
                self.early_stop = 'stop_training'
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.encoder(x)

class unetWithDiscModel(pl.LightningModule):
    def __init__(self, model, discrimator, lamb=0.1):
        super().__init__()
        self.encoder = model
        self.discrimator = discrimator
#         for params in self.discrimator.parameters():
#             params.require_grad = False
        self.lamb = lamb
        
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        sx, sy = batch[0]
        tx, ty = batch[1]
        outputs = self.encoder(torch.cat([sx,tx]))
        
        
        
        loss = sigmoid_focal_loss(outputs, torch.cat([sy,ty]), reduction='sum')
        
        if self.discrimator:
            with torch.no_grad():
                domain_label = torch.zeros([outputs.shape[0], 1], device=outputs.device)
                domain_label[:sx.shape[0]] = 1
                domain_logits = self.discrimator(outputs)
            domain_loss = torch.nn.BCEWithLogitsLoss()(domain_logits, domain_label)
            loss += self.lamb*domain_loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.encoder(x)
    

    
