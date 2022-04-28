import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
class DiceLoss(nn.Module):
    '如果輸入是被sigmoid過的，Diceloss的區間為(0,1]，smooth保證分子/分母不為0'
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss



class FocalLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py
    '''
    '輸入不需要activation'
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            C = inputs.shape[1] # num class
            inputs = inputs.transpose(1,-1)
            inputs = inputs.reshape(-1, C)
        
        target = target.to(torch.int64)
        # flatten all pixel
        target = target.transpose(1,-1)
        target = target.reshape(-1,1) # input should be a tensor (N, 1, H, W), 1 for 1 ch, class shound be [1,C], dtype=Long
        
        logpt = F.log_softmax(inputs) # log(softmax(x))
        logpt = logpt.gather(1, target) # explain by list, logpt = [logpt[i, target[i]] for i in range(len(target))]
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()) # reverse the log operation

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data) 
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()