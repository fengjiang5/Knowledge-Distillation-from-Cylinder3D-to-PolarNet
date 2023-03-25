from turtle import forward
import torch
import torch.nn as nn


class Weight_MSE(nn.Module):
    def __init__(self, reduction='mean'):
        super(Weight_MSE, self).__init__()
        reduction = reduction
        
    def forward(self, target, logit, weight=None):
        B,C,X,Y = target.size()
        # weight = weight.reshape(-1,)
        mse = torch.sum((target - logit) ** 2, dim=1)
        weight = torch.sum(weight, dim=-1)
        mse_loss = torch.sum(mse*weight)/(B*X*Y)
        return mse_loss ** 0.5