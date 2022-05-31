# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import iouloss

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class HybridLoss(nn.Module):
    def __init__(self, device, weight_bce=1.0, weight_iou=1.0):
        super(HybridLoss, self).__init__()

        # bce损失权重
        self.weight_bce = weight_bce
        # iou损失权重
        self.weight_iou = weight_iou

    
    # bce损失    
    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(1 - input_ + eps)
        return torch.mean(wbce_loss)
    
    # iou损失
    iouloss = iouloss.IOU(size_average=True)

    def forward(self, y_pred, y_gt):
    
        bce_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        iou_loss = self.iouloss(y_pred, y_gt)

        # 计算总损失
        total_loss = self.weight_bce * bce_loss + self.weight_iou * iou_loss
        
        return total_loss

if __name__ == '__main__':
    

    if torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')
    
    dummy_input = torch.autograd.Variable(torch.sigmoid(torch.randn(2, 1, 8, 16)), requires_grad=True).to(device)
    dummy_gt = torch.autograd.Variable(torch.ones_like(dummy_input)).to(device)
    
    # Input Size : torch.Size([2, 1, 8, 16])
    print('Input Size :', dummy_input.size())

    criteria = HybridLoss(device=device)
    loss = criteria(dummy_input, dummy_gt)
    
    # Loss Value : tensor(1.4758, grad_fn=<AddBackward0>)
    print('Loss Value :', loss)

