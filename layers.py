# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()  
        
        self.double_conv = nn.Sequential(     
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),    
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        
        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        
        return self.conv(x)
