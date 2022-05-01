# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from resnet import ResNet
from layers import *

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        
        self.resnet = ResNet()
        self.up_conv_1 = DoubleConv(128, 64)
        self.up_conv_2 = DoubleConv(256, 128)
        self.up_conv_3 = DoubleConv(512, 256)
        self.up_conv_4 = DoubleConv(1024, 512)
        self.up_conv_5 = DoubleConv(2048, 1024)
        self.out_conv = OutConv(64, 1)

        self.conv_5r = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.bn_5r = nn.BatchNorm2d(1024)  
        self.relu_5r = nn.ReLU(inplace=True)
        self.conv_4r = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.bn_4r = nn.BatchNorm2d(512)  
        self.relu_4r = nn.ReLU(inplace=True)
        self.conv_3r = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.bn_3r = nn.BatchNorm2d(256)  
        self.relu_3r = nn.ReLU(inplace=True)
        self.conv_2r = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.bn_2r = nn.BatchNorm2d(128)  
        self.relu_2r = nn.ReLU(inplace=True)
        
    def forward(self, input):
        # ResNet_F1:176x176x64;ResNet_F2:88x88x256;ResNet_F3:44x44x512;ResNet_F4:22x22x1024;ResNet_F5:11x11x2048;
        ResNet_F1, ResNet_F2, ResNet_F3, ResNet_F4, ResNet_F5 = self.resnet(input)
        # decoder_f5：11*11*1024
        decoder_f5 = self.up_conv_5(ResNet_F5)
        # decoder_f5_r:22*22*1024
        decoder_f5_r = F.interpolate(decoder_f5, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_f5_r = self.relu_5r(self.bn_5r(self.conv_5r(decoder_f5_r)))
        # decoder_f4:22*22*512
        decoder_f4 = self.up_conv_4(decoder_f5_r)
        # decoder_f4_r:44*44*512
        decoder_f4_r = F.interpolate(decoder_f4, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_f4_r = self.relu_4r(self.bn_4r(self.conv_4r(decoder_f4_r)))
        # decoder_f3:44*44*256
        decoder_f3 = self.up_conv_3(decoder_f4_r)
        # decoder_f3_r:88*88*256
        decoder_f3_r = F.interpolate(decoder_f3, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_f3_r = self.relu_3r(self.bn_3r(self.conv_3r(decoder_f3_r)))
        # decoder_f2:88*88*128
        decoder_f2 = self.up_conv_2(decoder_f3_r)
        # decoder_f2_r:176*176*128
        decoder_f2_r = F.interpolate(decoder_f2, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_f2_r = self.relu_2r(self.bn_2r(self.conv_2r(decoder_f2_r)))
        # decoder_f1:176*176*64
        decoder_f1 = self.up_conv_1(decoder_f2_r)
        # out_f :352*352*1
        decoder_f1_r = F.interpolate(decoder_f1, scale_factor=2, mode='bilinear', align_corners=True)
        out_f = self.out_conv(decoder_f1_r)
        #
        return out_f, decoder_f1, decoder_f2, decoder_f3, decoder_f4
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.backbone = backbone()
        self.conv_0 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn_0 = nn.BatchNorm2d(1)
        self.relu_0 = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(1)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(1)
        self.relu_3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        out_f, decoder_f1, decoder_f2, decoder_f3, decoder_f4 = self.backbone(input)
        decoder_f1 = self.relu_3(self.bn_3(self.conv_3(decoder_f1)))
        decoder_f1 = F.interpolate(decoder_f1, scale_factor=2, mode='bilinear', align_corners=True)
        decoder_f2 = self.relu_2(self.bn_2(self.conv_2(decoder_f2)))
        decoder_f2 = F.interpolate(decoder_f2, scale_factor=4, mode='bilinear', align_corners=True)
        decoder_f3 = self.relu_1(self.bn_1(self.conv_1(decoder_f3)))
        decoder_f3 = F.interpolate(decoder_f3, scale_factor=8, mode='bilinear', align_corners=True)
        decoder_f4 = self.relu_0(self.bn_0(self.conv_0(decoder_f4)))
        decoder_f4 = F.interpolate(decoder_f4, scale_factor=16, mode='bilinear', align_corners=True)
        out_all = out_f + decoder_f1 + decoder_f2 + decoder_f3 + decoder_f4
        
        return self.sigmoid(out_all), self.sigmoid(out_f), self.sigmoid(decoder_f1), self.sigmoid(decoder_f2), self.sigmoid(decoder_f3), self.sigmoid(decoder_f4)

def print_network(model, name):
    num_params = 0
    for p in model.parameters():  # 计算所有参数中元素的个数
        num_params += p.numel()
    total_num = sum(p.numel() for p in model.parameters())    
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))
    print("The number of total parameters: {}".format(total_num))
    print("The number of trainable parameters: {}".format(trainable_num))
        
if __name__ == '__main__':
    input = torch.randn((4, 3, 352, 352))
    # net = ResNet()
    net = Model()
    C1,C2,C3,C4,C5,C6 = net(input)
    print_network(net,'Model Structure')
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
    print(C6.size())
