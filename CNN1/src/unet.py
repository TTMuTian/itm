import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

class DownSamplingLayer(nn.Module):
    """
    input shape: B, in_ch, H, W
    output : (out_sc, out_n)
        out_sc shape: B, out_ch, H, W
        out_n shape: B, out_ch, H/2, W/2
    """
    def __init__(self, in_ch, out_ch):
        super(DownSamplingLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.MP_BN_ReLU = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, in_p):
        """
        in_p: input, from previous layer
        out_sc: output1, to skip connection
        out_n: output2, to next layer
        """
        out_sc = self.Conv_BN_ReLU_2(in_p)
        out_n = self.MP_BN_ReLU(out_sc)
        return out_sc, out_n


class UpSamplingLayer(nn.Module):
    """
    input shape: B, in_ch, H, W
    output shape: B, 2*out_ch, 2*H, 2*W
        output: out_sc concatenate out_n in dimension 1
        out_sc shape: B, out_ch, 2*H, 2*W
        out_n shape: B, out_ch, 2*H, 2*W
    """
    def __init__(self, in_ch, out_ch):
        super(UpSamplingLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=2*out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(2*out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*out_ch, out_channels=2*out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(2*out_ch),
            nn.ReLU()
        )
        self.UpConv_BN_ReLU = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, in_p, in_sc):
        """
        in_p: input1, from previous layer
        in_sc: input2, from skip connection
        out: output, which consists of UpSamplingLayer(input1) and input2
        """
        out_n = self.Conv_BN_ReLU_2(in_p)
        out_n = self.UpConv_BN_ReLU(out_n)
        out = torch.cat((out_n, in_sc), dim=1)
        return out


class UNet(nn.Module):
    """
    a U-Net
    """
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        channel = [2**(i+5) for i in range(6)] # 32, 64, 128, 256, 512, 1024

        # Encoder
        self.en1 = DownSamplingLayer(in_ch, channel[0])
        self.en2 = DownSamplingLayer(channel[0], channel[1])
        self.en3 = DownSamplingLayer(channel[1], channel[2])
        self.en4 = DownSamplingLayer(channel[2], channel[3])
        self.en5 = DownSamplingLayer(channel[3], channel[4])
        # Decoder
        self.de1 = UpSamplingLayer(channel[4], channel[4])
        self.de2 = UpSamplingLayer(channel[5], channel[3])
        self.de3 = UpSamplingLayer(channel[4], channel[2])
        self.de4 = UpSamplingLayer(channel[3], channel[1])
        self.de5 = UpSamplingLayer(channel[2], channel[0])
        # Conv_3
        self.ccc = nn.Sequential(
            nn.Conv2d(in_channels=channel[1], out_channels=channel[0], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channel[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, stride=1, padding=1),
            # # nn.BatchNorm2d(channel[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[0], out_channels=1, kernel_size=1, stride=1, padding=0),
            # # nn.BatchNorm2d(channel[0]),
            # nn.ReLU(),
        )

    def forward(self, input):
        out_sc1, out_n1 = self.en1(input)
        out_sc2, out_n2 = self.en2(out_n1)
        out_sc3, out_n3 = self.en3(out_n2)
        out_sc4, out_n4 = self.en4(out_n3)
        out_sc5, out_n5 = self.en5(out_n4)
        out_n6 = self.de1(out_n5, out_sc5)
        out_n7 = self.de2(out_n6, out_sc4)
        out_n8 = self.de3(out_n7, out_sc3)
        out_n9 = self.de4(out_n8, out_sc2)
        out_n10 = self.de5(out_n9, out_sc1)
        out = self.ccc(out_n10)
        return out


class DiceLoss(nn.Module):
    """
    calculate dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, label, smooth=1e-5):
        # output = torch.sigmoid(output)
        output = output.view(-1)
        label = label.view(-1)
        intersection = (output * label).sum()
        dice = (2 * intersection + smooth) / (output.sum() + label.sum() + smooth)
        return 1 - dice

if __name__ == '__main__':
    inputs = torch.randn(2, 1, 512, 512)
    net = UNet(in_ch=1, out_ch=1)
    outputs = net(inputs)
    print(outputs.shape)