import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config_sh as config
import pdb

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.down_conv = ConvDown(10,64)
        self.conv1 = Conv(64,64)
        self.up = ConvUp(64,10)
        self.conv2 = Conv(10,10)
        self.out_conv = Conv(10,10)

    def forward(self, x):
        weight = self.down_conv(x)
        weight = self.conv1(weight)
        weight = self.up(weight)
        weight = self.conv2(weight)
        weight_x = weight * x + x_net
        out = self.out_conv(weight_x)
        return out

class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, kernels_per_layer=kernels_per_layer,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.down_conv(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.up_conv(x)


