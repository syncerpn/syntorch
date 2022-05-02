import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model.common import default_conv, MeanShift

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4):
        super(SMB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.relu = nn.ReLU(True)

        # body
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers-1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)

        # collect
        self.collect = nn.Conv2d(out_channels*self.n_layers, out_channels, 1, 1, 0)

    def forward(self, x):
        out = []
        fea = x
        for i in range(self.n_layers):
            fea = self.body[i](fea)
            fea = self.relu(fea)
            out.append(fea)

        out = self.collect(torch.cat(out, 1))

        return out

class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SMM, self).__init__()
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        out = self.body(x)
        out = self.ca(out) + x

        return out

class SMSR(nn.Module):
    def __init__(self, n_feats=64, rgb_range=255.0, style='RGB', scale=2, conv=default_conv):
        super(SMSR, self).__init__()

        if style == 'RGB':
            n_colors = 3
        elif style == 'Y':
            n_colors = 1
        else:
            print('[ERRO] unknown style')
            assert(0)

        kernel_size = 3
        self.scale = scale

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size),
                        nn.ReLU(True),
                        conv(n_feats, n_feats, kernel_size)]

        # define body module
        modules_body = [SMM(n_feats, n_feats, kernel_size) for _ in range(5)]

        # define collect module
        self.collect = nn.Sequential(
            nn.Conv2d(64*5, 64, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats, n_colors*self.scale*self.scale, 3, 1, 1),
            nn.PixelShuffle(self.scale),
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x0 = self.sub_mean(x)
        x = self.head(x0)

        out_fea = []
        fea = x
        for i in range(5):
            fea = self.body[i](fea)
            out_fea.append(fea)
        out_fea = self.collect(torch.cat(out_fea, 1)) + x

        x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.add_mean(x)
        # print(x.shape)

        return x