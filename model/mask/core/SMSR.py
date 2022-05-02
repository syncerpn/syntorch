import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model.mask.agent.common import transform
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4, index=0, target_layer_index=[0], mask_type='sigmoid'):
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

        self.index = index

        self.masks = None # ready to use mask
        self.generate_mask = None # agent mask generation method
        self.target_layer_index = target_layer_index
        self.mask_type='sigmoid'

    def forward(self, x):
        agent_input_fmap = {}
        agent_output_fmap = {}

        sparsity_percent = 0.0
        sparsity_layers = {}

        out = []
        fea = x
        for i in range(self.n_layers):
            l = self.index * 4 + i
            if l in self.target_layer_index:
                if self.masks is not None:
                    mask = transform(self.masks[l], self.mask_type)
                elif self.generate_mask is not None:
                    masks = self.generate_mask({l:fea})
                    mask = transform(masks[l], self.mask_type)
                    
                agent_input_fmap[l] = fea


            fea = self.body[i](fea)
            fea = self.relu(fea)

            if l in self.target_layer_index:
                if (self.masks is not None) or (self.generate_mask is not None):
                    fea = fea * mask
                    sparsity_layers[l] = 1.0 - torch.sum(mask, (1,2,3)) / (mask.numel() / mask.shape[0])
                else:
                    sparsity_layers[l] = 1.0 - torch.sum((fea!=0).float(), (1,2,3)) / (fea.numel() / fea.shape[0])
                
                sparsity_percent += sparsity_layers[l] / len(self.target_layer_index)
                agent_output_fmap[l] = fea

            out.append(fea)

        out = self.collect(torch.cat(out, 1))

        return out, agent_input_fmap, agent_output_fmap, sparsity_percent, sparsity_layers

class SMM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, index=0, target_layer_index=[0], mask_type='sigmoid'):
        super(SMM, self).__init__()
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4, index=index, target_layer_index=target_layer_index, mask_type=mask_type)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        out, agent_input_fmap, agent_output_fmap, sparsity_percent, sparsity_layers = self.body(x)
        out = self.ca(out) + x

        return out, agent_input_fmap, agent_output_fmap, sparsity_percent, sparsity_layers

class SMSR(nn.Module):
    def __init__(self, n_feats=64, rgb_range=255.0, style='RGB', scale=2, conv=default_conv, target_layer_index=[0], mask_type='sigmoid'):
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
        modules_body = [SMM(n_feats, n_feats, kernel_size, index=i, target_layer_index=target_layer_index, mask_type=mask_type) for i in range(5)]

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

        self.masks = None # ready to use mask
        self.generate_mask = None # agent mask generation method
        self.target_layer_index = target_layer_index
        self.mask_type='sigmoid'

    def forward(self, x):
        agent_input_fmap = {}
        agent_output_fmap = {}

        sparsity_percent = 0.0
        sparsity_layers = {}

        x0 = self.sub_mean(x)
        x = self.head(x0)

        out_fea = []
        fea = x
        for i in range(5):
            fea, agent_input_fmap_i, agent_output_fmap_i, sparsity_percent_i, sparsity_layers_i = self.body[i](fea)

            for j in agent_input_fmap_i:
                agent_input_fmap[j]  = agent_input_fmap_i[j]
                agent_output_fmap[j] = agent_output_fmap_i[j]
                sparsity_layers[j]   = sparsity_layers_i[j]
            
            sparsity_percent += sparsity_percent_i

            out_fea.append(fea)
        out_fea = self.collect(torch.cat(out_fea, 1)) + x

        x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.add_mean(x)

        return x, agent_input_fmap, agent_output_fmap, sparsity_percent, sparsity_layers

    def attach_mask(self, masks, method='ready'):
        if method == 'ready':
            self.masks = masks
            for i in range(5):
                self.body[i].body.masks = masks # SMSR.SMM[i].SMB.masks; fucking confusing, lol
        elif method == 'generator':
            self.generate_mask = masks
            for i in range(5):
                self.body[i].body.generate_mask = masks # SMSR.SMM[i].SMB.generate_mask; fucking confusing, lol
        else:
            print('[ERRO] unknown masking method')
            assert(0)

    def detach_mask(self):
        self.masks = None
        for i in range(5):
            self.body[i].body.masks = None # SMSR.SMM[i].SMB.masks; fucking confusing, lol

        self.generate_mask = None
        for i in range(5):
            self.body[i].body.generate_mask = None # SMSR.SMM[i].SMB.generate_mask; fucking confusing, lol