"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()

        

class NewSResBlock(nn.Module):
    def __init__(self):
        super(NewSResBlock, self).__init__()

        self.head = nn.Conv2d(32, 32, 3, 1, 1)

        self.tail = nn.Conv2d(32, 32, 3, 1, 1)

        self.head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.head.weight)

        self.tail.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.tail.weight)



    def forward(self, x):
        z = x
        for i in range(len(self.conv)):
            z = F.relu(self.conv[i](z))
        return z

class _NewNet_ns(nn.Module): #hardcode
    def __init__(self, ns, scale=2):
        super(_NewNet_ns, self).__init__()

        self.scale = scale
        self.ns = ns

        self.head = nn.ModuleList()
        self.body = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 64, 3, 1, 1))
        self.head.append(nn.Conv2d(64, 32, 1, 1, 0))
        
        for _ in range(self.ns):
            self.body.append(NewSResBlock())
        
        self.tail.append(nn.Conv2d(32, 64, 1, 1, 0))
        self.tail.append(nn.Conv2d(64, scale * scale, 3, 1, 1))
        
        # init_head:
        for i in range(len(self.head)):
            self.head[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.head[i].weight)

        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x, branch=0, fea_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        branch_fea, feas = self.branch[branch](z)

        z = F.relu(self.tail[0](branch_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, feas

        return y