"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class LargeModule(nn.Module):
    def __init__(self):
        super(LargeModule, self).__init__()
        
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(32, 32, 3, 1, 1)) #2a

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        z = F.relu(self.conv[0](z))
        return z

class SmallModule(nn.Module):
    def __init__(self):
        super(SmallModule, self).__init__()

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(32, 4, 1, 1, 0)) #2a
        self.conv.append(nn.Conv2d(4, 32, 3, 1, 1)) #2a

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        z = F.relu(self.conv[0](z))
        z = F.relu(self.conv[1](z))
        return z

class FusionNet_8(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(FusionNet_8, self).__init__()

        self.scale = scale

        self.head = nn.ModuleList()
        self.branch = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 64, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(64, 32, 1, 1, 0)) #1
        
        self.branch.append(LargeModule())
        self.branch.append(SmallModule())
        
        self.tail.append(nn.Conv2d(32, 64, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(64, scale * scale, 3, 1, 1)) #7:last layer
        
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

        branch_fea = self.branch[branch](z)

        z = F.relu(self.tail[0](branch_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, branch_fea

        return y

    def forward_merge_random(self, x, p=0.1, fea_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        branch_fea_0 = self.branch[0](z)
        branch_fea_1 = self.branch[1](z)
        
        merge_map = torch.rand([1,1,branch_fea_0.shape[2],branch_fea_0.shape[3]])
        merge_map = (merge_map < p).type(torch.float32)
        merge_map = merge_map.repeat([branch_fea_0.shape[0],branch_fea_0.shape[1],1,1])

        merge_map = merge_map.to('cuda')
        merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)

        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, merge_fea

        return y