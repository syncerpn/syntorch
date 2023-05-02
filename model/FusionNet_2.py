"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class LargeModule(nn.Module):
    def __init__(self, scale):
        super(LargeModule, self).__init__()
        
        self.scale = scale

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2a
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2b
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2c
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2d

        self.tail = nn.ModuleList()
        self.tail.append(nn.Conv2d(16, 32, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(32, scale * scale, 3, 1, 1)) #7:last layer

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        z = F.relu(self.conv[0](z))
        z = F.relu(self.conv[1](z))
        z = F.relu(self.conv[2](z))
        z = F.relu(self.conv[3](z))

        z = F.relu(self.tail[0](z))
        z = self.tail[1](z)

        return z

class SmallModule(nn.Module):
    def __init__(self, scale):
        super(SmallModule, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1, groups=4)) #3a
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1, groups=4)) #3b
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1, groups=4)) #3c
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1, groups=4)) #3d

        self.tail = nn.ModuleList()
        self.tail.append(nn.Conv2d(16, 32, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(32, scale * scale, 3, 1, 1)) #7:last layer

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        z = F.relu(self.conv[0](z))
        z = F.relu(self.conv[1](z))
        z = F.relu(self.conv[2](z))
        z = F.relu(self.conv[3](z))

        z = F.relu(self.tail[0](z))
        z = self.tail[1](z)

        return z

class FusionNet_2(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(FusionNet_2, self).__init__()

        self.scale = scale

        self.head = nn.ModuleList()
        self.branch = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 32, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(32, 16, 1, 1, 0)) #1
        
        self.branch.append(LargeModule(scale=scale))
        self.branch.append(SmallModule(scale=scale))
        
        # init_head:
        for i in range(len(self.head)):
            self.head[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.head[i].weight)

    def forward(self, x, branch=0, fea_out=False):
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        branch_fea = self.branch[branch](z)

        y = residual_stack(branch_fea, x, self.scale)

        if fea_out:
            return y, branch_fea

        return y