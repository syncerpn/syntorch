"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class VarNet(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(VarNet, self).__init__()

        self.scale = scale

        self.tail = nn.ModuleList()

        self.tail.append(nn.Conv2d(1, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_tail:
        for i in range(len(self.tail)):
            self.tail[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.tail[i].weight)

    def forward(self, x, fea_out=False):
        z = x
        z = self.tail[0](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, z

        return y