"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack

class SVDSR(nn.Module):
    def __init__(self, n_layer, filter_size=64, scale=2):
        super(SVDSR, self).__init__()

        assert(n_layer >= 2)
        self.n_layer = n_layer
        self.scale = scale
        self.filter_size = filter_size
        
        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d(1, filter_size, 3, 1, 1))
        
        for i in range(n_layer-2):
            self.conv.append(nn.Conv2d(filter_size, filter_size, 3, 1, 1))

        self.conv.append(nn.Conv2d(filter_size, scale * scale, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)
            
    def forward(self, x):
        z = x
        for i in range(self.n_layer-1):
            z = F.relu(self.conv[i](z))

        z = self.conv[len(self.conv)-1](z)

        y = residual_stack(z, x, self.scale)
        return y