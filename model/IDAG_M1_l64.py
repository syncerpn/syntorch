"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class IDAG_M1_l64(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M1_l64, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d( 1, 64, 3, 1, 1)) #0
        self.conv.append(nn.Conv2d(64, 16, 1, 1, 0)) #1
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #3
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #4
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #5
        self.conv.append(nn.Conv2d(16, 64, 1, 1, 0)) #6

        self.conv.append(nn.Conv2d(64, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_network:
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x):
        z = x
        for i in range(7): # exclude the last layer
            z = F.relu(self.conv[i](z))

        z = self.conv[7](z)

        y = residual_stack(z, x, self.scale)
        return y

    def save_dn_module(self, file_prefix):
        I = list(range(len(self.conv)))
        T = list(range(len(self.conv)))

        for i, t in zip(I, T):
            file_name = file_prefix + str(t)
            with open(file_name, 'w') as f:
                bias = self.conv[i].bias.data.numpy()
                weight = self.conv[i].weight.data.numpy().flatten()
                data = np.concatenate((bias, weight))
                data.tofile(f)