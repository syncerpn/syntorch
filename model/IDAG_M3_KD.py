"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
# import numpy as np

class IDAG_M3_KD(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M3_KD, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d( 1, 32, 3, 1, 1)) #0
        self.conv.append(nn.Conv2d(32, 16, 1, 1, 0)) #1
        
        self.conv.append(nn.Conv2d(16,  4, 1, 1, 0)) #2a
        self.conv.append(nn.Conv2d( 4,  4, 3, 1, 1)) #2b
        self.conv.append(nn.Conv2d( 4, 32, 1, 1, 0)) #2c
        
        self.conv.append(nn.Conv2d(32,  4, 1, 1, 0)) #3a
        self.conv.append(nn.Conv2d( 4,  4, 3, 1, 1)) #3b
        self.conv.append(nn.Conv2d( 4, 32, 1, 1, 0)) #3c
        
        self.conv.append(nn.Conv2d(32,  4, 1, 1, 0)) #4a
        self.conv.append(nn.Conv2d( 4,  4, 3, 1, 1)) #4b
        self.conv.append(nn.Conv2d( 4, 32, 1, 1, 0)) #4c
        
        self.conv.append(nn.Conv2d(32,  4, 1, 1, 0)) #5a
        self.conv.append(nn.Conv2d( 4,  4, 3, 1, 1)) #5b
        self.conv.append(nn.Conv2d( 4, 32, 1, 1, 0)) #5c
        
        self.conv.append(nn.Conv2d(32, 16, 1, 1, 0)) #6

        self.conv.append(nn.Conv2d(16, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_network:
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, kd_train=False):
        if kd_train:
            feas = []
            z = x
            z = F.relu(self.conv[0](z))
            z = F.relu(self.conv[1](z))

            for i in range(4):
                fea_i = z
                fea_i = F.relu(self.conv[2+i*3+0](fea_i))
                fea_i = F.relu(self.conv[2+i*3+1](fea_i))
                fea_i = F.relu(self.conv[2+i*3+2](fea_i))
                feas.append(fea_i)
                z = fea_i

            z = F.relu(self.conv[14](z))
            z = self.conv[15](z)

            y = residual_stack(z, x, self.scale)
            return y, feas
        else:
            z = x
            for i in range(15): # exclude the last layer
                z = F.relu(self.conv[i](z))

            z = self.conv[15](z)

            y = residual_stack(z, x, self.scale)
            return y