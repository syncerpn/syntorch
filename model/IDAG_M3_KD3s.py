"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class IDAG_M3_KD3s(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M3_KD3s, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d( 1, 64, 3, 1, 1)) #0
        self.conv.append(nn.Conv2d(64, 32, 1, 1, 0)) #1
        
        self.conv.append(nn.Conv2d(32, 16, 1, 1, 0)) #2a
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2b
        self.conv.append(nn.Conv2d(16, 32, 1, 1, 0)) #2c
        
        self.conv.append(nn.Conv2d(32, 16, 1, 1, 0)) #3a
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #3b
        self.conv.append(nn.Conv2d(16, 32, 1, 1, 0)) #3c
        
        self.conv.append(nn.Conv2d(32, 64, 1, 1, 0)) #4

        self.conv.append(nn.Conv2d(64, scale * scale, 3, 1, 1)) #5:last layer
        
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

            for i in range(2):
                fea_i = z
                fea_i = F.relu(self.conv[2+i*3+0](fea_i))
                fea_i = F.relu(self.conv[2+i*3+1](fea_i))
                fea_i = F.relu(self.conv[2+i*3+2](fea_i))
                feas.append(fea_i)
                z = fea_i

            z = F.relu(self.conv[8](z))
            z = self.conv[9](z)

            y = residual_stack(z, x, self.scale)
            return y, feas
        else:
            z = x
            for i in range(9): # exclude the last layer
                z = F.relu(self.conv[i](z))

            z = self.conv[9](z)

            y = residual_stack(z, x, self.scale)
            return y

    def save_dn_module(self, file_prefix):
        I = list(range(len(self.conv)))
        T = [ 0,  1,  2,  3,  4,  5,  6,  7, 8, 9]

        for i, t in zip(I, T):
            file_name = file_prefix + str(t)
            with open(file_name, 'w') as f:
                bias = self.conv[i].bias.data.numpy()
                weight = self.conv[i].weight.data.numpy().flatten()
                data = np.concatenate((bias, weight))
                data.tofile(f)