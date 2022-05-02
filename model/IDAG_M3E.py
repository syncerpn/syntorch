"""
@author: nghiant
"""

import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack

class IDAG_M3E(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M3E, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d( 1, 64, 3, 1, 1)) #0
        self.conv.append(nn.Conv2d(64, 32, 1, 1, 0)) #1
        self.conv.append(nn.Conv2d(32, 32, 3, 1, 1)) #2
        self.conv.append(nn.Conv2d(32, 32, 3, 1, 1)) #3
        self.conv.append(nn.Conv2d(32, 32, 3, 1, 1)) #4
        self.conv.append(nn.Conv2d(32, 32, 3, 1, 1)) #5
        self.conv.append(nn.Conv2d(32, 64, 1, 1, 0)) #6

        self.conv.append(nn.Conv2d(64, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_network:
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)
    
    def forward(self, x):
        z = x
        for i in range(7): # exclude the last layer
            if i == 0:
                z = F.relu(self.conv[0](z))
            if i == 1:
                z = F.relu(self.conv[1](z)) # residual r
                r = z
            if i == 2:
                z = F.relu(self.conv[2](z))
            if i == 3:
                z = F.relu(self.conv[3](z))
            if i == 4:
                z = F.relu(self.conv[4](z))
            if i == 5:
                z = F.relu(self.conv[5](z) + r)
            if i == 6:
                z = F.relu(self.conv[6](z))

        z =        self.conv[7](z)

        y = residual_stack(z, x, self.scale)
        return y