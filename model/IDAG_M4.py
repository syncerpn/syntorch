"""
@author: nghiant
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import math
import torch

class IDAG_M4(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M4, self).__init__()

        self.scale = scale

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv2d( 1, 48, 3, 1, 1)) #0
        self.conv.append(nn.Conv2d(48, 24, 1, 1, 0)) #1
        self.conv.append(nn.Conv2d(24, 16, 3, 1, 1)) #2
        self.conv.append(nn.Conv2d(16, 24, 3, 1, 1)) #3
        self.conv.append(nn.Conv2d(24, 16, 3, 1, 1)) #4
        self.conv.append(nn.Conv2d(16, 24, 3, 1, 1)) #5
        self.conv.append(nn.Conv2d(24, 48, 1, 1, 0)) #6

        self.conv.append(nn.Conv2d(48, scale * scale, 3, 1, 1)) #7:last layer
        
        # init_network:
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

        self.q_weights = []
        self.prepare_q_weights(nbits=8)

    def prepare_q_weights(self, nbits=16):
        self.o_weights = [self.conv[i].weight.data for i in range(len(self.conv) - 1)]
        self.q_weights = []
        for i in range(len(self.conv) - 1):
            w_max = torch.max(abs(self.conv[i].weight.data))
            step = 2 ** (math.ceil(math.log(2 * w_max / (2 ** nbits - 1), 2)))
            print(step)
            self.q_weights.append((torch.round(self.conv[i].weight.data / step + 0.5) - 0.5) * step)

    def quantize(self):
        for i in range(len(self.conv) - 1):
            self.conv[i].weight.data = self.q_weights[i]
            self.conv[i].cuda()

    def revert(self):
        for i in range(len(self.conv) - 1):
            self.conv[i].weight.data = self.o_weights[i]
            self.conv[i].cuda()

    def forward(self, x):
        z = x
        for i in range(7): # exclude the last layer
            z = F.relu(self.conv[i](z))

        z = self.conv[7](z)

        y = residual_stack(z, x, self.scale)
        return y

    def save_dn_module(self, file_prefix):
        I = list(range(len(self.conv)))
        T = [ 0,  1,  2,  3,  4,  5,  6,  7]

        for i, t in zip(I, T):
            file_name = file_prefix + str(t)
            with open(file_name, 'w') as f:
                bias = self.conv[i].bias.data.numpy()
                weight = self.conv[i].weight.data.numpy().flatten()
                data = np.concatenate((bias, weight))
                data.tofile(f)