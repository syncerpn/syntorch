"""
@author: nghiant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class unit_agent_uni_1x1(nn.Module):
    def __init__(self, inc, outc, nimm=0, immc=4):
        super(unit_agent_uni_1x1, self).__init__()
        self.conv = nn.ModuleList()
        
        self.inc = inc   # in channels
        self.outc = outc # out channels

        self.nimm = nimm # number of intermediate layers; at least 2
        self.immc = immc # intermediate layer channels

        self.conv.append(nn.Conv2d(self.inc, self.immc, 1, 1, 0))      # 1 x 1 x inc x immc
        for i in range(self.nimm):
            self.conv.append(nn.Conv2d(self.immc, self.immc, 3, 1, 1)) # 3 x 3 x immc x immc
        self.conv.append(nn.Conv2d(self.immc, self.outc, 1, 1, 0))     # 1 x 1 x immc x outc
        
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.kaiming_normal_(self.conv[i].weight)

    def forward(self, x):
        #forward method for getting the masks
        z = F.relu(self.conv[0](x))
        for i in range(self.nimm):
            z = F.relu(self.conv[1+i](z))
        d = self.conv[self.nimm+1](z)
        probs = torch.sigmoid(d)
        return probs

class unit_agent_uni_1x1_nobias(nn.Module):
    def __init__(self, inc, outc, nimm=0, immc=4):
        super(unit_agent_uni_1x1_nobias, self).__init__()
        self.conv = nn.ModuleList()
        
        self.inc = inc   # in channels
        self.outc = outc # out channels

        self.nimm = nimm # number of intermediate layers; at least 2
        self.immc = immc # intermediate layer channels

        self.conv.append(nn.Conv2d(self.inc, self.immc, 1, 1, 0, bias=False))      # 1 x 1 x inc x immc
        for i in range(self.nimm):
            self.conv.append(nn.Conv2d(self.immc, self.immc, 3, 1, 1, bias=False)) # 3 x 3 x immc x immc
        self.conv.append(nn.Conv2d(self.immc, self.outc, 1, 1, 0, bias=False))     # 1 x 1 x immc x outc
        
        for i in range(len(self.conv)):
            nn.init.kaiming_normal_(self.conv[i].weight)

    def forward(self, x):
        #forward method for getting the masks
        z = F.relu(self.conv[0](x))
        for i in range(self.nimm):
            z = F.relu(self.conv[1+i](z))
        d = self.conv[self.nimm+1](z)
        probs = torch.sigmoid(d)
        return probs

class unit_agent_mix_1x1_3x3(nn.Module):
    def __init__(self, inc, outc, nimm=0, immc=4):
        super(unit_agent_mix_1x1_3x3, self).__init__()
        self.conv = nn.ModuleList()
        
        self.inc = inc   # in channels
        self.outc = outc # out channels

        self.nimm = nimm # number of intermediate layers; at least 2
        self.immc = immc # intermediate layer channels

        self.conv.append(nn.Conv2d(self.inc, self.immc, 1, 1, 0, bias=False))      # 1 x 1 x inc x immc
        for i in range(self.nimm):
            self.conv.append(nn.Conv2d(self.immc, self.immc, 3, 1, 1, bias=False)) # 3 x 3 x immc x immc
        self.conv.append(nn.Conv2d(self.immc, self.outc, 3, 1, 1, bias=False))     # 3 x 3 x immc x outc
        
        for i in range(len(self.conv)):
            nn.init.kaiming_normal_(self.conv[i].weight)

    def forward(self, x):
        #forward method for getting the masks
        z = F.relu(self.conv[0](x))
        for i in range(self.nimm):
            z = F.relu(self.conv[1+i](z))
        d = self.conv[self.nimm+1](z)
        probs = torch.sigmoid(d)
        return probs

class unit_agent_mix_1x1_3x3_nobias(nn.Module):
    def __init__(self, inc, outc, nimm=0, immc=4):
        super(unit_agent_mix_1x1_3x3_nobias, self).__init__()
        self.conv = nn.ModuleList()
        
        self.inc = inc   # in channels
        self.outc = outc # out channels

        self.nimm = nimm # number of intermediate layers; at least 2
        self.immc = immc # intermediate layer channels

        self.conv.append(nn.Conv2d(self.inc, self.immc, 1, 1, 0))      # 1 x 1 x inc x immc
        for i in range(self.nimm):
            self.conv.append(nn.Conv2d(self.immc, self.immc, 3, 1, 1)) # 3 x 3 x immc x immc
        self.conv.append(nn.Conv2d(self.immc, self.outc, 3, 1, 1))     # 3 x 3 x immc x outc
        
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.kaiming_normal_(self.conv[i].weight)

    def forward(self, x):
        #forward method for getting the masks
        z = F.relu(self.conv[0](x))
        for i in range(self.nimm):
            z = F.relu(self.conv[1+i](z))
        d = self.conv[self.nimm+1](z)
        probs = torch.sigmoid(d)
        return probs

class unit_agent_uni_3x3(nn.Module):
    def __init__(self, inc, outc, nimm=0, immc=4):
        super(unit_agent_uni_3x3, self).__init__()
        self.conv = nn.ModuleList()
        
        self.inc = inc   # in channels
        self.outc = outc # out channels

        self.nimm = nimm # number of intermediate layers; at least 2
        self.immc = immc # intermediate layer channels

        self.conv.append(nn.Conv2d(self.inc, self.immc, 3, 1, 1))      # 3 x 3 x inc x immc
        for i in range(self.nimm):
            self.conv.append(nn.Conv2d(self.immc, self.immc, 3, 1, 1)) # 3 x 3 x immc x immc
        self.conv.append(nn.Conv2d(self.immc, self.outc, 3, 1, 1))     # 3 x 3 x immc x outc
        
        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.kaiming_normal_(self.conv[i].weight)

    def forward(self, x):
        #forward method for getting the masks
        z = F.relu(self.conv[0](x))
        for i in range(self.nimm):
            z = F.relu(self.conv[1+i](z))
        d =self.conv[self.nimm+1](z)
        probs = torch.sigmoid(d)
        return probs