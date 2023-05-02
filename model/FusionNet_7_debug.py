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
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2a
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2b
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2c
        self.conv.append(nn.Conv2d(16, 16, 3, 1, 1)) #2d

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        f1 = F.relu(self.conv[0](z))
        f2 = F.relu(self.conv[1](f1))
        f3 = F.relu(self.conv[2](f2))
        f4 = F.relu(self.conv[3](f3))
        return f4, [f1,f2,f3,f4]

class SmallModule(nn.Module):
    def __init__(self):
        super(SmallModule, self).__init__()

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(16, 4, 1, 1, 0)) #2a
        self.conv.append(nn.Conv2d(4, 16, 3, 1, 1)) #2a
        self.conv.append(nn.Conv2d(16, 4, 1, 1, 0)) #2b
        self.conv.append(nn.Conv2d(4, 16, 3, 1, 1)) #2b
        self.conv.append(nn.Conv2d(16, 4, 1, 1, 0)) #2c
        self.conv.append(nn.Conv2d(4, 16, 3, 1, 1)) #2c
        self.conv.append(nn.Conv2d(16, 4, 1, 1, 0)) #2d
        self.conv.append(nn.Conv2d(4, 16, 3, 1, 1)) #2d

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, kd_train=False):
        z = x
        z = F.relu(self.conv[0](z))
        f1 = F.relu(self.conv[1](z))
        z = F.relu(self.conv[2](f1))
        f2 = F.relu(self.conv[3](z))
        z = F.relu(self.conv[4](f2))
        f3 = F.relu(self.conv[5](z))
        z = F.relu(self.conv[6](f3))
        f4 = F.relu(self.conv[7](z))
        return f4, [f1,f2,f3,f4]

class FusionNet_7_debug(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(FusionNet_7_debug, self).__init__()

        self.scale = scale

        self.head = nn.ModuleList()
        self.branch = nn.ModuleList()
        self.tail = nn.ModuleList()

        self.head.append(nn.Conv2d( 1, 32, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(32, 16, 1, 1, 0)) #1
        
        self.branch.append(LargeModule())
        self.branch.append(SmallModule())
        
        self.tail.append(nn.Conv2d(16, 32, 1, 1, 0)) #6
        self.tail.append(nn.Conv2d(32, scale * scale, 3, 1, 1)) #7:last layer
        
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

        branch_fea, feas = self.branch[branch](z)

        z = F.relu(self.tail[0](branch_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, feas

        return y

    # def forward_merge_random(self, x, p=0.1, fea_out=False):
    #     z = x
    #     z = F.relu(self.head[0](z))
    #     z = F.relu(self.head[1](z))

    #     merge_map = None

    #     for ii in range(4):
    #         branch_fea_0 = self.branch[0](z, stages=[ii])
    #         branch_fea_1 = self.branch[1](z, stages=[ii])
            
    #         if merge_map is None:
    #             merge_map = torch.rand([1,1,branch_fea_0.shape[2],branch_fea_0.shape[3]])
    #             merge_map = (merge_map < p).type(torch.float32)
    #             merge_map = merge_map.repeat([branch_fea_0.shape[0],branch_fea_0.shape[1],1,1])

    #             merge_map = merge_map.to('cuda')

    #         merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)
    #         z = merge_fea

    #     z = F.relu(self.tail[0](merge_fea))
    #     z = self.tail[1](z)

    #     y = residual_stack(z, x, self.scale)

    #     if fea_out:
    #         return y, merge_fea

    #     return y

    # def forward_merge_gradient_sobel(self, x, p=0.1, fea_out=False):
    #     z = x
    #     z = F.relu(self.head[0](z))
    #     z = F.relu(self.head[1](z))

    #     #create sampling map using gradient sobel
    #     sobel_filter_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    #     sobel_filter_x.weight.data = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
    #     sobel_filter_x.to('cuda')

    #     sobel_filter_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    #     sobel_filter_y.weight.data = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
    #     sobel_filter_y.to('cuda')

    #     grad_map_x = sobel_filter_x.forward(x)
    #     grad_map_y = sobel_filter_y.forward(x)
    #     grad = abs(grad_map_x) + abs(grad_map_y)

    #     grad_sorted, _ = torch.sort(grad.view(-1), descending=True)
    #     grad_sorted_index = min(max(int(p * torch.numel(grad)), 0), torch.numel(grad)-1)
    #     grad_sorted_threshold = grad_sorted[grad_sorted_index]
        
    #     merge_map = (grad > grad_sorted_threshold).type(torch.float32)
    #     #done

    #     for ii in range(4):
    #         branch_fea_0 = self.branch[0](z, stages=[ii])
    #         branch_fea_1 = self.branch[1](z, stages=[ii])
    #         merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)
    #         z = merge_fea
        
    #     z = F.relu(self.tail[0](merge_fea))
    #     z = self.tail[1](z)

    #     y = residual_stack(z, x, self.scale)

    #     if fea_out:
    #         return y, merge_fea

    #     return y