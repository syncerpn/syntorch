"""
@author: nghiant
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

class IDAG_M3(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(IDAG_M3, self).__init__()

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

    def forward_log_mul(self, x):
        ##
        nbit = 7
        num_range = [(4.0,1.0),(4.0,4.0),(1.0,8.0),(1.0,8.0),(1.0,4.0),(1.0,2.0),(2.0,1.0)]
        ##
        out_shape = [1, -1, x.shape[2], x.shape[3]]

        w_unfolder_3x3 = nn.Unfold(3, stride=1, padding=0)
        w_unfolder_1x1 = nn.Unfold(1, stride=1, padding=0)
        z_unfolder_3x3 = nn.Unfold(3, stride=1, padding=1)
        z_unfolder_1x1 = nn.Unfold(1, stride=1, padding=0)

        z = x
        for i in range(7):
            w_mat = None
            z_mat = None
            if self.conv[i].kernel_size[0] == 3:
                w_mat = w_unfolder_3x3(self.conv[i].weight)
                w_mat = w_mat.view(w_mat.size(0), -1)
                z_mat = z_unfolder_3x3(z)
                z_mat = z_mat[0, :]

            elif self.conv[i].kernel_size[0] == 1:
                w_mat = w_unfolder_1x1(self.conv[i].weight)
                w_mat = w_mat.view(w_mat.size(0), -1)
                z_mat = z_unfolder_1x1(z)
                z_mat = z_mat[0, :]

            #normal case: mul
            # z = torch.mm(w_mat, z_mat)

            #log-mul: log2 then add
            # w_max, z_max = num_range[i]
            w_max = torch.max(torch.abs(w_mat))
            z_max = torch.max(torch.abs(z_mat))

            log_pos_end = 2 ** (nbit - 1) - 1
            log_neg_end = -2 ** (nbit - 1)
            nth_root_factor = ((w_max * z_max) ** (1/log_pos_end)).float()
            print(w_max, z_max, nth_root_factor)

            w_mat_sign = (w_mat > 0).float() - (w_mat < 0).float()
            z_mat_sign = (z_mat > 0).float() - (z_mat < 0).float()
            w_mat = torch.round(torch.log2(torch.abs(w_mat)) / torch.log2(nth_root_factor))
            z_mat = torch.round(torch.log2(torch.abs(z_mat)) / torch.log2(nth_root_factor))

            z = torch.zeros((w_mat.size(0), z_mat.size(1))).cuda()

            for ni in range(w_mat.size(0)):
                w_mat_col = torch.unsqueeze(w_mat[ni, :], 1).expand(z_mat.size(0), z_mat.size(1))
                w_mat_col_sign = torch.unsqueeze(w_mat_sign[ni, :], 1).expand(z_mat.size(0), z_mat.size(1))

                z[ni, :] = torch.sum(w_mat_col_sign * z_mat_sign * nth_root_factor ** (w_mat_col + z_mat), dim=0)

            z = torch.reshape(z, out_shape)
            for c in range(z.shape[1]):
                z[:,c,:,:] += self.conv[i].bias[c]

            z = F.relu(z)

        z = self.conv[7](z)

        y = residual_stack(z, x, self.scale)
        return y

    def forward(self, x, kd_train=False):
        if kd_train:
            feas = []
            z = x
            for i in range(7): # exclude the last layer
                z = F.relu(self.conv[i](z))
                # 4 block learning
                # if i >= 2 and i <= 5:
                #     feas.append(z)
                # 4 block learning

                # 2 block learning
                if i == 3 and i == 5:
                    feas.append(z)
                # 2 block learning

            z = self.conv[7](z)

            y = residual_stack(z, x, self.scale)
            return y, feas

        else:
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