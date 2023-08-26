import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import residual_stack
import numpy as np

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    # make random noise fullfilled in noise matrix
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log() # gumbel distribution
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        # global average pooling: feature -> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale -> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y

class LargeModule(nn.Module):
    def __init__(self, ns):
        super(LargeModule, self).__init__()

        self.ns = ns
        
        self.conv = nn.ModuleList()
        for i in range(ns):
            self.conv.append(nn.Conv2d(16, 16, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, stages=[]):
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERRO] invalid stage {s}"

        z = x
        if stages:
            for s in range(self.ns):
                if s in stages:
                    z = F.relu(self.conv[s](z))
            return z
        else:
            feas = []
            for s in range(self.ns):
                z = F.relu(self.conv[s](z))
                feas.append(z)
            return z, feas
        
class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, ns=4):
        super(SMB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ns = ns

        self.tau = 1
        self.relu = nn.ReLU(True)

        # channel mask
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, ns, 2))

        # body conv
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.ns - 1):
            self.conv.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))

        for i in range(self.ns):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)
        
        # collect information after concat
        self.collect = nn.Conv2d(out_channels * ns, out_channels, 1, 1, 0) #6

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self): # apply channel mask to sparse conv
        # channel mask
        ch_mask = self.ch_mask.softmax(3).round()
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for s in range(self.ns):    # 1 dense 0 sparse
            if s == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, :, s, 1].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, s, 0].sum(0)))

            else:
                self.d_in_num.append(int(ch_mask[0, :, s-1, 1].sum(0)))
                self.s_in_num.append(int(ch_mask[0, :, s-1, 0].sum(0)))
                self.d_out_num.append(int(ch_mask[0, :, s, 1].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, s, 0].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        # filter weights corresponding to channel mask
        #NOTE: At stage s:
        ##       - channels of each filter based on ch_mask[s-1]
        ##       - number of filters based on ch_mask[s]
        for s in range(self.ns):
            if s == 0:
                kernel_s.append([])
                if self.d_out_num[s] > 0:
                    kernel_d2d.append(self.conv[s].weight[ch_mask[0, :, s, 1]==1, ...]).view(self.d_out_num[s], -1)
                else:
                    kernel_d2d.append([])
                if self.s_out_num[s] > 0:
                    kernel_d2s.append(self.conv[s].weight[ch_mask[0, :, s, 0]==1, ...]).view(self.s_out_num[s], -1)

            else:
                if self.d_in_num[s] > 0 and self.d_out_num[s] > 0:
                    kernel_d2d.append(
                        self.conv[s].weight[ch_mask[0, :, s, 1]==1, ...][:, ch_mask[0, :, s-1, 1]==1, ...].view(self.d_out_num[s], -1))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[s] > 0 and self.s_out_num[s] > 0:
                    kernel_d2s.append(
                        self.conv[s].weight[ch_mask[0, :, s, 0]==1, ...][:, ch_mask[0, :, s-1, 1]==1, ...].view(self.s_out_num[s], -1))
                else: 
                    kernel_d2s.append([])
                if self.s_in_num[s] > 0:
                    kernel_s.append(
                        torch.cat(
                            self.conv[s].weight[ch_mask[0, :, s, 1]==1][:, ch_mask[0, :, s-1, 0]==1],  # s2d
                            self.conv[s].weight[ch_mask[0, :, s, 0]==1][:, ch_mask[0, :, s-1, 0]==1]), # s2s
                            0).view(self.d_out_num[s] + self.s_out_num[s], 0)
                else:
                    kernel_s.append([])

        # the last 1x1 conv: we need to map all the dense/sparse into dense channels to perform fully 1x1 conv at the last layer
        ch_mask = ch_mask[0, ...].transpose(1, 0).contiguous().view(-1, 2)
        self.d_in_num.append(int(ch_mask[:, 1].sum(0))) # total dense channels at every stage
        self.s_in_num.append(int(ch_mask[:, 0].sum(0))) # total sparse channels at every stage
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(self.collect.weight[:, ch_mask[..., 1]==1, ...].squeeze())
        kernel_d2s.append([])
        kernel_s.append(self.collect.weight[:, ch_mask[..., 0]==1, ...].squeeze())

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias


    def _generate_indices(self): # apply spatial mask to sparse conv
        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze()) # in this function, 1 is sparse

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0] #x
        self.w_idx_1x1 = mask_indices[:, 1] #y

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A #NOTE: Why + A

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1) # 1 1 1 2 2 2 3 3 3 
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1) # 1 2 3 1 2 3 1 2 3

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices # replace sparse position to idx from 1 to S

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)
        
    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        Perform sparse convolution at a specific stage, the result might be None in the case that there are lack of 
        sparse/dense channel at that stage

        Args:
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: stage index

        Returns: feature_dense and feature_sparse after the conv at stage
        '''

        ### dense input ###
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    # transform to patches for conv in linear order
                    fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0) 
                    # matmul corresponding weight
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col) 
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3)) # 1, dout, H', W'
                else:
                    fea_col = fea_dense.view(self.d_in_num[index], -1)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel_d2s[index], self._mask_select(fea_dense, k))
            
        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = torch.mm(self.kernel_s[index], F.pad(fea_sparse, [1,0,0,0])[:, self.idx_s2s].view(self.s_in_num[index] * k * k, -1))

        ### fusion and concat ###

        ## s2d and d2d 
        if self.d_out_num[index] > 0:           
            if self.d_in_num[index] > 0:        # if has d2d
                if self.s_in_num[index] > 0:    # if has d2d and s2d
                    fea_d2d[0, :, self.h_idx_1x1, self.w_idx_1x1] += fea_s2ds[:self.d_out_num[index], :]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:                               # not d2d but s2d -> sparse input with position from s2d and 0 otherwise
                fea_d = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
                fea_d[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds[:self.d_out_num[index], :]
        else:                                   # neither d2d nor s2d
            fea_d = None

        # d2s and s2s
        if self.s_out_num[index] > 0:           
            if self.d_in_num[index] > 0:        # if d2s
                if self.s_in_num[index] > 0:    # if d2s and s2s
                    fea_s = fea_d2s + fea_s2ds[-self.s_out_num[index]:, :]
                else:                           # not s2s but d2s
                    fea_s = fea_d2s
            else:                               # not d2s but s2s
                fea_s = fea_s2ds[-self.s_out_num[index]:, :]
        else:                                   # neither d2s nor s2s
            fea_s = None


        ### add bias (bias is only used in the last 1x1 conv) ###
        if index == self.ns: # last stage
            fea_d += self.bias.view(1, -1, 1, 1)
        
        return fea_d, fea_s
    
    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature [B, C, H, W]
        x[1]: spatial mask [B, 1, H, W]
        '''
        if self. training:
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

            out = []
            fea = x[0]
            for i in range(self.ns):
                if i == 0:
                    fea = self.conv[i](fea)
                    fea = fea * ch_mask[:, :, i:i+1, :1] * spa_mask + fea * ch_mask[:, :, i: i+1, 1:]
                else:
                    fea_d = self.conv[i](fea * ch_mask[:, :, i-1:i, 1:])
                    fea_s = self.conv[i](fea * ch_mask[:, :, i-1:i, :1])
                    fea = fea_d * ch_mask[:, :, i:i+1, :1] * spa_mask + fea_d * ch_mask[:, :, i:i+1, 1:] + \
                          fea_s * ch_mask[:, :, i:i+1, :1] * spa_mask + fea_s * ch_mask[:, :, i:i+1, 1:] * spa_mask
                    
                fea = self.relu(fea)
                out.append(fea)

            out = self.collect(torch.cat(out,1))

            return out, ch_mask
        
        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            feas = []

            for i in range(self.ns):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k = 3, index=i)
                feas.append([fea_d, fea_s])

                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))
            
            # 1x1 conv
            fea_dense = torch.cat(fea_dense, 1)
            fea_sparse = torch.cat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index = self.ns)

            return out, feas
        
class SMLargeModule(nn.Module):
    def __init__ (self, ns):
        super(SMLargeModule, self).__init__()
        self.ns = ns

        # spatial mask
        self.spa_mask = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 3, 2, 1, output_padding=1),
        )

        self.body = SMB(16, 16, ns=ns) # include 1x1 conv
        self.ca = ChannelAttention(16)
        self.tau = 1
        
    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        if self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = gumbel_softmax(spa_mask, 1, self.tau)

            out, ch_mask = self.body([x, spa_mask[:, :1, ...]])
            out = self.ca(out) + x

            return out, [spa_mask[:, :1, ...], ch_mask]
        
        if not self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = (spa_mask[:, :1, ...] > spa_mask[:, 1:, ...]).float()

            out, feas = self.body([x, spa_mask])
            out = self.ca(out) + x

            return out, feas

    
class SmallModule(nn.Module):
    def __init__(self, ns):
        super(SmallModule, self).__init__()

        self.ns = ns

        self.conv = nn.ModuleList()
        for i in range(ns):
            self.conv.append(nn.Conv2d(16, 4, 1, 1, 0))
            self.conv.append(nn.Conv2d(4, 16, 3, 1, 1))

        for i in range(len(self.conv)):
            self.conv[i].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.conv[i].weight)

    def forward(self, x, stages=[]):
        for s in stages:
            assert (s < self.ns) and (s >= 0), f"[ERRO] invalid stage {s}"

        z = x
        if stages:
            for s in range(self.ns):
                if s in stages:
                    z = F.relu(self.conv[2*s](z))
                    z = F.relu(self.conv[2*s+1](z))
            return z
        else:
            feas = []
            for s in range(self.ns):
                z = F.relu(self.conv[2*s](z))
                z = F.relu(self.conv[2*s+1](z))
                feas.append(z)
            return z, feas
        

class FusionNet_7_ns_SMSR(nn.Module): #hardcode
    def __init__(self, scale=2):
        super(FusionNet_7_ns_SMSR, self).__init__()

        self.scale = scale
        self.ns = 4 # for testing

        self.head = nn.ModuleList() # feature map 
        self.branch = nn.ModuleList() # 2 branches: Simple and Complex
        self.tail = nn.ModuleList() # concate output features
        self.mask = nn.ModuleList() # mask before put into branches

        self.head.append(nn.Conv2d( 1, 32, 5, 1, 2)) #0
        self.head.append(nn.Conv2d(32, 16, 1, 1, 0)) #1

        self.branch.append(SMLargeModule(self.ns))
        self.branch.append(SmallModule(self.ns))
        
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

    def forward_merge_mask(self, x, masks: dict, fea_out=False):
        # mask in masks are binary; 1.0 uses for C or branch 0, and vice versa
        z = x
        z = F.relu(self.head[0](z))
        z = F.relu(self.head[1](z))

        feas = []

        for ii in range(self.ns):
            branch_fea_0 = self.branch[0](z, stages=[ii])
            branch_fea_1 = self.branch[1](z, stages=[ii])
            
            assert ii in masks, f"[ERRO]: missing mask for stage {ii}"

            merge_map = masks[ii]
            merge_fea = branch_fea_0 * merge_map + branch_fea_1 * (1.0 - merge_map)

            z = merge_fea
            feas.append(merge_fea)
        
        z = F.relu(self.tail[0](merge_fea))
        z = self.tail[1](z)

        y = residual_stack(z, x, self.scale)

        if fea_out:
            return y, feas

        return y