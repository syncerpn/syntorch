import torch
import torch.nn as nn
from scipy.optimize import root_scalar
import numpy as np
import cv2

def objfun(x, b, k):
    a = b*x
    a = np.where(a<1, a, 1)
    return sum(a) - k

class GradientSobelFilter:
    def __init__(self):
        self.sfx = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sfx.weight.data = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        self.sfx.to('cuda')

        self.sfy = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sfy.weight.data = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
        self.sfy.to('cuda')

    def generate_mask(self, x, p):
        grad_map_x = self.sfx.forward(x)
        grad_map_y = self.sfy.forward(x)
        grad = abs(grad_map_x) + abs(grad_map_y)

        grad_sorted, _ = torch.sort(grad.view(-1), descending=True)
        grad_sorted_index = min(max(int(p * torch.numel(grad)), 0), torch.numel(grad)-1)
        grad_sorted_threshold = grad_sorted[grad_sorted_index]
        
        merge_map = (grad > grad_sorted_threshold).type(torch.float32)

        return merge_map

class RandomFlatMasker:
    def __init__(self):
        pass

    def generate_mask(self, mask_shape, p):
        merge_map = torch.rand(mask_shape)
        merge_map = (merge_map < p).type(torch.float32)
        # merge_map = merge_map.repeat([branch_fea_0.shape[0],branch_fea_0.shape[1],1,1])

        merge_map = merge_map.to('cuda')
        return merge_map
    
class SMSRMaskFuse:
    def __init__(self, xC, xS, spa_mask, ch_mask, sp):
        self.xC = xC
        self.xS = xS
        self.sp = sp
        self.spa_mask = spa_mask # 1 layer
        self.ch_mask = ch_mask # 2 layer
        
    def optimal_sampling(self, x, sp):
        """Guided sampling
        
        Args:
            spa_mask (torch.tensor): Probability mask
            sp (float): Sampling rate [0, 1]
        Return sampled mask
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            x = np.squeeze(x)
        rows, cols = x.shape
        n = rows*cols
        myfunc = lambda v: objfun(v, x.reshape(-1), round(n*sp))
        
        sol = root_scalar(myfunc, method='toms748', bracket=[0, 1e16])
        tau = sol.root
        tmp1 = np.where(x*tau < 1, x*tau, 1)
        p = np.where(tmp1>1e-16, tmp1, 1e-16)
        rand = np.random.rand(rows, cols)
        out = np.where(rand <= p, 1, 0)
        dense_spa = torch.from_numpy(out).reshape(1, 1, rows, cols).cuda()
        sparse_spa = torch.from_numpy(1 - out).reshape(1, 1, rows, cols).cuda()
        return dense_spa, sparse_spa
    
    def sampling_fuse(self, spatial_only=False):
        
        dense_spa_mask, sparse_spa_mask = self.optimal_sampling(self.spa_mask, sp=self.sp)
        dense_ch_mask = self.ch_mask[:, :, :1]
        sparse_ch_mask = self.ch_mask[:, :, 1:]
        
        if spatial_only:
            out =  self.xC * dense_spa_mask 
        else:
            out = self.xC * dense_ch_mask.view(1, -1, 1, 1) + self.xC * sparse_ch_mask.view(1, -1, 1, 1) * dense_spa_mask + \
                self.xS * sparse_ch_mask.view(1, -1, 1, 1) * sparse_spa_mask
        return out.cuda()
    
    def normal_fuse(self):
        dense_spa_mask = self.spa_mask.round()
        sparse_spa_mask = 1 - dense_spa_mask
        
        dense_ch_mask = self.ch_mask[:, :, 0].round()
        sparse_ch_mask = self.ch_mask[:, :, 1].round()

        out = self.xC * dense_ch_mask.view(1, -1, 1, 1) + self.xC * sparse_ch_mask.view(1, -1, 1, 1) * dense_spa_mask \
            + self.xS * sparse_ch_mask.view(1, -1, 1, 1) * sparse_spa_mask
        return out.cuda()


