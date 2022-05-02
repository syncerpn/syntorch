"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.agent.common import transform
from model.common import residual_stack
from model.SVDSR import SVDSR as Base_Model

class SVDSR(Base_Model): #hardcode
    def __init__(self, n_layer, filter_size=64, scale=2, target_layer_index=[0], mask_type='sigmoid'):
        super(SVDSR, self).__init__(n_layer, filter_size, scale)

        self.masks = None # ready to use mask
        self.generate_mask = None # agent mask generation method
        self.target_layer_index = target_layer_index
        self.mask_type='sigmoid'

    def forward(self, x):
        agent_input_fmap = {}
        agent_output_fmap = {}

        sparsity_percent = 0.0
        sparsity_layers = {}

        z = x
        for i in range(self.n_layer-1): # exclude the last layer
            if i in self.target_layer_index:
                if self.masks is not None:
                    mask = transform(self.masks[i], self.mask_type)
                elif self.generate_mask is not None:
                    masks = self.generate_mask({i:z})
                    mask = transform(masks[i], self.mask_type)
                    
                agent_input_fmap[i] = z

            z = F.relu(self.conv[i](z))

            if i in self.target_layer_index:
                if (self.masks is not None) or (self.generate_mask is not None):
                    z = z * mask
                    sparsity_layers[i] = 1.0 - torch.sum(mask, (1,2,3)) / (mask.numel() / mask.shape[0])
                else:
                    sparsity_layers[i] = 1.0 - torch.sum((z!=0).float(), (1,2,3)) / (z.numel() / z.shape[0])
                
                sparsity_percent += sparsity_layers[i] / len(self.target_layer_index)

                agent_output_fmap[i] = z

        z =        self.conv[len(self.conv)-1](z)

        y = residual_stack(z, x, self.scale)
        return y, agent_input_fmap, agent_output_fmap, sparsity_percent, sparsity_layers

    def attach_mask(self, masks, method='ready'):
        if method == 'ready':
            self.masks = masks
        elif method == 'generator':
            self.generate_mask = masks
        else:
            print('[ERRO] unknown masking method')
            assert(0)

    def detach_mask(self):
        self.masks = None
        self.generate_mask = None