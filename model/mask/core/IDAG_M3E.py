"""
@author: nghiant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.agent.common import transform
from model.common import residual_stack
from model.IDAG_M3E import IDAG_M3E as Base_Model

class IDAG_M3E(Base_Model): #hardcode
    def __init__(self, scale=2, target_layer_index=[1,2,3,4,5,6], mask_type='sigmoid'):
        super(IDAG_M3E, self).__init__(scale)

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
        for i in range(7): # exclude the last layer
            if i in self.target_layer_index:
                if self.masks is not None:
                    mask = transform(self.masks[i], self.mask_type)
                elif self.generate_mask is not None:
                    masks = self.generate_mask({i:z})
                    mask = transform(masks[i], self.mask_type)
                    
                agent_input_fmap[i] = z

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

            if i in self.target_layer_index:
                if (self.masks is not None) or (self.generate_mask is not None):
                    z = z * mask
                    sparsity_layers[i] = 1.0 - torch.sum(mask, (1,2,3)) / (mask.numel() / mask.shape[0])
                else:
                    sparsity_layers[i] = 1.0 - torch.sum((z!=0).float(), (1,2,3)) / (z.numel() / z.shape[0])
                
                sparsity_percent += sparsity_layers[i] / len(self.target_layer_index)

                agent_output_fmap[i] = z

        z =        self.conv[7](z)

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