"""
@author: nghiant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.agent import module

class SVDSR_parasitic_v1(nn.Module):
    def __init__(self, target_layer_index, immc=4):
        super(SVDSR_parasitic_v1, self).__init__()
        self.immc = immc

        self.agent_list = {}

        self.target_layer_index = target_layer_index

        for l in self.target_layer_index:
            if l == 0:
                self.agent_list[l] = module.unit_agent_uni_3x3( 1, 64, 1, immc)
            else:
                self.agent_list[l] = module.unit_agent_uni_1x1(64, 64, 0, immc)

    def load_state_dict(self, state_dict):
        for l in self.target_layer_index:
            agent_list[l].load_state_dict(state_dict[l])

    def forward(self, input_dict):
        #forward method for getting the masks
        agent_fmap = {}
        for l in input_dict:
            if l not in self.target_layer_index:
                print('[ERRO] core layer ' + str(l) + ' does not have an agent')
                assert(0)
            agent_fmap[l] = self.agent_list[l](input_dict[l])

        return agent_fmap

    def parameters(self):
        all_params = []
        for l in self.target_layer_index:
            all_params += list(self.agent_list[l].parameters())

        return all_params