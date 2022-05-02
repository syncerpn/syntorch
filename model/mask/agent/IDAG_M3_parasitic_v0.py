"""
@author: nghiant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.agent import module

class IDAG_M3_parasitic_v0(nn.Module):
    def __init__(self, target_layer_index, immc=4):
        super(IDAG_M3_parasitic_v0, self).__init__()
        self.immc = immc

        self.agent_list = {}

        self.target_layer_index = target_layer_index

        for l in self.target_layer_index:
            if l == 1:
                self.agent_list[l] = module.unit_agent_uni_1x1(64, 32, 0, immc)
            elif l == 6:
                self.agent_list[l] = module.unit_agent_uni_1x1(32, 64, 0, immc)
            else:
                self.agent_list[l] = module.unit_agent_mix_1x1_3x3(32, 32, 0, immc)

    def load_state_dict(self, state_dict):
        for l in self.target_layer_index:
            self.agent_list[l].load_state_dict(state_dict[l])

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

    # def save_params(self):
    #     file_prefix = 'agent_'
    #     for l in self.target_layer_index:
    #         if l == 1:
    #             i = 0
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = self.agent_list[l].conv[i].bias.data.numpy()
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)

    #             i = i+1
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = self.agent_list[l].conv[i].bias.data.numpy()
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)

    #         elif l == 6:
    #             i = 0
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = self.agent_list[l].conv[i].bias.data.numpy()
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)

    #             i = i+1
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = self.agent_list[l].conv[i].bias.data.numpy()
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)
    #         else:
    #             i = 0
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = np.zeros((4,), dtype=np.float32)
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)

    #             i = i+1
    #             file_name = file_prefix + str(l) + '_' + str(i)
    #             with open(file_name, 'w') as f:
    #                 bias = np.zeros((32,), dtype=np.float32)
    #                 weight = self.agent_list[l].conv[i].weight.data.numpy().flatten()
    #                 data = np.concatenate((bias, weight))
    #                 data.tofile(f)