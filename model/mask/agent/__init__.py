import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.agent.IDAG_M1P_parasitic_v0 import IDAG_M1P_parasitic_v0
from model.mask.agent.IDAG_M1_parasitic_v0 import IDAG_M1_parasitic_v0
from model.mask.agent.IDAG_M3_parasitic_v0 import IDAG_M3_parasitic_v0
from model.mask.agent.IDAG_M3_parasitic_v1 import IDAG_M3_parasitic_v1
from model.mask.agent.IDAG_M3_parasitic_v2 import IDAG_M3_parasitic_v2
from model.mask.agent.IDAG_M3_parasitic_v3 import IDAG_M3_parasitic_v3
from model.mask.agent.IDAG_M3E_parasitic_v1 import IDAG_M3E_parasitic_v1
from model.mask.agent.SVDSR_parasitic_v0 import SVDSR_parasitic_v0
from model.mask.agent.SMSR_parasitic_v0 import SMSR_parasitic_v0

def config(args, core):
    arch = args.agent.split('-')
    name = arch[0]
    if (name == 'IDAG_M3_parasitic_v0'):
        immc = int(arch[1])
        agent = IDAG_M3_parasitic_v0(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M3_parasitic_v1'):
        immc = int(arch[1])
        agent = IDAG_M3_parasitic_v1(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M3_parasitic_v2'):
        immc = int(arch[1])
        agent = IDAG_M3_parasitic_v2(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M3_parasitic_v3'):
        immc = int(arch[1])
        agent = IDAG_M3_parasitic_v3(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M1_parasitic_v0'):
        immc = int(arch[1])
        agent = IDAG_M1_parasitic_v0(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M1P_parasitic_v0'):
        immc = int(arch[1])
        agent = IDAG_M1P_parasitic_v0(immc=immc, target_layer_index=core.target_layer_index)

    elif (name == 'IDAG_M3E_parasitic_v1'):
        immc = int(arch[1])
        agent = IDAG_M3E_parasitic_v1(immc=immc, target_layer_index=core.target_layer_index)

    elif (arch[0] == 'SVDSR_parasitic_v0'):
        immc = int(arch[1])
        agent = SVDSR_parasitic_v0(immc=immc, target_layer_index=core.target_layer_index)

    elif (arch[0] == 'SMSR_parasitic_v0'):
        immc = int(arch[1])
        agent = SMSR_parasitic_v0(immc=immc, target_layer_index=core.target_layer_index)
    
    else:
        print('[ERRO] unknown model tag')
        assert(0)

    if args.agent_checkpoint is not None:
        print('[INFO] load agent from checkpoint: ' + args.agent_checkpoint)
        checkpoint_data = torch.load(args.agent_checkpoint)
        agent.load_state_dict(checkpoint_data)

    return agent

print('[ OK ] Module "model.mask.agent"')