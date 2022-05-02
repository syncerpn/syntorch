import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mask.core.IDAG_M3 import IDAG_M3
from model.mask.core.IDAG_M1 import IDAG_M1
from model.mask.core.IDAG_M3E import IDAG_M3E
from model.mask.core.IDAG_M1P import IDAG_M1P
from model.mask.core.SVDSR import SVDSR
from model.mask.core.SMSR import SMSR

def config(args):
    arch = args.core.split('-')
    name = arch[0]
    if   (name == 'IDAG_M1'):
        core = IDAG_M1(scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    elif (name == 'IDAG_M3'):
        core = IDAG_M3(scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    elif (name == 'IDAG_M1P'):
        core = IDAG_M1P(scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    elif (name == 'IDAG_M3E'):
        core = IDAG_M3E(scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    elif (name == 'SVDSR'):
        n_layer = int(arch[1])
        filter_size = int(arch[2])
        core = SVDSR(n_layer, filter_size=filter_size, scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    elif (name == 'SMSR'):
        n_feats = int(arch[1])
        core = SMSR(n_feats=n_feats, rgb_range=args.rgb_range, style=args.style, scale=args.scale, target_layer_index=args.target_layer_index, mask_type=args.mask_type)
    else:
        print('[ERRO] unknown model tag')
        assert(0)

    if args.checkpoint is not None:
        print('[INFO] load core from checkpoint: ' + args.checkpoint)
        if callable(getattr(core, "load", None)):
            core.load(args.checkpoint)
        else:
            checkpoint_data = torch.load(args.checkpoint)
            core.load_state_dict(checkpoint_data)

    return core

print('[ OK ] Module "model.mask.core"')