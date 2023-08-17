import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.IDAG_M6 import IDAG_M6
from model.IDAG_M6_r3 import IDAG_M6_r3
from model.IDAG_M5 import IDAG_M5
from model.IDAG_M5_m16 import IDAG_M5_m16
from model.IDAG_M4 import IDAG_M4
from model.IDAG_M3 import IDAG_M3
from model.IDAG_M3_g4 import IDAG_M3_g4
from model.IDAG_M3_KD import IDAG_M3_KD
from model.IDAG_M3_KD2 import IDAG_M3_KD2
from model.IDAG_M3_KD3 import IDAG_M3_KD3
from model.IDAG_M3_KD3s import IDAG_M3_KD3s
from model.IDAG_M2 import IDAG_M2
from model.IDAG_M1 import IDAG_M1
from model.IDAG_M1_l32 import IDAG_M1_l32
from model.IDAG_M1_l64 import IDAG_M1_l64
from model.IDAG_M1_r3 import IDAG_M1_r3
from model.IDAG_M1_c3 import IDAG_M1_c3
from model.IDAG_M1P import IDAG_M1P
from model.IDAG_M3E import IDAG_M3E
from model.SVDSR import SVDSR
from model.SMSR import SMSR
from model.FusionNet import FusionNet
from model.FusionNet_2 import FusionNet_2
from model.FusionNet_3 import FusionNet_3
from model.FusionNet_4 import FusionNet_4
from model.FusionNet_5 import FusionNet_5
from model.FusionNet_6 import FusionNet_6
from model.FusionNet_7 import FusionNet_7
from model.FusionNet_8 import FusionNet_8
from model.FusionNet_9 import FusionNet_9

from model.FusionNet_7_debug import FusionNet_7_debug
from model.FusionNet_7_gsi import FusionNet_7_gsi
from model.FusionNet_7_gsi_mirror import FusionNet_7_gsi_mirror
from model.FusionNet_6_gsi import FusionNet_6_gsi

from model.FusionNet_7_2s import FusionNet_7_2s
from model.FusionNet_7_3s import FusionNet_7_3s

def config(args):
    arch = args.core.split('-')
    name = arch[0]
    if   (name == 'IDAG_M1'):
        core = IDAG_M1(scale=args.scale)
    elif (name == 'IDAG_M1_l32'):
        core = IDAG_M1_l32(scale=args.scale)
    elif (name == 'IDAG_M1_l64'):
        core = IDAG_M1_l64(scale=args.scale)
    elif (name == 'IDAG_M1_r3'):
        core = IDAG_M1_r3(scale=args.scale)
    elif (name == 'IDAG_M1_c3'):
        core = IDAG_M1_c3(scale=args.scale)
    elif (name == 'IDAG_M1P'):
        core = IDAG_M1P(scale=args.scale)
    elif (name == 'IDAG_M2'):
        core = IDAG_M2(scale=args.scale)
    elif (name == 'IDAG_M4'):
        core = IDAG_M4(scale=args.scale)
    elif (name == 'IDAG_M6_r3'):
        core = IDAG_M6_r3(scale=args.scale)
    elif (name == 'IDAG_M5'):
        core = IDAG_M5(scale=args.scale)
    elif (name == 'IDAG_M5_m16'):
        core = IDAG_M5_m16(scale=args.scale)
    elif (name == 'IDAG_M6'):
        core = IDAG_M6(scale=args.scale)
    elif (name == 'IDAG_M3'):
        core = IDAG_M3(scale=args.scale)
    elif (name == 'IDAG_M3_g4'):
        core = IDAG_M3_g4(scale=args.scale)
    elif (name == 'IDAG_M3_KD'):
        core = IDAG_M3_KD(scale=args.scale)
    elif (name == 'IDAG_M3_KD2'):
        core = IDAG_M3_KD2(scale=args.scale)
    elif (name == 'IDAG_M3_KD3'):
        core = IDAG_M3_KD3(scale=args.scale)
    elif (name == 'IDAG_M3_KD3s'):
        core = IDAG_M3_KD3s(scale=args.scale)
    elif (name == 'SVDSR'): #SVDSR-n_layer-filter_size
        n_layer = int(arch[1])
        filter_size = int(arch[2])
        core = SVDSR(n_layer, filter_size=filter_size, scale=args.scale)
    elif (name == 'SMSR'): #SVDSR-n_layer-filter_size
        n_feats = int(arch[1])
        core = SMSR(n_feats=n_feats, rgb_range=args.rgb_range, style=args.style, scale=args.scale)
    elif (name == 'FusionNet'):
        core = FusionNet(scale=args.scale)
    elif (name == 'FusionNet_2'):
        core = FusionNet_2(scale=args.scale)
    elif (name == 'FusionNet_3'):
        core = FusionNet_3(scale=args.scale)
    elif (name == 'FusionNet_4'):
        core = FusionNet_4(scale=args.scale)
    elif (name == 'FusionNet_5'):
        core = FusionNet_5(scale=args.scale)
    elif (name == 'FusionNet_6'):
        core = FusionNet_6(scale=args.scale)
    elif (name == 'FusionNet_7'):
        core = FusionNet_7(scale=args.scale)
    elif (name == 'FusionNet_8'):
        core = FusionNet_8(scale=args.scale)
    elif (name == 'FusionNet_9'):
        core = FusionNet_9(scale=args.scale)

    elif (name == 'FusionNet_7_debug'):
        core = FusionNet_7_debug(scale=args.scale)

    elif (name == 'FusionNet_7_gsi'):
        core = FusionNet_7_gsi(scale=args.scale)
    elif (name == 'FusionNet_7_gsi_mirror'):
        core = FusionNet_7_gsi_mirror(scale=args.scale)

    elif (name == 'FusionNet_6_gsi'):
        core = FusionNet_6_gsi(scale=args.scale)

    elif (name == 'FusionNet_7_2s'):
        core = FusionNet_7_2s(scale=args.scale)
    elif (name == 'FusionNet_7_3s'):
        core = FusionNet_7_3s(scale=args.scale)
        
    else:
        print('[ERRO] unknown model tag')
        assert(0)

    if args.checkpoint is not None:
        print('[INFO] load core from torch checkpoint: ' + args.checkpoint)
        if callable(getattr(core, "load", None)):
            core.load(args.checkpoint)
        else:
            checkpoint_data = torch.load(args.checkpoint)
            # to_remove = []
            # for k in checkpoint_data:
            #     if 'branch.1' in k:
            #         to_remove.append(k)

            # for k in to_remove:
            #     del(checkpoint_data[k])
                
            core.load_state_dict(checkpoint_data, strict=False)

    return core

def config_kd_teacher(args):
    arch = args.kd_teacher_core.split('-')
    name = arch[0]
    if   (name == 'IDAG_M1'):
        core = IDAG_M1(scale=args.scale)
    elif (name == 'IDAG_M1P'):
        core = IDAG_M1P(scale=args.scale)
    elif (name == 'IDAG_M2'):
        core = IDAG_M2(scale=args.scale)
    elif (name == 'IDAG_M3'):
        core = IDAG_M3(scale=args.scale)
    elif (name == 'IDAG_M3_KD'):
        core = IDAG_M3_KD(scale=args.scale)
    elif (name == 'IDAG_M3_KD2'):
        core = IDAG_M3_KD2(scale=args.scale)
    elif (name == 'IDAG_M3_KD3'):
        core = IDAG_M3_KD3(scale=args.scale)
    elif (name == 'IDAG_M3_KD3s'):
        core = IDAG_M3_KD3s(scale=args.scale)
    elif (name == 'SVDSR'): #SVDSR-n_layer-filter_size
        n_layer = int(arch[1])
        filter_size = int(arch[2])
        core = SVDSR(n_layer, filter_size=filter_size, scale=args.scale)
    elif (name == 'SMSR'): #SVDSR-n_layer-filter_size
        n_feats = int(arch[1])
        core = SMSR(n_feats=n_feats, rgb_range=args.rgb_range, style=args.style, scale=args.scale)
    else:
        print('[ERRO] unknown model tag')
        assert(0)

    if args.kd_teacher_checkpoint is not None:
        print('[INFO] load kd teacher core from torch checkpoint: ' + args.kd_teacher_checkpoint)
        if callable(getattr(core, "load", None)):
            core.load(args.kd_teacher_checkpoint)
        else:
            checkpoint_data = torch.load(args.kd_teacher_checkpoint)
            core.load_state_dict(checkpoint_data)

    return core


print('[ OK ] Module "model"')