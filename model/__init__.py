import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FusionNet_7_debug import FusionNet_7_debug
from model.FusionNet_7_gsi import FusionNet_7_gsi
from model.FusionNet_6_gsi import FusionNet_6_gsi

from model.FusionNet_7_1s import FusionNet_7_1s
from model.FusionNet_7_2s import FusionNet_7_2s
from model.FusionNet_7_3s import FusionNet_7_3s
from model.FusionNet_7_4s import FusionNet_7_4s

def config(args):
    arch = args.core.split("-")
    name = arch[0]
    if   (name == "FusionNet_6"):
        core = FusionNet_6(scale=args.scale)
    elif (name == "FusionNet_7"):
        core = FusionNet_7(scale=args.scale)
    elif (name == "FusionNet_8"):
        core = FusionNet_8(scale=args.scale)
    elif (name == "FusionNet_9"):
        core = FusionNet_9(scale=args.scale)

    elif (name == "FusionNet_7_debug"):
        core = FusionNet_7_debug(scale=args.scale)

    elif (name == "FusionNet_7_gsi"):
        core = FusionNet_7_gsi(scale=args.scale)

    elif (name == "FusionNet_6_gsi"):
        core = FusionNet_6_gsi(scale=args.scale)

    elif (name == "FusionNet_7_1s"):
        core = FusionNet_7_1s(scale=args.scale)
    elif (name == "FusionNet_7_2s"):
        core = FusionNet_7_2s(scale=args.scale)
    elif (name == "FusionNet_7_3s"):
        core = FusionNet_7_3s(scale=args.scale)
    elif (name == "FusionNet_7_4s"):
        core = FusionNet_7_4s(scale=args.scale)
        
    else:
        print("[ERRO] unknown model tag")
        assert(0)

    if args.checkpoint is not None:
        print("[INFO] load core from torch checkpoint: " + args.checkpoint)
        checkpoint_data = torch.load(args.checkpoint)
        core.load_state_dict(checkpoint_data, strict=False)

    return core

print("[ OK ] Module \"model\"")