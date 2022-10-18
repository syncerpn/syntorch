import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
import shutil

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print('[INFO] Setting learning_rate to %.2E'%lr)

#nghiant: 2022/10/18, pull this from its grave to the newer version
#nghiant: derived from oriinal pytorch implementation; somehow improve the SR PSNR by 0.2 point, which is quite super
from typing import Union, Iterable
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def normalize_grad_(
        parameters: _tensor_or_tensors, ratio: float, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        ratio (float or int): amount of grad to keep
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) != 0:
        device = parameters[0].grad.device
        for p in parameters:
            grad_norm = torch.norm(p.grad.detach(), norm_type).to(device)
            p.grad.detach().mul_(torch.tensor(ratio).to(p.grad.device)/grad_norm)