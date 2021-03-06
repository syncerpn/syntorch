import torch
import torch.nn as nn

def residual_stack(z, x, scale=2):
    _,_,h,w = z.size()
    y = torch.repeat_interleave(torch.repeat_interleave(x,scale,dim=2),scale,dim=3)
    for ih in range(scale):
        for iw in range(scale):
            y[:,:,ih:h*scale:scale, iw:w*scale:scale] += z[:,ih*scale+iw:ih*scale+iw+1,:,:]
    return y

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)