import os
import torch
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

class GradientSobelFilter:
    def __init__(self):
        self.sfx = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sfx.weight.data = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        self.sfx.to('cuda')

        self.sfy = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sfy.weight.data = torch.Tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
        self.sfy.to('cuda')

    def generate_mask(self, x, p):
        grad_map_x = self.sfx.forward(x)
        grad_map_y = self.sfy.forward(x)
        grad = abs(grad_map_x) + abs(grad_map_y)

        grad_sorted, _ = torch.sort(grad.view(-1), descending=True)
        grad_sorted_index = min(max(int(p * torch.numel(grad)), 0), torch.numel(grad)-1)
        grad_sorted_threshold = grad_sorted[grad_sorted_index]
        
        merge_map = (grad > grad_sorted_threshold).type(torch.float32)

        return merge_map

class RandomFlatMasker:
    def __init__(self):
        pass

    def generate_mask(self, mask_shape, p):
        merge_map = torch.rand(mask_shape)
        merge_map = (merge_map < p).type(torch.float32)
        # merge_map = merge_map.repeat([branch_fea_0.shape[0],branch_fea_0.shape[1],1,1])

        merge_map = merge_map.to('cuda')
        return merge_map