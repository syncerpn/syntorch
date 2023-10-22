import os
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import tqdm
import matplotlib.pyplot as plt

#custom modules
import data
import evaluation
import model
from option import parser
from template import test_sr_fusionnet_t as template
import numpy as np
import cv2

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)
    
def save_all_spatial_masks(idx: int, spatial_masks: list):
    # spatial_masks: list of mask (torch.tensor)
    dir = './image_masks'
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    for i, mask in enumerate(spatial_masks):
        mask = mask[0, ...]
        mask = mask.cpu().numpy().transpose(1, 2, 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        plt.imsave(os.path.join(dir, f'mask_{idx}_layer_{i}.jpg'), mask)
        
    return

def test_hourglass():
    perf_fs_masked = []
    perf_fs_nomask = []
    perf_fs_val = []
    
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x = x.cuda()
        yt = yt.cuda()
        core.train()

        # masked train mode
        with torch.no_grad():
            yf, ch_mask = core.forward(x, masked=True)
        perf_f_masked = evaluation.calculate(args, yf, yt)
        perf_fs_masked.append(perf_f_masked.cpu())
        
        spatial_masks = core.get_spatial_masks()
        save_all_spatial_masks(batch_idx, spatial_masks)
        
        with torch.no_grad():
            yf, ch_mask = core.forward(x, masked=False)
        perf_f_nomask = evaluation.calculate(args, yf, yt)
        perf_fs_nomask.append(perf_f_nomask.cpu())
        
        core.eval()
        with torch.no_grad():
            yf, ch_mask = core.forward(x)
        perf_f_val = evaluation.calculate(args, yf, yt)
        perf_fs_val.append(perf_f_val.cpu())

    mean_masked = torch.stack(perf_fs_masked, 0).mean()
    mean_nomask = torch.stack(perf_fs_nomask, 0).mean()
    mean_val = torch.stack(perf_fs_val, 0).mean()
    log_str = f"[INFO] Masked perf {mean_masked} | No-mask Perf {mean_nomask} | Val Perf {mean_val}"
    print(log_str)
        
# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()

test_hourglass()

