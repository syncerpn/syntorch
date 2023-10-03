import os
import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import tqdm

#custom modules
import data
import evaluation
import loss
import model
import optimizer
import utils
from option import parser
from template import test_sr_fusionnet_t as template
import numpy as np

from mask_generator_lib import GradientSobelFilter, RandomFlatMasker

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

def single_forward(branch):
    print(f"Testing for branch {branch}")
    for psi in range(5):
        perf_fs = []
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward(x,branch)
            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())
        mean_perf_f = torch.stack(perf_fs, 0).mean()
        log_str = f"[INFO] P: {mean_perf_f}"

def single_forward_v2(branch):
    for psi in range(5):
        print(f"\nExp {psi}")
        perf_fs = []
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf, ch_mask = core.forward(x,branch)
            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())
           
        print(core.get_sparsity_inference())

        mean_perf_f = torch.stack(perf_fs, 0).mean()
        log_str = f"[INFO] P: {mean_perf_f}"
        print(log_str)
        
def single_forward_v2(branch):
    for psi in range(5):
        print(f"\nExp {psi}")
        perf_fs = []
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
 
            x = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf, ch_mask = core.forward(x,branch)
            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())
         
           
        print(core.get_sparsity_inference())

        mean_perf_f = torch.stack(perf_fs, 0).mean()
        log_str = f"[INFO] P: {mean_perf_f}"
        print(log_str)
        
def pretrained_infer(branch):
    perfs= []
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total= len(XYtest)):
        x = x.cuda()
        yt = yt.cuda()
        with torch.no_grad():
        
            yf, ch_mask = core.forward(x, branch)
            perf = evaluation.calculate(args, yf, yt)
            print(f"Branch {batch_idx} - perf {perf}")
            perfs.append(perf)
         
    if len(perfs) > 0:
        perfs = torch.stack(perfs, 0).mean()
        print(f"Mean branch {branch} perf: {perfs}")
        # print(core.get_sparsity_inference())


# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()
core.eval()

pretrained_infer(1)

# print("Large\n")
# single_forward_v2(0)
# print("\nSmall\n")
# single_forward_v2(1)
