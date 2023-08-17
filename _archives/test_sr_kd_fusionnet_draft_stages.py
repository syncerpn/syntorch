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
from option import args
from template import test_sr_fusionnet_t as template
import numpy as np

if args.template is not None:
    template.set_template(args)

def test(branches=[]):
    for bri in branches:

        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf, feas = core.forward(x, branch=bri, fea_out=True)
            
            for fea_id, fea in enumerate(feas):
                np.save(f'{batch_idx}_{bri}_{fea_id}', fea.detach().cpu().numpy())

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - BRANCH_ID: {bri} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_test():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward_merge_random(x, p=psi/10)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE RANDOM - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_test_gradient_sobel():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward_merge_gradient_sobel(x, p=psi/10)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE GRADSO - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_test_gradient_sobel_stages(stages=[]):
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward_merge_gradient_sobel_stages(x, p=psi/10, stages=stages)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE GRADSO STAGES {stages} - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

def merge_test_gradient_sobel_stages_psis(stages_psis=[]):
    perf_fs = []
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf = core.forward_merge_gradient_sobel_stages_psis(x, stages_psis=stages_psis)

        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = f'[INFO] TS - MERGE GRADSO STAGES PSIS {stages_psis} - P: {mean_perf_f:.3f}'
    print(log_str)

def merge_test_first_layer_uncertainty():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward_merge_first_layer_uncertainty(x, p=psi/10)

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE GRADSO - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

# test([0,1])
# merge_test()
# merge_test_gradient_sobel()
# merge_test_first_layer_uncertainty()
# merge_test_gradient_sobel_stages(stages=[0])
# merge_test_gradient_sobel_stages(stages=[1])
# merge_test_gradient_sobel_stages(stages=[2])
# merge_test_gradient_sobel_stages(stages=[3])
# merge_test_gradient_sobel_stages(stages=[0,3])
merge_test_gradient_sobel_stages_psis(stages_psis={0:0.2,1:0.5,2:0.8,3:0.5})