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

from mask_generator_lib import GradientSobelFilter, RandomFlatMasker

if args.template is not None:
    template.set_template(args)

def merge_test():
    for psi in range(11):
        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                merge_map = rfm.generate_mask([1, 1, x.shape[2], x.shape[3]], psi/10)
                yf = core.forward_merge_mask(x, {0: merge_map})

            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - MERGE RANDOM - PSI: {psi/10:.1f} - P: {mean_perf_f:.3f}'
        print(log_str)


# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

rfm = RandomFlatMasker()
gsf = GradientSobelFilter()

merge_test(branches=[0,1])