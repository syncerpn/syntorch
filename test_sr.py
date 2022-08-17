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
from template import test_sr_t as template

if args.template is not None:
    template.set_template(args)

def test():
    perf_fs = []
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf = core.forward(x)
    
        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = '[INFO] TS - P: %.4f' % (mean_perf_f)
    print(log_str)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.save_dn_module('m4_32.966_')
core.cuda()

test()