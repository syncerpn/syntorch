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
    for psi in range(11):
        perf_fs = []
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward(x,branch)
                print(yf.detach().cpu().size())




# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()
core.eval()

print("Large\n")
single_forward(0)
print("\nSmall\n")
single_forward(1)