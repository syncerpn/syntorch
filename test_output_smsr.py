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

def write_to_file(text, file, mode='a'):
    with open(file, mode) as f:
        f.write(f"{text}\n")

save_output_file = "test_output.txt"
write_to_file("Test output\n", save_output_file, 'w')

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

def compare_output(branch):
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total= len(XYtest)):
        write_to_file(f"Batch {batch_idx}", save_output_file)
        x = x.cuda()
        yt = yt.cuda()
        
        # training forward
        core.train()
        with torch.no_grad():
            yf_train, sparsity = core.forward(x, branch)
            perf_train = evaluation.calculate(args, yf_train, yt)
        write_to_file(f"Perf train: {perf_train}", save_output_file)
            
        # evaluation forward
        core.eval()
        with torch.no_grad():
            yf_val, _ = core.forward(x, branch)
            perf_val = evaluation.calculate(args, yf_val, yt)
        write_to_file(f"Perf val: {perf_val}", save_output_file)
            
        write_to_file(f"Sparsity {batch_idx}: {sparsity.mean()}", save_output_file)
        write_to_file(f"Check similarity {batch_idx}: {(torch.abs(yf_val - yf_train) <= 1e-1).float().mean()}", save_output_file)
       
# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()

compare_output(0)