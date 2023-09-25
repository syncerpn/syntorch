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
    perf_trains = []
    perf_vals = []
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total= len(XYtest)):
        write_to_file(f"Batch {batch_idx}", save_output_file)
        x = x.cuda()
        yt = yt.cuda()
        
        # training forward
        core.train()
        with torch.no_grad():
            yf_train, sparsity_train = core.forward(x, branch)
            perf_train = evaluation.calculate(args, yf_train, yt)
        write_to_file(f"Perf train: {perf_train}", save_output_file)
        perf_trains.append(perf_train)
            
        # evaluation forward
        core.eval()
        with torch.no_grad():
            yf_val, sparsity_val = core.forward(x, branch)
            perf_val = evaluation.calculate(args, yf_val, yt)
        write_to_file(f"Perf val: {perf_val}", save_output_file)
        perf_vals.append(perf_val)
            
        write_to_file(f"Density train: {sparsity_train.mean()}", save_output_file)
        write_to_file(f"Density val: {sparsity_val.mean()}", save_output_file)
        write_to_file(f"Check similarity: {(torch.abs(yf_val - yf_train) <= 1e-1).float().mean()}", save_output_file)
        write_to_file(f"Mean difference: {torch.abs(yf_val - yf_train).mean()}", save_output_file)
        
    perf_trains = torch.stack(perf_trains, 0)
    perf_vals = torch.stack(perf_vals, 0)
    write_to_file(50*"=", save_output_file)
    write_to_file(f"Mean perf train: {perf_trains.cpu().mean()}", save_output_file)
    write_to_file(f"Mean perf val: {perf_vals.cpu().mean()}", save_output_file)
    
    
def compare_psnr_by_dense(branch):
    perf_trains = []
    perf_vals = []
    perf_train_softs = []
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total= len(XYtest)):
        write_to_file(f"Batch {batch_idx}", save_output_file)
        x = x.cuda()
        yt = yt.cuda()
        
        # training forward
        core.train()
        with torch.no_grad():
            yf_train, _ = core.forward(x, branch, masked=False)
            perf_train = evaluation.calculate(args, yf_train, yt)
        write_to_file(f"PSNR with mask removed: {perf_train}", save_output_file)
        perf_trains.append(perf_train)
        
        # training forward - soft mask
        core.train()
        with torch.no_grad():
            yf_train, _ = core.forward(x, branch, masked=True)
            perf_train_soft = evaluation.calculate(args, yf_train, yt)
        write_to_file(f"PSNR with soft mask: {perf_train_soft}", save_output_file)
        perf_train_softs.append(perf_train_soft)
            
        # evaluation forward
        core.eval()
        with torch.no_grad():
            yf_val, sparsity_val = core.forward(x, branch)
            perf_val = evaluation.calculate(args, yf_val, yt)
        write_to_file(f"PSNR with hard mask: {perf_val}", save_output_file)
        perf_vals.append(perf_val)
            
        # write_to_file(f"Density train: {sparsity_train.mean()}", save_output_file)
        write_to_file(f"Density val: {sparsity_val.mean()}", save_output_file)
        write_to_file(f"Check similarity: {(torch.abs(yf_val - yf_train) <= 1e-1).float().mean()}", save_output_file)
        write_to_file(f"Mean difference: {torch.abs(yf_val - yf_train).mean()}", save_output_file)
        
    perf_train_softs = torch.stack(perf_train_softs, 0)
    perf_trains = torch.stack(perf_trains, 0)
    perf_vals = torch.stack(perf_vals, 0)
    write_to_file(50*"=", save_output_file)
    write_to_file(f"Mean PSNR with mask removed: {perf_trains.cpu().mean()}", save_output_file)
    write_to_file(f"Mean PSNR with soft mask: {perf_train_softs.cpu().mean()}", save_output_file)
    write_to_file(f"Mean PSNR with hard mask: {perf_vals.cpu().mean()}", save_output_file)
       
# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()

compare_psnr_by_dense(0)