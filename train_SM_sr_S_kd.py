import os
import torch
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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(epoch, optim):
    perfs = []
    total_loss = 0
    for batch_ids, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x = x.cuda()
        yt = yt.cuda()

        yf, sparsity = core.forward(x, branch=0)
        perf = evaluation.calculate(args, yf, yt)

        loss_func = loss.create_loss_func(args.loss)

        loss_SR = loss_func(yf, yt)
        loss_sparsity = sparsity.mean() # we try to reduct the sparsity to perseve information from features
        lambda_0=0.1
        lambda_sparsity = min((epoch-1) / 50, 1) * lambda_0
        batch_loss = loss_SR + lambda_sparsity * loss_sparsity

        optim.zero_grad()
        batch_loss.backward()

        optim.step()
        total_loss += batch_loss

        perfs.append(perf.cpu())

    perf = torch.stack(perfs, 0).mean()

    log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f' % (epoch, perf, total_loss)
    print(log_str)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=1)

core = model.config(args)
core.cuda()
