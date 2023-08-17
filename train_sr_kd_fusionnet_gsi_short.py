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
from template import train_sr_fusionnet_t as template

if args.template is not None:
    template.set_template(args)

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch, optim):
    perfs = []

    total_loss = 0
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x  = x.cuda()
        yt = yt.cuda()
        
        yf = core.forward(x, branch=0)

        perf = evaluation.calculate(args, yf, yt)

        loss_func = loss.create_loss_func(args.loss)

        batch_loss = loss_func(yf, yt)
        optim.zero_grad()
        batch_loss.backward()

        optim.step()

        total_loss += batch_loss
        perfs.append(perf.cpu())

    perf = torch.stack(perfs, 0).mean()

    log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f' % (epoch, perf, total_loss)
    print(log_str)

def train_kd(epoch, optim):
    perfs = []

    total_loss = 0
    total_loss_sr = 0
    total_loss_feas = {}
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x  = x.cuda()
        yt = yt.cuda()
        
        y_teacher, feas_teacher, grad_map = core.forward(x, branch=0, fea_out=True)
        y_student, feas_student, _ = core.forward(x, branch=1, fea_out=True)

        perf = evaluation.calculate(args, y_student, yt)

        loss_func = loss.create_loss_func(args.loss)
        loss_sr = loss_func(y_student, yt)
        loss_feas = {}

        batch_loss = 0
        batch_loss += loss_sr

        for fi, fea_student, fea_teacher in zip(list(range(len(feas_student))), feas_student, feas_teacher):
            # loss_feas[fi] = loss_func(1.0 / (0.1 + grad_map) * fea_student, 1.0 / (0.1 + grad_map) * fea_teacher)
            loss_feas[fi] = loss_func(fea_student, fea_teacher)
            # loss_feas[fi] = 0
            batch_loss += loss_feas[fi]

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        total_loss += batch_loss
        total_loss_sr += loss_sr
        for fi in loss_feas:
            if fi not in total_loss_feas:
                total_loss_feas[fi] = 0
            total_loss_feas[fi] += loss_feas[fi]

        perfs.append(perf.cpu())

    perf = torch.stack(perfs, 0).mean()

    log_str = f'[INFO] E: {epoch} | P: {perf:.3f} | LOSS: {total_loss:.3f} | LOSS_SR: {total_loss_sr:.3f} | LOSS_FEAS: {" ".join([format(total_loss_feas[fi], ".3f") for fi in total_loss_feas])}'
    print(log_str)

def test(epoch, branches=[]):
    for bri in branches:

        perf_fs = []
        #walk through the test set
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
            x  = x.cuda()
            yt = yt.cuda()

            with torch.no_grad():
                yf = core.forward(x, branch=bri)
            
            perf_f = evaluation.calculate(args, yf, yt)
            perf_fs.append(perf_f.cpu())

        mean_perf_f = torch.stack(perf_fs, 0).mean()

        log_str = f'[INFO] TS - BRANCH_ID: {bri} - P: {mean_perf_f:.3f}'
        print(log_str)

        torch.save(core.state_dict(), args.cv_dir + '/branch_%d_ckpt_E_%d_P_%.3f.t7' % (bri, epoch, mean_perf_f))

def test_merge(epoch, psi=0.2):
    perf_fs = []
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf = core.forward_merge_gradient_sobel(x, p=psi)
        
        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = f'[INFO] TS - MERGE - PSI: {psi} - P: {mean_perf_f:.3f}'
    print(log_str)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

core = model.config(args)
core.cuda()

sub_params = []
sub_params += core.branch[1].parameters()

optim_phase_2 = optimizer.create_optimizer(sub_params, args)
lr_scheduler_phase_2 = utils.LrScheduler(optim_phase_2, args.lr, args.lr_decay_ratio, args.epoch_step)

print('[INFO] train small branch with KD')
for epoch in range(args.start_epoch, args.max_epochs+1):
    lr_scheduler_phase_2.adjust_learning_rate(epoch)

    if epoch % 10 == 0:
        test(epoch, [0,1])
        test_merge(epoch, psi=0.2)

    train_kd(epoch, optim_phase_2)
