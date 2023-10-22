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
from template import train_sr_fusionnet_t as template

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser.add_argument("--skip-C", action= "store_true", help="skip training C phase")

args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def write_to_file(text, file, mode='a'):
    with open(file, mode) as f:
        f.write(f"{text}\n")
        
save_path = os.path.join(args.cv_dir, 'sparsity.txt')
write_to_file("Loss sparsity", save_path, 'w')

def train(epoch, optim):
    core.train()
    perfs = []
    total_loss = 0
    for batch_ids, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        x = x.cuda()
        yt = yt.cuda()

        yf, sparsity = core.forward(x)
        perf = evaluation.calculate(args, yf, yt)

        loss_func = loss.create_loss_func(args.loss)

        loss_SR = loss_func(yf, yt)
        loss_sparsity = sparsity.mean() # we try to reduce the sparsity to perseve information from features
        lambda0 = 0.0
        # lambda_sparsity = min((epoch - 1) / 50, 1) * lambda0
        # lambda_sparsity = 0.0 if epoch < 40 else -0.001
        lambda_sparsity = 0.0005
        batch_loss = loss_SR + lambda_sparsity * loss_sparsity
        
        optim.zero_grad()
        batch_loss.backward()

        optim.step()
        total_loss += batch_loss

        perfs.append(perf.cpu())

    perf = torch.stack(perfs, 0).mean()

    write_to_file(f"Epoch {epoch}: Loss sparsity {loss_sparsity}", save_path, 'a')
    log_str = '[INFO] E: %d | P: %.3f | LOSS: %.3f' % (epoch, perf, total_loss)
    print(log_str)

def test(epoch): # in train mode

    perf_fs = []
    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf, _ = core.forward(x)
        
        perf_f = evaluation.calculate(args, yf, yt)
        perf_fs.append(perf_f.cpu())

    mean_perf_f = torch.stack(perf_fs, 0).mean()

    log_str = f'[INFO] TEST - HourGlass - P: {mean_perf_f:.3f}'
    print(log_str)

    torch.save(core.state_dict(), args.cv_dir + '/hourglass_ckpt_E_%d_P_%.3f.t7' % (epoch, mean_perf_f))
    torch.save(core.state_dict(), args.cv_dir + "/_latest.t7")


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

# train only the C branch
all_params = []
all_params += core.body.parameters()
all_params += core.head.parameters()
all_params += core.tail.parameters()

optim_phase_1 = optimizer.create_optimizer(all_params, args)
lr_scheduler_phase_1 = utils.LrScheduler(optim_phase_1, args.lr, args.lr_decay_ratio, args.epoch_step)

print('[INFO] train hourglass')
print(f'[INFO] train large branch for {args.max_epochs}')
for epoch in range(args.start_epoch, args.max_epochs+1):
    lr_scheduler_phase_1.adjust_learning_rate(epoch)

    if epoch % 10 == 0:
        test(epoch)

    train(epoch, optim_phase_1)

