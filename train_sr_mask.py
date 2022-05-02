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
import model.mask.core as mcore
import model.mask.agent as magent
import optimizer
import utils
from option import args
from template import train_sr_mask_t as template

if args.template is not None:
    template.set_template(args)

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

### def
def train(epoch):
    #turn on train mode on agent
    agent.train()
    sps, perfs = [], []

    #walk through the train set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
        #get input batch and push to GPU
        x  = x.cuda()
        yt = yt.cuda()
        
        with torch.no_grad():
            yf, agent_input_fmap, agent_gt_fmap, _, _ = core(x) #forward input through core to get feature maps, which will be used as ground truth masks later

        #forward input through agent
        agent_fmap = agent(agent_input_fmap)
        
        masks = agent_fmap

        #forward same input through core
        with torch.no_grad():
            core.attach_mask(masks, method='ready') #activate masks-ready mode
            yq, _, _, sp, _ = core(x) #forward input through core with masks applied on the output of the target layers; return the HR image for calculating PSNR and sparsity
            core.detach_mask()

        #calculate psnr when masks are applied
        perf = evaluation.calculate(args, yq, yt)

        for i in core.target_layer_index:
            agent_fmap[i] = agent_fmap[i].clamp(args.clamp, 1-args.clamp)

        loss = 0
        losses = {}
        for i in core.target_layer_index:
            gt_ones  = (agent_gt_fmap[i] != 0).float()
            num_pixel = agent_gt_fmap[i].numel()

            if args.zero_one_balance:
                losses[i] = args.wone * -(1.0-gt_ones.sum()/num_pixel) * gt_ones * torch.log(agent_fmap[i]) + (1-args.wone) * -(gt_ones.sum()/num_pixel) * (1.0-gt_ones) * torch.log(1.0 - agent_fmap[i])
            else:
                losses[i] = args.wone *                                 -gt_ones * torch.log(agent_fmap[i]) + (1-args.wone) *                             -(1.0-gt_ones) * torch.log(1.0 - agent_fmap[i])
            
            loss += losses[i].sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        sps.append(sp.data.cpu())
        perfs.append(perf.cpu())

    #performance stats
    perf = torch.stack(perfs, 0).mean()

    sps = torch.cat(sps, 0)
    mean_sp = sps.mean()
    var_sp = sps.std()

    log_str = '[INFO] E: %d | P: %.3f | SPM: %.3f | SPV: %.3f | LOSS: %.3f' % (epoch, perf, mean_sp, var_sp, loss)
    print(log_str)

def test(epoch):
    #turn on eval mode on agent
    agent.eval()

    perf_fs, perf_ms, sps = [], [], []
    spls_v = {}
    spls_c = {}
    spls_g = {}

    for i in core.target_layer_index:
        spls_v[i] = 0
        spls_c[i] = 0
        spls_g[i] = 0

    #walk through the test set
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            yf, _, gt_fmap, _, _ = core.forward(x) #forward image through core to calculate reference/baseline psnr later
            
            core.attach_mask(agent, method='generator') #activate parasitic forwarding mode
            yq, _, _, sp, spl = core(x) #forward image through core with masks applied to calculate psnr of the solution
            core.detach_mask()
        
        perf_f = evaluation.calculate(args, yf, yt) #calculate baseline psnr here, of the core network without mask
        perf_m = evaluation.calculate(args, yq, yt) #calculate psnr of the core network when masks applied

        #keep track of psnr to compare later
        perf_fs.append(perf_f.cpu())
        perf_ms.append(perf_m.cpu())
        sps.append(sp.data.cpu())
        for i in core.target_layer_index:
            spls_v[i] += torch.sum(spl[i], dim=0)
            spls_c[i] += spl[i].shape[0]
            spls_g[i] += torch.mean((gt_fmap[i] == 0).float())

    #performance stats
    mean_perf_m = torch.stack(perf_ms, 0).mean()
    mean_perf_f = torch.stack(perf_fs, 0).mean()
    perf_degrade = mean_perf_m - mean_perf_f
    sps = torch.stack(sps, 0)
    mean_sp = sps.mean()
    var_sp = sps.std()

    log_str = '[INFO] TS - P: %.4f | D: %.4f | SPM: %.3f | SPV: %.3f' % (mean_perf_m, perf_degrade, mean_sp, var_sp)
    print(log_str)

    for i in core.target_layer_index:
        print('[INFO] SP layer %2d: %.3f / %.3f (%3.1f)' % (i, spls_v[i] / spls_c[i], spls_g[i] / len(XYtest), (spls_v[i] / spls_c[i]) * 100 / (spls_g[i] / len(XYtest))))

    state = {l: agent.agent_list[l].state_dict() for l in agent.target_layer_index}
    torch.save(state, args.cv_dir + '/ckpt_E_%d_P_%.3f_SP_%.3f.t7' % (epoch, mean_perf_m, mean_sp))

### main
######################################################################################################################### data
# load training data
print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=16)

######################################################################################################################### model
# create and load core model
args.target_layer_index = [int(i) for i in args.target_layer_index.split(',')]
args.target_layer_index.sort()

core = mcore.config(args)
print('[INFO] policy network generates masks for layers: ', (core.target_layer_index))

if args.zero_one_balance:
    print('[INFO] zero-one loss balancing is used')

# create and load policy model
agent = magent.config(args, core)

# cuda config
core.eval().cuda()
agent.cuda()

for l in agent.target_layer_index:
    agent.agent_list[l].cuda()

######################################################################################################################### optimizer
all_params = agent.parameters()

optim = optimizer.create_optimizer(all_params, args)
lr_scheduler = utils.LrScheduler(optim, args.lr, args.lr_decay_ratio, args.epoch_step)

for epoch in range(args.start_epoch, args.max_epochs + 1):
    lr_scheduler.adjust_learning_rate(epoch)

    if epoch % 10 == 0:
        test(epoch)

    train(epoch)