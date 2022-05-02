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
from template import test_sr_mask_t as template

if args.template is not None:
    template.set_template(args)

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
            yq, _, new_fmap, sp, spl = core(x) #forward image through core with masks applied to calculate psnr of the solution
            core.detach_mask()
        
        perf_f = evaluation.calculate(args, yf, yt) #calculate baseline psnr here, of the core network without mask
        perf_m = evaluation.calculate(args, yq, yt) #calculate psnr of the core network when masks applied
        print(perf_m)

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

### main
######################################################################################################################### data
# load test data
XYtests = {}
batch_size_tests = {}

testset_tags = args.testset_tag.split(',')
testset_dirs = args.testset_dir.split(',')
eval_tags = args.eval_tag.split(',')
for testset_tag, testset_dir in zip(testset_tags, testset_dirs):
    args.testset_tag = testset_tag
    args.testset_dir = testset_dir
    testset, batch_size_tests[testset_tag] = data.load_testset(args)
    XYtests[testset_tag] = torchdata.DataLoader(testset, batch_size=batch_size_tests[testset_tag], shuffle=False, num_workers=16)

######################################################################################################################### model
# create and load core model
args.target_layer_index = [int(i) for i in args.target_layer_index.split(',')]
args.target_layer_index.sort()

core = mcore.config(args)
print('[INFO] policy network generates masks for layers: ', (core.target_layer_index))

# create and load policy model
agent = magent.config(args, core)

# cuda config
core.eval().cuda()
agent.cuda()

for l in agent.target_layer_index:
    agent.agent_list[l].cuda()

for testset_tag in testset_tags:
    print('[INFO] testset: ' + testset_tag)
    for eval_tag in eval_tags:
        print('[INFO] metric: ' + eval_tag)
        XYtest = XYtests[testset_tag]
        test(0)