import argparse
parser = argparse.ArgumentParser(description='Image Super-Resolution Trainer (clean)', fromfile_prefix_chars='@')

#training hyper-param
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--weight_decay',type=float, default=1e-4, help="Weight decay, Default: 1e-4")

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epoch_step', type=int, default=20, help='epochs after which lr is decayed')
parser.add_argument('--start_epoch', type=int, default=0, help='starting point')
parser.add_argument('--max_epochs', type=int, default=80, help='total epochs to run')
parser.add_argument('--loss', default='L2', help='loss function')

parser.add_argument('--optimizer', default='SGD', help='optimizer')
#--sgd
parser.add_argument('--momentum', type=float, default=0.9, help='learning rate')
#--adam

#gradnorm
parser.add_argument('--gradnorm', type=float, default=0, help='local gradient normalization')

#data
parser.add_argument('--max_load', default=0, help='max number of samples to use; useful for reducing loading time during debugging; 0 = load all')
parser.add_argument('--style', default='Y', help='Y-channel or RGB style')
parser.add_argument('--trainset_tag', default='SR291B', help='train data directory')
parser.add_argument('--trainset_patch_size', type=int, default=21, help='train data directory')
parser.add_argument('--trainset_preload', type=int, default=0, help='train data directory')
parser.add_argument('--trainset_dir', default='/home/dataset/sr291_21x21_dn/2x/', help='train data directory')
parser.add_argument('--testset_tag', default='Set14B', help='train data directory')
parser.add_argument('--testset_dir', default='/home/dataset/set14_dnb/2x/', help='test data directory')

#model
parser.add_argument('--rgb_range', type=float, default=1.0, help='int/float images')
parser.add_argument('--scale', type=int, default=2, help='scaling factor')
parser.add_argument('--core', default='SMSR_normal', help='core model (template specified in sr_mask_core.py)')
parser.add_argument('--checkpoint', default=None, help='checkpoint to load core from')

#eval
parser.add_argument('--eval_tag', default='psnr', help='evaluation tag; available: "psnr", "accuracy"')

#output
parser.add_argument('--cv_dir', default='backup/nobias_test/', help='checkpoint directory (models and logs are saved here)')

#mask training option
parser.add_argument('--zero-one-balance', type=bool, default=True, help='auto balancing zero and one loss')
parser.add_argument('--wone', type=float, default=0.80, help='weighted learner')
parser.add_argument('--clamp', type=float, default=1e-4, help='cross-entropy clipping bound for safe training')

parser.add_argument('--target_layer_index', default=None, help='target layer fRor mask generation')
parser.add_argument('--agent', default='IDAG_M3_parasitic_v0-4', help='agent model (template specified in sr_mask_agent.py)')
parser.add_argument('--agent_checkpoint', default=None, help='checkpoint to load agent from')
parser.add_argument('--mask_type', default='sigmoid', help='checkpoint to load agent from')

#template
parser.add_argument('--template', default=None)

args = parser.parse_args()