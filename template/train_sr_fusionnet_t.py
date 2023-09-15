import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'outputs_' + args.template + '_' + timestamp + '/'

	if   args.template == 'FusionNet_7_1s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=150
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_1s'

	elif args.template == 'FusionNet_7_2s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=150
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_2s'

	elif args.template == 'FusionNet_7_3s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=150
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_3s'

	elif args.template == 'FusionNet_7_4s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=150
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_4s'

	elif args.template == 'FusionSM_7_4s_v2':
		print('[INFO] Template found (FusionSM full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=1024
		args.epoch_step=100
		args.max_epochs=50
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.testset_tag='Set14B'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionSM_7_4s_v2'
		args.cv_dir = "model_checkpoints"

	else:
		print('[ERRO] Template not found')
		assert(0)
