import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'IDAG_M3_parasitic_v0':
		print('[INFO] Template found')
		args.lr=1e-2
		args.lr_decay_ratio=0.1
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
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
		args.zero_one_balance=True
		args.wone=0.8
		args.clamp=1e-4
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v0-4'
		args.mask_type='sigmoid'

	elif args.template == 'IDAG_M3_parasitic_v0_s1':
		print('[INFO] Template found')
		args.lr=1e-2
		args.lr_decay_ratio=0.1
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
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
		args.zero_one_balance=False
		args.wone=0.95
		args.clamp=1e-4
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v0-4'
		args.mask_type='sigmoid'

	elif args.template == 'IDAG_M3_parasitic_v3':
		print('[INFO] Template found')
		args.lr=1e-2
		args.lr_decay_ratio=0.1
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
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
		args.zero_one_balance=True
		args.wone=0.8
		args.clamp=1e-4
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v3-4'
		args.mask_type='sigmoid'

	elif args.template == 'IDAG_M3_parasitic_v1':
		print('[INFO] Template found')
		args.lr=1e-2
		args.lr_decay_ratio=0.1
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
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
		args.zero_one_balance=False
		args.wone=0.95
		args.clamp=1e-4
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v1-4'
		args.mask_type='sigmoid'

	elif args.template == 'SMSR_parasitic_v0':
		print('[INFO] Template found')
		args.lr=1e-2
		args.lr_decay_ratio=0.1
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
		args.optimizer='Adam'
		args.max_load=0
		args.style='RGB'
		args.trainset_tag='DIV2K'
		args.trainset_patch_size=96
		args.trainset_dir='/home/dataset/DIV2K/'
		args.trainset_preload=700
		args.testset_tag='SetN'
		args.testset_dir='/home/dataset/Set14/'
		args.rgb_range=255
		args.scale=2
		args.zero_one_balance=True
		args.wone=0.8
		args.clamp=1e-4
		args.core='SMSR-64'
		args.target_layer_index='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
		args.agent='SMSR_parasitic_v0-4'
		args.mask_type='sigmoid'

	else:
		print('[ERRO] Template not found')
		assert(0)
