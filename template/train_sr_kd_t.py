import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'IDAG_M1P_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M1P'
		args.checkpoint=None

	elif args.template == 'IDAG_M1_3':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M1'
		args.checkpoint=None

	elif args.template == 'SVDSR-10-64_2':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='SVDSR-10-64'
		args.checkpoint=None

	elif args.template == 'IDAG_M3_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M3'
		args.checkpoint=None

	elif args.template == 'IDAG_M3_KD_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M3_KD'
		args.checkpoint=None
		args.kd_teacher_core = 'IDAG_M3'
		args.kd_teacher_checkpoint = 'backup/IDAG_M3_adam/ckpt_E_270_P_33.106.t7'

	elif args.template == 'IDAG_M3_KD2_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M3_KD2'
		args.checkpoint=None
		args.kd_teacher_core = 'IDAG_M3'
		args.kd_teacher_checkpoint = 'backup/IDAG_M3_adam/ckpt_E_270_P_33.106.t7'

	elif args.template == 'IDAG_M3_KD3_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M3_KD3'
		args.checkpoint=None
		args.kd_teacher_core = 'IDAG_M3'
		args.kd_teacher_checkpoint = 'backup/IDAG_M3_adam/ckpt_E_270_P_33.106.t7'

	elif args.template == 'IDAG_M3_KD3s_1':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M3_KD3s'
		args.checkpoint=None
		args.kd_teacher_core = 'IDAG_M3'
		args.kd_teacher_checkpoint = 'backup/IDAG_M3_adam/ckpt_E_270_P_33.106.t7'

	elif args.template == 'IDAG_M2_adam':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=300
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
		args.core='IDAG_M2'
		args.checkpoint=None

	elif args.template == 'SMSR_1':
		print('[INFO] Template found')
		args.lr=2e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=200
		args.max_epochs=1000
		args.loss='L1'
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
		args.core='SMSR-64'
		args.checkpoint=None

	else:
		print('[ERRO] Template not found')
		assert(0)
