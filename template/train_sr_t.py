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

	elif args.template == 'VarNet_1':
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
		args.core='VarNet'
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

	elif args.template == 'IDAG_M1_l32_1':
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
		args.core='IDAG_M1_l32'
		args.checkpoint=None

	elif args.template == 'IDAG_M1_l64_1':
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
		args.core='IDAG_M1_l64'
		args.checkpoint=None

	elif args.template == 'IDAG_M1_r3_1':
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
		args.core='IDAG_M1_r3'
		args.checkpoint=None

	elif args.template == 'IDAG_M1_c3_1':
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
		args.core='IDAG_M1_c3'
		args.checkpoint=None

	elif args.template == 'IDAG_M6_4': #high weight decay
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0.000001
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
		args.core='IDAG_M6'
		args.checkpoint=None

	elif args.template == 'IDAG_M5_1':
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
		args.core='IDAG_M5'
		args.checkpoint=None

	elif args.template == 'IDAG_M5_m16_1':
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
		args.core='IDAG_M5_m16'
		args.checkpoint=None

	elif args.template == 'IDAG_M4_1':
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
		args.core='IDAG_M4'
		args.checkpoint=None

	elif args.template == 'IDAG_M4_2':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=2000
		args.max_epochs=3000
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='DIV2K'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/DIV2K/'
		args.trainset_preload=800
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M4'
		args.checkpoint=None

	elif args.template == 'IDAG_M4_3':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=2000
		args.max_epochs=3000
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=0
		args.style='Y'
		args.trainset_tag='DIV2K'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/DIV2K/'
		args.trainset_preload=800
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M4'
		# args.checkpoint=None

	elif args.template == 'IDAG_M4_4': #high weight decay
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0.000001
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
		args.core='IDAG_M4'
		# args.checkpoint=None

	elif args.template == 'IDAG_M6_r3_4': #high weight decay
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0.000001
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
		args.core='IDAG_M6_r3'
		args.checkpoint=None

	elif args.template == 'IDAG_M3_gradnorm_1':
		print('[INFO] Template found')
		args.lr=1e-1
		args.lr_decay_ratio=0.1
		args.weight_decay=0
		args.batch_size=128
		args.epoch_step=20
		args.max_epochs=80
		args.loss='L2'
		args.optimizer='SGD'
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
		args.gradnorm=0.1

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

	elif args.template == 'IDAG_M3_g4_4':
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0.000001
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
		args.core='IDAG_M3_g4'
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

	elif args.template == 'IDAG_M2_4': #high weight decay
		print('[INFO] Template found')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0.000001
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

	elif args.template == 'FusionNet_1':
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
		args.core='FusionNet'
		args.checkpoint=None

	else:
		print('[ERRO] Template not found')
		assert(0)
