import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'FusionNet_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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

	elif args.template == 'FusionNet_2_1':
		print('[INFO] Template found (FusionNet_2 full branch trainer)')
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
		args.core='FusionNet_2'
		args.checkpoint=None

	elif args.template == 'FusionNet_3_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_3'
		# args.checkpoint=None

	elif args.template == 'FusionNet_4_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_4'
		args.checkpoint=None

	elif args.template == 'FusionNet_5_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_5'
		args.checkpoint=None

	elif args.template == 'FusionNet_6_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_6'
		args.checkpoint=None

	elif args.template == 'FusionNet_7_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_7'
		args.checkpoint=None

	elif args.template == 'FusionNet_7_gsi_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-2
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=10
		args.max_epochs=30
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=16000
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_gsi'
		# args.checkpoint=None

	elif args.template == 'FusionNet_7_gsi_mirror_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-4
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=300
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=16000
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_gsi_mirror'
		# args.checkpoint=None

	elif args.template == 'FusionNet_6_gsi_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
		args.lr=1e-2
		args.lr_decay_ratio=0.5
		args.weight_decay=0
		args.batch_size=16
		args.epoch_step=100
		args.max_epochs=300
		args.loss='L1'
		args.optimizer='Adam'
		args.max_load=16000
		args.style='Y'
		args.trainset_tag='SR291B'
		args.trainset_patch_size=21
		args.trainset_dir='/home/dataset/sr291_21x21_dn/2x/'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_6_gsi'
		# args.checkpoint=None

	elif args.template == 'FusionNet_8_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_8'
		args.checkpoint=None

	elif args.template == 'FusionNet_9_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_9'
		args.checkpoint=None

	elif args.template == 'FusionNet_7_2s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_7_2s'
		args.checkpoint=None

	elif args.template == 'FusionNet_7_3s_1':
		print('[INFO] Template found (FusionNet full branch trainer)')
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
		args.core='FusionNet_7_3s'
		args.checkpoint=None

	else:
		print('[ERRO] Template not found')
		assert(0)
