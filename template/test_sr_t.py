import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'IDAG_M1P_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M1P'

	elif args.template == 'IDAG_M1_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M1'

	elif args.template == 'SVDSR-10-64_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='SVDSR-10-64'

	elif args.template == 'IDAG_M2_Set5B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set5B'
		args.testset_dir='/home/dataset/set5_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M2'

	elif args.template == 'IDAG_M2_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M2'

	elif args.template == 'IDAG_M3_Set5B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set5B'
		args.testset_dir='/home/dataset/set5_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3'

	elif args.template == 'IDAG_M3_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3'

	elif args.template == 'IDAG_M3_KD3_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3_KD3'

	elif args.template == 'IDAG_M4_Set14B':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M4'

	elif args.template == 'SMSR_Set14':
		print('[INFO] Template found')
		args.style='RGB'
		args.testset_tag='SetN'
		args.testset_dir='/home/dataset/Set14/'
		args.rgb_range=255.0
		args.scale=2
		args.core='SMSR-64'

	elif args.template == 'SMSR_Set5':
		print('[INFO] Template found')
		args.style='RGB'
		args.testset_tag='SetN'
		args.testset_dir='/home/dataset/Set5/'
		args.rgb_range=255.0
		args.scale=2
		args.core='SMSR-64'

	else:
		print('[ERRO] Template not found')
		assert(0)
