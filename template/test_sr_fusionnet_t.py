import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'FusionNet_2_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_2'

	elif args.template == 'FusionNet_6_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set5B'
		args.testset_dir='/home/dataset/set5_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_6'

	elif args.template == 'FusionNet_7_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7'

	elif args.template == 'FusionNet_7_debug_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_debug'

	elif args.template == 'FusionNet_7_2s_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set5B'
		args.testset_dir='/home/dataset/set5_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_2s'

	elif args.template == 'FusionNet_7_3s_1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set5B'
		args.testset_dir='/home/dataset/set5_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='FusionNet_7_3s'

	else:
		print('[ERRO] Template not found')
		assert(0)
