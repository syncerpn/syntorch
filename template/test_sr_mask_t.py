import time

def set_template(args):
	timestamp = str(int(time.time()))
	if args.template is not None:
		args.cv_dir = 'backup/' + args.template + '_' + timestamp + '/'

	if args.template == 'IDAG_M3_parasitic_v0':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v0-4'
		args.mask_type='sigmoid'

	elif args.template == 'IDAG_M3_parasitic_v3':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v3-4'
		args.mask_type='sigmoid'

	elif args.template == 'IDAG_M3_parasitic_v1':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='IDAG_M3'
		args.target_layer_index='1,2,3,4,5,6'
		args.agent='IDAG_M3_parasitic_v1-4'
		args.mask_type='sigmoid'

	elif args.template == 'SVDSR_10_parasitic_v0-4':
		print('[INFO] Template found')
		args.style='Y'
		args.testset_tag='Set14B'
		args.testset_dir='/home/dataset/set14_dnb/2x/'
		args.rgb_range=1.0
		args.scale=2
		args.core='SVDSR-10-64'
		args.target_layer_index='0,1,2,3,4,5,6,7,8'
		args.agent='SVDSR_parasitic_v0-4'
		args.mask_type='sigmoid'

	else:
		print('[ERRO] Template not found')
		assert(0)
