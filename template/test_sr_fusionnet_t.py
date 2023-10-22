import time

def set_template(args):
	if args.template is not None:
		if args.template == "FusionNet_7_1s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNet_7_1s"

		elif args.template == "FusionNet_7_2s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNet_7_2s"

		elif args.template == "FusionNet_7_3s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNet_7_3s"

		elif args.template == "FusionNet_7_4s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNet_7_4s"

		if args.template == "FusionSM_7_4s":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionSM_7_4s"

		if args.template == "FusionSM_7_4s_v2":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionSM_7_4s_v2"
   
		if args.template == "FusionSM_7_4s_v2_Kaggle":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir='/kaggle/input/fn-data-and-cktpt/data_ckpt/dataset/set14_dnb/set14_dnb/2x/'
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionSM_7_4s_v2"
			args.checkpoint='/kaggle/working/syntorch/model_checkpoints/_latest.t7'
   
		if args.template == "FusionSM_7_4s_v2_Kaggle_test":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir='/kaggle/input/fn-data-and-cktpt/data_ckpt/dataset/set14_dnb/set14_dnb/2x/'
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionSM_7_4s_v2_test"
			# args.checkpoint='/kaggle/working/syntorch/model_checkpoints/_latest.t7'
			args.checkpoint='/kaggle/working/syntorch/trained_store/64/_latest.t7'
   
		if args.template == "Hourglass_Kaggle":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir='/kaggle/input/fn-data-and-cktpt/data_ckpt/dataset/set14_dnb/set14_dnb/2x/'
			args.rgb_range=1.0
			args.scale=2
			args.core="HourglassResidual"
			args.checkpoint='/kaggle/working/syntorch/trained_store/hourglass/_latest.t7'

		else:
			assert 0, f"[ERRO] Template not found {args.template}"