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

		elif args.template == "FusionNetB_7_1s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_7_1s"

		elif args.template == "FusionNetB_7_2s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_7_2s"

		elif args.template == "FusionNetB_7_3s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_7_3s"

		elif args.template == "FusionNetB_7_4s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_7_4s"

		elif args.template == "FusionNetB_8_1s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_8_1s"

		elif args.template == "FusionNetB_8_2s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_8_2s"

		elif args.template == "FusionNetB_8_3s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_8_3s"

		elif args.template == "FusionNetB_8_4s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="FusionNetB_8_4s"

		else:
			assert 0, f"[ERRO] Template not found {args.template}"