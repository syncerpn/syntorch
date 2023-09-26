import time

def set_template(args):
	if args.template is not None:
		if args.template == "MaskNetB_7_4s_1":
			print(f"[INFO] Template found: {args.template}")
			args.style="Y"
			args.testset_tag="Set14B"
			args.testset_dir="/home/dataset/set14_dnb/2x/"
			args.rgb_range=1.0
			args.scale=2
			args.core="MaskNetB_7_4s"

		else:
			assert 0, f"[ERRO] Template not found {args.template}"