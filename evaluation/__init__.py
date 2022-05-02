from evaluation.psnr import calculate_psnr as psnr
from evaluation.ssim import calculate_ssim as ssim

def calculate(args, y, yt):
    if args.eval_tag == 'psnr':
        return psnr(y, yt, args.scale, args.rgb_range)
    elif args.eval_tag == 'ssim':
        return ssim(y, yt, args.scale)
    else:
        print('[ERRO] unknown evaluation tag')
        assert(0)

print('[ OK ] Module "evaluation"')