import torch

def transform(z, mask_type='sigmoid'):
    if mask_type=='sigmoid':
        m = z.data.clone()
        m[z >  0.5] = 1.0
        m[z <= 0.5] = 0.0
        return m
    else:
        print('[ERRO] Unknown mask type')
        assert(0)
