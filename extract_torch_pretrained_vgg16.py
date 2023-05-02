import os
import torch
import numpy as np

import torchvision.models as tmodels

file_name_prefix = 'vgg16_layer_'

m = tmodels.vgg16(pretrained=True)
print(m.classifier)

I = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
T = [0, 1, 3, 4,  6,  7,  8, 10, 11, 12, 14, 15, 16]
for i, t in zip(I, T):
    file_name = file_name_prefix + str(t)
    with open(file_name, 'w') as f:
        B = m.features[i].bias.data.numpy()
        print(B.shape)
        W = m.features[i].weight.data.numpy().flatten()
        print(W.shape)
        data = np.concatenate((B, W))
        data.tofile(f)

I = [ 0,  3,  6]
T = [19, 20, 21]
for i, t in zip(I, T):
    file_name = file_name_prefix + str(t)
    with open(file_name, 'w') as f:
        B = m.classifier[i].bias.data.numpy()
        print(B.shape)
        W = m.classifier[i].weight.data.numpy().flatten()
        print(W.shape)
        data = np.concatenate((B, W))
        data.tofile(f)

# i = 4
# t = 10
# file_name = file_name_prefix + str(t)
# with open(file_name, 'w') as f:
#     B = m.classifier[i].bias.data.numpy()
#     print(B.shape)
#     W = m.classifier[i].weight.data.numpy().flatten()
#     print(W.shape)
#     data = np.concatenate((B, W))
#     data.tofile(f)

# i = 6
# t = 11
# file_name = file_name_prefix + str(t)
# with open(file_name, 'w') as f:
#     B = m.classifier[i].bias.data.numpy()
#     print(B.shape)
#     W = m.classifier[i].weight.data.numpy().flatten()
#     print(W.shape)
#     data = np.concatenate((B, W))
#     data.tofile(f)