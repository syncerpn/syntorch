import os
import torch
import numpy as np

import torchvision.models as tmodels

file_name_prefix = 'alexnet_layer_'

m = tmodels.vgg13(pretrained=True)
print(m.features)

# I = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]
# T = [0]
# for i, t in zip(I, T)
# file_name = file_name_prefix + str(t)
# with open(file_name, 'w') as f:
#     B = m.features[i].bias.data.numpy()
#     print(B.shape)
#     W = m.features[i].weight.data.numpy().flatten()
#     print(W.shape)
#     data = np.concatenate((B, W))
#     data.tofile(f)

# i = 1
# t = 9
# file_name = file_name_prefix + str(t)
# with open(file_name, 'w') as f:
#     B = m.classifier[i].bias.data.numpy()
#     print(B.shape)
#     W = m.classifier[i].weight.data.numpy().flatten()
#     print(W.shape)
#     data = np.concatenate((B, W))
#     data.tofile(f)

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