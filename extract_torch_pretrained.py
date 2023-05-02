import os
import torch
import numpy as np

import torchvision.models as tmodels

file_name_prefix = 'alexnet_layer_'

m = tmodels.alexnet(pretrained=True)
print(m.features)

i = 0
t = 0
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.features[i].bias.data.numpy()
    print(B.shape)
    W = m.features[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 3
t = 2
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.features[i].bias.data.numpy()
    print(B.shape)
    W = m.features[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 6
t = 4
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.features[i].bias.data.numpy()
    print(B.shape)
    W = m.features[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 8
t = 5
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.features[i].bias.data.numpy()
    print(B.shape)
    W = m.features[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 10
t = 6
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.features[i].bias.data.numpy()
    print(B.shape)
    W = m.features[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 1
t = 9
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.classifier[i].bias.data.numpy()
    print(B.shape)
    W = m.classifier[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 4
t = 10
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.classifier[i].bias.data.numpy()
    print(B.shape)
    W = m.classifier[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)

i = 6
t = 11
file_name = file_name_prefix + str(t)
with open(file_name, 'w') as f:
    B = m.classifier[i].bias.data.numpy()
    print(B.shape)
    W = m.classifier[i].weight.data.numpy().flatten()
    print(W.shape)
    data = np.concatenate((B, W))
    data.tofile(f)