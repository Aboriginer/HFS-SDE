import os
import numpy as np
import mat73
import scipy.io as scio
import torch
from utils.utils import *


label = "label65.mat"
mask = "low_frequency_acs20.mat"

label = mat73.loadmat(label)["sublabel"]
label = label.astype(np.complex128)
label = np.expand_dims(label, axis=0)
label = np.expand_dims(label, axis=0)
label = torch.from_numpy(label)
label = label[:, :, :, :, 17]
k0 = fft2c_2d(label)

mask = scio.loadmat(mask)["mask"]
mask = mask.astype(np.complex128)
mask = np.expand_dims(mask, axis=0)
mask = np.expand_dims(mask, axis=0)
mask = torch.from_numpy(mask)
k0 = k0 * (1.0 - mask)
label = ifft2c_2d(k0)

save_mat(".", label, "hfs", 0, normalize=True)
