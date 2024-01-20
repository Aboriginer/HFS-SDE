import os
import numpy as np
import mat73
import scipy.io as scio
import torch
from utils.utils import *


file = "sense_head_uniform_acc8.mat"

recon = scio.loadmat(file)["recon"]
recon = recon.astype(np.complex128)
recon = torch.from_numpy(recon)
save_mat(".", recon, "recon", 0, normalize=True)
