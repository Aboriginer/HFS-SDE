from bart import bart
import numpy as np
import h5py
from utils.utils import *  # Importing custom utility functions
import sys

sys.path.insert(0, "../")  # Adding a directory to the sys path


# Function to crop an image
def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty : starty + cropy, startx : startx + cropx]


# Path to k-space data
data_path = "data/photom/kspace/1.h5"

# Path to save the maps
save_path = "data/photom/map/1.h5"

with h5py.File(data_path, "r") as hf:
    kspace_data = hf["kspace"][:]  # Read k-space data (shape: slice x nc x kx x ky)

# Make sure that the k-space shape is (slice, nc, kx, ky)
print("data_size:", kspace_data.shape)

kspace_data = IFFT2c(kspace_data)  # Perform Inverse Fast Fourier Transform (IFFT2c)

# Crop to 320x320
kspace_data = crop(kspace_data, cropx=320, cropy=320)

kspace_data = FFT2c(kspace_data)  # Perform Fast Fourier Transform (FFT2c)

data_shape = list(kspace_data.shape)

smaps = np.zeros(data_shape, dtype=np.complex64)

kspace_data = kspace_data.transpose(0, 2, 3, 1)

# Make sure that the k-space shape is (slice, kx, ky, nc)
# Estimate coil sensitivity map
for i in range(kspace_data.shape[0]):
    # input shape of bart: [slice, kx, ky, nc]
    smaps_bart = bart(1, "ecalib -m 1 -S", np.ascontiguousarray(kspace_data[i:i+1, :, :, :]))
    smaps[i] = np.ascontiguousarray(
        smaps_bart[0].transpose(2, 0, 1).astype(np.complex64)
    )

# Save the coil sensitivity map
hf = h5py.File(save_path, mode="w")
hf.create_dataset("s_maps", data=smaps)
hf.close()
