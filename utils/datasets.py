import os
import sys
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *
import pickle


class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIKneeDataSet, self).__init__()
        self.config = config
        input_pkl = "data/data_slice.pkl"
        if mode == "training":
            self.kspace_dir = "data/photom/kspace/"
            self.maps_dir = "data/photom/map/"

        elif mode == "test":
            self.kspace_dir = (
                "/data0/chentao/data/fastMRI/T1_knee_1000/fastMRI_knee_test/T1_data/"
            )
            self.maps_dir = "/data0/chentao/data/fastMRI/T1_knee_1000/fastMRI_knee_test/output_maps/"

        elif mode == "sample":
            self.kspace_dir = (
                "/data0/chentao/data/fastMRI/T1_knee_1000/fastMRI_knee_sample/T1_data/"
            )
            self.maps_dir = "/data0/chentao/data/fastMRI/T1_knee_1000/fastMRI_knee_sample/output_maps/"
        elif mode == "photom":
            self.kspace_dir = "data/photom/kspace/"
            self.maps_dir = "data/photom/map/"
        elif mode == "datashift":
            self.kspace_dir = (
                "/data1/chentao/data/fastMRI/T1_knee_1000/fastMRI_brain/brain_T2/"
            )
            self.maps_dir = (
                "/data1/chentao/data/fastMRI/T1_knee_1000/fastMRI_brain/output_maps/"
            )
        else:
            raise NotImplementedError

        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir)
        print(len(self.file_list))

        self.num_slices = np.zeros(
            (
                len(
                    self.file_list,
                )
            ),
            dtype=int,
        )

        with open(input_pkl, "rb") as pkl_file:
            data_dict = pickle.load(pkl_file)

        for idx, file in enumerate(self.file_list):
            temp_path = os.path.join(self.kspace_dir, os.path.basename(file))
            print("Input file:", temp_path)
            # Exclude the first 6 frames
            if self.mode != "sample" and self.mode != "photom":
                self.num_slices[idx] = int(
                    max(data_dict[os.path.basename(file)] - 6, 1)
                )
            else:
                self.num_slices[idx] = int(data_dict[os.path.basename(file)])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Subject number
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Frame number of the subject's scan
        slice_idx = (
            int(idx)
            if scan_idx == 0
            else int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        )

        # Load maps for specific scan and slice
        maps_file = os.path.join(
            self.maps_dir, os.path.basename(self.file_list[scan_idx])
        )
        with h5py.File(maps_file, "r") as data:
            # Exclude the first 6 frames
            if self.mode != "sample" and self.mode != "p":
                slice_idx = min(int(np.sum(self.num_slices)) - 1, slice_idx + 6)

            maps_idx = data["s_maps"][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = crop(maps_idx, cropx=320, cropy=320)
            maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(
            self.kspace_dir, os.path.basename(self.file_list[scan_idx])
        )
        with h5py.File(raw_file, "r") as data:
            ksp_idx = data["kspace"][slice_idx]  # 15x640x368
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(IFFT2c(ksp_idx), cropx=320, cropy=320)
            ksp_idx = FFT2c(ksp_idx)
            ksp_idx = np.squeeze(ksp_idx, 0)
            if self.config.data.normalize_type == "minmax":
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
            elif self.config.data.normalize_type == "std":
                minv = np.std(ksp_idx)
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)
            elif self.config.data.normalize_type == "img_std":
                ksp_idx = np.expand_dims(ksp_idx, 0)
                ksp_idx = IFFT2c(ksp_idx)
                ksp_idx = normalize_l2(ksp_idx)
                ksp_idx = FFT2c(ksp_idx)
                ksp_idx = np.squeeze(ksp_idx, 0)

            kspace = np.asarray(ksp_idx)

        return kspace, maps

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == "fastMRI_knee":
        dataset = FastMRIKneeDataSet(config, mode)

    if mode == "training":
        data = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )
        # data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        from utils.utils import worker_init_fn

        data = DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            num_workers=0,
        )

    print(mode, "data loaded")

    return data
