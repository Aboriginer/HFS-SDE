# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training DDPM with VP SDE."""

from configs.default_fastMRI_configs import get_default_configs


def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = "hfssde"
    training.continuous = True
    training.reduce_mean = True
    training.mask_type = "low_frequency"  # low_frequency, uniform, center
    training.acc = "None"  # center low None
    training.acs = "16"
    training.mean_equal = "noequal"  # equal or noequal

    # sampling
    sampling = config.sampling
    sampling.batch_size = 1
    sampling.method = "pc"
    sampling.predictor = "euler_maruyama"  # reverse_diffusion or euler_maruyama
    sampling.corrector = "langevin"  # langevin or none
    sampling.folder = "2023_10_07T20_39_52_ddpm_hfssde_low_frequency_16_std_20.0_N_1000"

    sampling.ckpt = 190
    sampling.mask_type = "uniform"  # uniform, cartesian, random_uniform or center
    sampling.acc = "10"
    sampling.acs = "24"
    sampling.fft = "nofft"  # fft or nofft

    sampling.accelerated_sampling = False
    if training.sde == "hfssde":
        sampling.snr = 0.16
        if sampling.accelerated_sampling:
            sampling.snr = 0.32
    elif training.sde == "vpsde":
        sampling.snr = 0.27  # 0.27
        if sampling.accelerated_sampling:
            sampling.snr = 0.43
    if sampling.accelerated_sampling:
        config.sampling.N = 100

    sampling.mse = 2.5
    sampling.corrector_mse = 0.1  ###
    sampling.datashift = "photom"  ### head or knee photom

    # data
    data = config.data
    data.centered = False  # True: Input is in [-1, 1]
    data.dataset_name = "fastMRI_knee"
    data.image_size = 320
    data.normalize_type = "std"  # minmax or std or img_std
    data.normalize_coeff = 1.5  # normalize coefficient

    # model
    model = config.model
    model.name = "ddpm"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True

    return config
