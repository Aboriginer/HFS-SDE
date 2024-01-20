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
"""Training DDPM with VE SDE."""

from configs.default_fastMRI_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.reduce_mean = True
    training.mask_type = "center"  # low_frequency, uniform, center
    training.acc = "None"
    training.acs = "24"
    training.mean_equal = "noequal"  # equal or noequal

    # sampling
    sampling = config.sampling
    sampling.batch_size = 1
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.folder = "2023_10_09T12_12_44_ddpm_vesde_N_1000"
    sampling.ckpt = 190
    sampling.mask_type = "uniform"  # uniform, random_uniform or center
    sampling.acc = "10"
    sampling.acs = "24"
    sampling.fft = "nofft"  # fft or nofft

    sampling.accelerated_sampling = False
    sampling.snr = 0.1
    sampling.mse = 5.0
    sampling.corrector_mse = 5.0
    sampling.datashift = "photom"  ### head or knee photom

    if sampling.accelerated_sampling:
        config.sampling.N = 300

    # data
    data = config.data
    data.centered = False  # True: Input is in [-1, 1]
    data.dataset_name = "fastMRI_knee"
    data.image_size = 320
    data.normalize_type = "std"  # minmax or std
    data.normalize_coeff = 1.5  # normalize coefficient

    # model
    model = config.model
    model.name = "ddpm"
    model.dropout = 0.0
    model.sigma_max = 348
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True

    return config
