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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import torch
import numpy as np
import abc
from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import model_utils as mutils
from utils.utils import *
from pathlib import Path
import cv2

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, atb_mask, train_mask):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            config=config,
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            corrector_mse=config.sampling.corrector_mse,
            sampling_fft=config.sampling.fft,
            atb_mask=atb_mask,
            train_mask=train_mask,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, atb_mask, train_mask, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn
        self.atb_mask = atb_mask
        self.train_mask = train_mask

    @abc.abstractmethod
    def update_fn(self, x, t, atb, csm):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(
        self,
        sde,
        score_fn,
        atb_mask,
        train_mask,
        snr,
        corrector_mse,
        sampling_fft,
        n_steps,
    ):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.corrector_mse = corrector_mse
        self.sampling_fft = sampling_fft
        self.n_steps = n_steps
        self.atb_mask = atb_mask
        self.train_mask = train_mask

    @abc.abstractmethod
    def update_fn(self, x, t, atb, csm):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, atb_mask, train_mask, probability_flow=False):
        super().__init__(sde, score_fn, atb_mask, train_mask, probability_flow)

    def update_fn(self, x, t, atb, csm):
        if isinstance(self.sde, sde_lib.HFS_SDE):
            x, x_mean = self.rsde.sde(x, t, atb, csm, self.atb_mask)
        else:
            dt = -1.0 / self.rsde.N  # 就是在离散化，就是delta t, reverse diffusion的dt在beta里
            z = torch.randn_like(x)
            drift, diffusion = self.rsde.sde(x, t, atb, csm, self.atb_mask)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, atb_mask, train_mask, probability_flow=False):
        super().__init__(sde, score_fn, atb_mask, train_mask, probability_flow)

    def update_fn(self, x, t, atb, csm):
        if isinstance(self.sde, sde_lib.HFS_SDE):
            z = torch.randn_like(x)
            x, x_mean = self.rsde.discretize(x, t, z, atb, csm, self.atb_mask)
        else:
            f, G = self.rsde.discretize(x, t, atb, csm, self.atb_mask)
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, atb_mask, train_mask, probability_flow=False):
        pass

    def update_fn(self, x, t, atb, csm):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        sde,
        score_fn,
        atb_mask,
        train_mask,
        snr,
        corrector_mse,
        sampling_fft,
        n_steps,
    ):
        super().__init__(
            sde,
            score_fn,
            atb_mask,
            train_mask,
            snr,
            corrector_mse,
            sampling_fft,
            n_steps,
        )
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
            and not isinstance(sde, sde_lib.HFS_SDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, atb, csm):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        corrector_mse = self.corrector_mse
        sampling_fft = self.sampling_fft

        if (
            isinstance(sde, sde_lib.VPSDE)
            or isinstance(sde, sde_lib.subVPSDE)
            or isinstance(sde, sde_lib.HFS_SDE)
        ):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            meas_grad = Emat_xyt(x, False, csm, self.atb_mask) - c2r(atb)
            meas_grad = Emat_xyt(meas_grad, True, csm, self.atb_mask)
            grad = score_fn(x, t)
            if isinstance(self.sde, sde_lib.HFS_SDE):
                grad = (
                    c2r(ifft2c_2d((1 - self.train_mask) * fft2c_2d(r2c(grad))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )
            meas_grad /= torch.norm(meas_grad)
            meas_grad *= torch.norm(grad)
            meas_grad *= corrector_mse

            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            if isinstance(self.sde, sde_lib.HFS_SDE):
                noise = (
                    c2r(ifft2c_2d((1 - self.train_mask) * fft2c_2d(r2c(noise))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )

            x_mean = x + step_size[:, None, None, None] * (grad - meas_grad)
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        sde,
        score_fn,
        atb_mask,
        train_mask,
        snr,
        corrector_mse,
        sampling_fft,
        n_steps,
    ):
        pass

    def update_fn(self, x, t, atb, csm):
        return x, x


def shared_predictor_update_fn(
    x,
    t,
    atb,
    csm,
    atb_mask,
    train_mask,
    sde,
    model,
    predictor,
    probability_flow,
    continuous,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(
            sde, score_fn, atb_mask, train_mask, probability_flow
        )
    else:
        predictor_obj = predictor(sde, score_fn, atb_mask, train_mask, probability_flow)
    return predictor_obj.update_fn(x, t, atb, csm)


def shared_corrector_update_fn(
    x,
    t,
    atb,
    csm,
    atb_mask,
    train_mask,
    sde,
    model,
    corrector,
    continuous,
    snr,
    corrector_mse,
    sampling_fft,
    n_steps,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(
            sde,
            score_fn,
            atb_mask,
            train_mask,
            snr,
            corrector_mse,
            sampling_fft,
            n_steps,
        )
    else:
        corrector_obj = corrector(
            sde,
            score_fn,
            atb_mask,
            train_mask,
            snr,
            corrector_mse,
            sampling_fft,
            n_steps,
        )
    return corrector_obj.update_fn(x, t, atb, csm)


def get_pc_sampler(
    config,
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    corrector_mse,
    sampling_fft,
    atb_mask,
    train_mask,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        atb_mask=atb_mask,
        train_mask=train_mask,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        atb_mask=atb_mask,
        train_mask=train_mask,
        corrector=corrector,
        sampling_fft=sampling_fft,
        continuous=continuous,
        snr=snr,
        corrector_mse=corrector_mse,
        n_steps=n_steps,
    )

    def pc_sampler(model, atb, atb_to_image, csm):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if isinstance(sde, sde_lib.HFS_SDE):
                z = sde.prior_sampling(shape).to(device)
                low_fre_img = (
                    c2r(Emat_xyt_complex(atb * train_mask, True, r2c(csm), 1.0))
                    .type(torch.FloatTensor)
                    .to(config.device)
                )
                x = low_fre_img + c2r(
                    ifft2c_2d((1 - train_mask) * fft2c_2d(r2c(z)))
                ).type(torch.FloatTensor).to(device)

            else:
                x = sde.prior_sampling(shape).to(device)

            if config.sampling.accelerated_sampling:
                sde.N = config.sampling.N
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, atb, csm, model=model)
                x = x.type(torch.FloatTensor).to(device)
                x_mean = x_mean.type(torch.FloatTensor).to(device)
                x, x_mean = predictor_update_fn(x, vec_t, atb, csm, model=model)
                x = x.type(torch.FloatTensor).to(device)
                x_mean = x_mean.type(torch.FloatTensor).to(device)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler


def add_title(path, title):
    img1 = cv2.imread(path)

    black = [0, 0, 0]
    constant = cv2.copyMakeBorder(
        img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black
    )
    height = 20
    violet = np.zeros((height, constant.shape[1], 3), np.uint8)
    violet[:] = (255, 0, 180)

    vcat = cv2.vconcat((violet, constant))

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0
    )
    cv2.imwrite(path, vcat)


def get_ode_sampler(
    sde,
    shape,
    inverse_scaler,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-3,
    device="cuda",
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(shape)
                .to(device)
                .type(torch.float32)
            )

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
