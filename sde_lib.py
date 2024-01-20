"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
from ast import Not
import torch
import numpy as np
from utils.utils import *


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, config):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.config = config
        self.N = config.model.num_scales

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        config = self.config
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.config = config
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, atb, csm, atb_mask):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                grad = score_fn(x, t)
                meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= self.config.sampling.mse
                drift = drift - diffusion[:, None, None, None] ** 2 * (
                    grad - meas_grad
                ) * (0.5 if self.probability_flow else 1.0)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, atb, csm, atb_mask):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                grad = score_fn(x, t)
                meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= self.config.sampling.mse
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * (grad - meas_grad) * (
                    0.5 if self.probability_flow else 1.0
                )

                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    def __init__(self, config):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(config)
        self.sigma_min = config.model.sigma_min
        self.sigma_max = config.model.sigma_max
        self.N = config.model.num_scales
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N)
        )
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.30
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t).to(t.device),
            self.discrete_sigmas[timestep.cpu() - 1].to(t.device),
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G


class VPSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.beta_0 = config.model.beta_min
        self.beta_1 = config.model.beta_max
        self.N = config.model.num_scales
        self.config = config
        self.discrete_betas = torch.linspace(
            self.beta_0 / self.N, self.beta_1 / self.N, self.N
        )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.32
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)

        f = torch.sqrt(alpha)[:, None, None, None] * x - x

        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, config):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.beta_0 = config.model.beta_min
        self.beta_1 = config.model.beta_max
        self.N = config.model.num_scales

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class HFS_SDE(SDE):
    def __init__(self, config):
        """Construct a MultiScale SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.config = config
        self.beta_0 = config.model.beta_min
        self.beta_1 = config.model.beta_max
        self.N = config.model.num_scales
        self.discrete_betas = torch.linspace(
            self.beta_0 / self.N, self.beta_1 / self.N, self.N
        )

        self.mask = get_mask(config, "sde")

        # TODO
        self.alphas = 1.0 - self.discrete_betas

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        raise NotImplementedError

    def marginal_prob(self, x, t):
        max_N = 1000000
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        M_hat_x = (
            c2r(ifft2c_2d((1.0 - self.mask) * fft2c_2d(r2c(x))))
            .type(torch.FloatTensor)
            .to(self.config.device)
        )

        if self.config.training.mean_equal == "equal":
            mean = ((1 + log_mean_coeff / max_N) ** max_N - 1) * M_hat_x + x

        else:
            mean = (torch.exp(log_mean_coeff[:, None, None, None]) - 1) * M_hat_x + x
        std_coeff = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        return mean, std_coeff

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        raise NotImplementedError

    def discretize(self, x, t, z):
        raise NotImplementedError

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        mask = self.mask
        config = self.config

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self.mask = mask

            @property
            def T(self):
                return T

            def sde(self, x, t, atb, csm, atb_mask):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                beta_t = config.model.beta_min + t * (
                    config.model.beta_max - config.model.beta_min
                )
                M_hat_x = (
                    c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(x))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )
                drift = -0.5 * beta_t[:, None, None, None] * M_hat_x

                diffusion = torch.sqrt(beta_t)

                grad = score_fn(x, t)
                ##################################
                # grad = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(grad)))).type(torch.FloatTensor).to(x.device)
                ##################################
                meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= config.sampling.mse

                drift = drift - diffusion[:, None, None, None] ** 2 * (
                    grad - meas_grad
                ) * (0.5 if self.probability_flow else 1.0)

                #############
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion

                dt = -1.0 / self.N
                z = torch.randn_like(x)

                z = (
                    c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(z))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )

                x_mean = x + drift * dt
                x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
                return x, x_mean

            def discretize(self, x, t, z, atb, csm, atb_mask):
                raise NotImplementedError

        return RSDE()
