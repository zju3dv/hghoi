import copy
from re import I
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.modeling.modules.builder import build_module
from motion.utils.dm import (
    linear_beta_schedule,
    cosine_beta_schedule,
    extract,
    identity,
)


class BaseModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.setup(cfg)
        self.init_build(cfg)
        self.build_model(cfg)

    def setup(self, cfg):
        pass

    def init_build(self, cfg):
        pass

    def build_model(self, cfg):
        raise NotImplementedError


class BaseHierarchyModel(BaseModel):
    def setup(self, cfg):
        super().setup(cfg)
        self.levels = self.cfg.levels


class BasePredictor(BaseModel):
    def build_model(self, cfg):
        decoder_cfg = cfg.clone()
        decoder_cfg.TYPE = cfg.DECODER_TYPE
        self.decoder = build_module(decoder_cfg)


class BaseAE(BasePredictor):
    def build_model(self, cfg):
        super().build_model(cfg)
        encoder_cfg = cfg.clone()
        encoder_cfg.TYPE = cfg.ENCODER_TYPE
        self.encoder = build_module(encoder_cfg)


class BaseVAE(BaseAE):
    def setup(self, cfg):
        super().setup(cfg)
        self.use_pred_dist_during_test = cfg.use_pred_dist_during_test

    def reparameterize(self, dist, is_socket=False):
        if self.training or self.use_pred_dist_during_test or is_socket:
            mu = dist["mu"]
            var = dist["var"]
        else:
            mu = torch.zeros_like(dist["mu"])
            var = torch.ones_like(dist["var"])
        std = var ** 0.5
        eps = torch.randn_like(std)
        return mu + eps * std


class BaseVAEONNX(BaseModel):
    def build_model(self, cfg):
        decoder_cfg = cfg.clone()
        decoder_cfg.TYPE = cfg.DECODER_TYPE + "ONNX"
        self.decoder = build_module(decoder_cfg)


class BaseVQVAE(BaseVAE):
    def build_model(self, cfg):
        super().build_model(cfg)
        quantize_cfg = cfg.clone()
        quantize_cfg.TYPE = cfg.QUANTIZE_TYPE
        self.quantize = build_module(quantize_cfg)


class BaseVQVAEONNX(BaseVAEONNX):
    def build_model(self, cfg):
        super().build_model(cfg)
        quantize_cfg = cfg.clone()
        quantize_cfg.TYPE = cfg.QUANTIZE_TYPE + "ONNX"
        self.quantize = build_module(quantize_cfg)


class BaseCodeModel(BaseModel):
    def build_model(self, cfg):
        encoder_cfg = cfg.clone()
        predictor_cfg = cfg.clone()
        encoder_cfg.TYPE = cfg.ENCODER_TYPE
        predictor_cfg.TYPE = cfg.PREDICTOR_TYPE
        self.encoder = build_module(encoder_cfg)
        self.predictor = build_module(predictor_cfg)


class BaseCodeModelONNX(BaseModel):
    def build_model(self, cfg):
        encoder_cfg = cfg.clone()
        predictor_cfg = cfg.clone()
        encoder_cfg.TYPE = cfg.ENCODER_TYPE + "ONNX"
        predictor_cfg.TYPE = cfg.PREDICTOR_TYPE + "ONNX"
        self.encoder = build_module(encoder_cfg)
        self.predictor = build_module(predictor_cfg)


class BaseDiffusionModel(BasePredictor):
    def setup(self, cfg):
        super().setup(cfg)

        self.is_minmax = cfg.is_minmax

        self.self_condition = cfg.self_condition
        self.objective = cfg.objective

        assert self.objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start)"

        num_timesteps = cfg.num_timesteps
        if cfg.beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        elif cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise NotImplementedError(cfg.schedule_type)
        betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = cfg.sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = cfg.ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma = cfg.p2_loss_weight_gamma  # 0.
        p2_loss_weight_k = cfg.p2_loss_weight_k  # 1.

        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data=None, x_self_cond=None, clip_x_start=False):
        model_output = self.decoder.inference_forward(x, t, x_self_cond, **data)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0)
            if self.is_minmax and clip_x_start
            else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output["pred"]
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output["pred"]
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, data=None, x_self_cond=None, clip_denoised=False):
        pred_noise, pred_x_start = self.model_predictions(
            x, t, data, x_self_cond, clip_denoised
        )
        x_start = pred_x_start

        if self.is_minmax and clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data=None, x_self_cond=None, clip_denoised=False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            data=data,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, data=None):
        batch, device = shape[0], self.betas.device

        x = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(x, t, data, self_cond, True)

        return x

    def inpainting(self, x_start, data, time_next, **kwargs):
        return x_start

    @torch.no_grad()
    def ddim_sample(self, shape, data=None, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                x, time_cond, data, self_cond, clip_x_start=clip_denoised
            )

            x_start = self.inpainting(x_start, data, time_next)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x

    def sample(self, *args, **kwargs):
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(*args, **kwargs)
