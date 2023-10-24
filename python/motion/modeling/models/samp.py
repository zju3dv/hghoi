import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from random import random
from motion.modeling.modules.embedding.embedding import EmbeddingLayer
from motion.modeling.model import (
    BaseModel,
    BaseAE,
    BaseVAE,
    BaseVQVAE,
    BaseCodeModel,
    BasePredictor,
    BaseDiffusionModel,
)
from motion.modeling.builder import MODELS
from motion.modeling.modules.builder import build_module
from motion.utils.dm import extract, identity


@MODELS.register_module()
class QPoseNet(BaseVQVAE):
    def forward(self, data):
        output = {}
        feat = self.encoder(**data)
        quant_output = self.quantize(feat)
        output.update(quant_output)
        dec_output = self.decoder(**data, **output)
        output.update(dec_output)
        output["quant_net"] = self.quantize
        return output

    def decode_forward(self, data):
        output = {}
        code = data["code"]
        quant_output = self.quantize.decode_forward(code)
        output.update(quant_output)
        dec_output = self.decoder(**data, **output)
        output.update(dec_output)
        output["quant_net"] = self.quantize
        return output


@MODELS.register_module()
class QMotionCode(BasePredictor):
    def forward(self, data):
        output = {}
        pred = self.decoder(**data)
        output.update(pred)
        return output

    def socket_forward(self, data, **kwargs):
        return self.decoder.inference_forward(**data, **kwargs)


class DDPM(BaseDiffusionModel):
    def setup(self, cfg):
        super().setup(cfg)
        self.self_condition_prob = cfg.self_condition_prob
        self.cond_mask_prob = cfg.get("cond_mask_prob", 0.0)
        self.guidance_scale = cfg.get("guidance_scale", 2.5)

    def model_predictions(self, x, t, data=None, x_self_cond=None, clip_x_start=False):
        if self.cond_mask_prob > 0.0:
            uncond_output = self.decoder.inference_forward(
                x, t, x_self_cond, **data, uncond=True
            )
            cond_output = self.decoder.inference_forward(x, t, x_self_cond, **data)
            model_output = {}
            for k in cond_output.keys():
                model_output[k] = uncond_output[k] + self.guidance_scale * (
                    cond_output[k] - uncond_output[k]
                )
        else:
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

    @torch.no_grad()
    def p_sample_loop(self, data, **kwargs):
        shape = self.decoder.get_shape(data)
        return super().p_sample_loop(shape, data)

    @torch.no_grad()
    def ddim_sample(self, data, **kwargs):
        shape = self.decoder.get_shape(data)
        return super().ddim_sample(shape, data, clip_denoised=True)

    def socket_forward(self, data, *args, **kwargs):
        return self.sample(data, *args, **kwargs)


@MODELS.register_module()
class DDPMMotion(DDPM):
    def forward(self, data, **kwargs):
        output = {}
        y = data["y"]
        b = y.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=y.device).long()
        noise = torch.randn_like(y)
        x_start = y
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < self.self_condition_prob:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, data)[1]
                x_self_cond = x_self_cond.detach()
                # x_self_cond.detach_()
        model_out = self.decoder(x, t, x_self_cond, **data)
        output.update(model_out)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        output["target"] = target
        loss_w = extract(self.p2_loss_weight, t, x.shape)
        loss_w = loss_w.expand(-1, y.shape[1], -1)  # b, l, 1
        output["loss_w"] = loss_w
        return output

    @torch.no_grad()
    def sample(self, data, *args, **kwargs):
        output = {}
        pred = super().sample(data, *args, **kwargs)
        output["y_hat"] = pred
        return output


@MODELS.register_module()
class DDPMMilestone(DDPM):
    def forward(self, data, **kwargs):
        output = {}
        y = data["y"]
        traj = data["traj"]
        b = y.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (b,), device=data["traj"].device
        ).long()
        noise_y = torch.randn_like(y)
        noise_traj = torch.randn_like(traj)
        x_start = torch.cat((traj, y), dim=-1)
        noise = torch.cat((noise_traj, noise_y), dim=-1)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < self.self_condition_prob:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, data)[1]
                x_self_cond = x_self_cond.detach()
                # x_self_cond.detach_()
        model_out = self.decoder(x, t, x_self_cond, **data)
        output.update(model_out)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        output.update(self.decoder.split_pred(model_out["pred"]))
        output.update(self.decoder.split_pred(target, prefix="target"))
        loss_w = extract(self.p2_loss_weight, t, x.shape)
        loss_w = loss_w.expand(-1, y.shape[1], -1)  # b, l, 1
        output["loss_w"] = loss_w
        return output

    @torch.no_grad()
    def sample(self, data, *args, **kwargs):
        output = {}
        if "control_t" not in data.keys() or data["control_t"] is None:
            time_output = self.decoder.pred_length(data)
            data.update(time_output)
            if "prior_t" in data.keys():
                data["lengths"] = time_output["pred_lengths"]
            output.update(time_output)
            output["t"] = time_output["pred_t"]
        else:
            data["pred_lengths"] = data["control_t"]
            data["lengths"] = data["control_t"]
            output["t"] = data["control_t"]
        pred = super().sample(data, *args, **kwargs)
        output.update(self.decoder.split_pred(pred, prefix=""))
        output["state"] = output["y_hat"]
        return output

    def inpainting(self, x_start, data, time_next):
        if (
            "control_x" in data.keys()
            and time_next >= 100
            and data["control_x"] is not None
        ):
            ratio = 0.5
            x_start[..., :2] = (
                ratio * x_start[..., :2] + (1 - ratio) * data["control_x"][..., :2]
            )
            x_start[..., 4:6] = (
                ratio * x_start[..., 4:6] + (1 - ratio) * data["control_x"][..., 4:6]
            )
            # x_start[..., 8:10] = data["control_x"][..., 8:10]
        return x_start


@MODELS.register_module()
class DDPMTraj(DDPM):
    def forward(self, data, **kwargs):
        output = {}
        y = data["y"]
        traj = data["traj"]
        b = y.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (b,), device=data["traj"].device
        ).long()
        noise_y = torch.randn_like(y)
        noise_traj = torch.randn_like(traj)
        x_start = torch.cat((traj, y), dim=-1)
        noise = torch.cat((noise_traj, noise_y), dim=-1)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < self.self_condition_prob:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, data)[1]
                x_self_cond = x_self_cond.detach()
                # x_self_cond.detach_()
        model_out = self.decoder(x, t, x_self_cond, **data)
        output.update(model_out)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        output.update(self.decoder.split_pred(model_out["pred"]))
        output.update(self.decoder.split_pred(target, prefix="target"))
        loss_w = extract(self.p2_loss_weight, t, x.shape)
        loss_w = loss_w.expand(-1, y.shape[1], -1)  # b, l, 1
        output["loss_w"] = loss_w
        return output

    @torch.no_grad()
    def sample(self, data, *args, **kwargs):
        output = {}
        data["L"] = self.cfg.L
        pred = super().sample(data, *args, **kwargs)
        output.update(self.decoder.split_pred(pred, prefix=""))
        output["state"] = output["y_hat"]
        return output
