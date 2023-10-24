import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
from motion.modeling.models.samp import DDPM
from motion.modeling.builder import MODELS
from motion.utils.dm import extract


@MODELS.register_module()
class GaussianDiffusion(DDPM):
    def forward(self, data, **kwargs):
        output = {}
        y = data["img"]
        b = y.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=y.device).long()
        noise = torch.randn_like(y)
        x_start = y
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < self.self_condition_prob:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, data)[1]
                x_self_cond.detach_()
        model_out = self.decoder(x, t, x_self_cond, **data)
        output.update(model_out)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        output["target"] = target
        loss_w = extract(self.p2_loss_weight, t, x.shape)
        # __import__("ipdb").set_trace()
        # loss_w = loss_w.expand(-1, y.shape[1], -1)  # b, l, 1
        output["loss_w"] = loss_w
        return output

    @torch.no_grad()
    def sample(self, data, *args, **kwargs):
        output = {}
        pred = super().sample(data, *args, **kwargs)
        output["pred"] = pred
        return output
