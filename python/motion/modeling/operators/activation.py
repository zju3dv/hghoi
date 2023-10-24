import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.utils.registry import Registry

ACTIVATIONS = Registry("activations")


@ACTIVATIONS.register_module()
class Softplus(nn.Softplus):
    pass


@ACTIVATIONS.register_module()
class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp()


@ACTIVATIONS.register_module()
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
