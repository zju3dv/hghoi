import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F


NORM_DICT = {
    "MN": lambda channels: ModifiedNorm(),
    "LN": lambda channels: nn.LayerNorm(channels),
    "DN": lambda channels: DivideNorm(),
}


class ModifiedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        out = x / torch.sqrt((x * x).mean(dim=2, keepdim=True) + self.eps)
        return out


class DivideNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 4.0


class LayerNormONNX(nn.Module):
    # ONNX does not support nn.LayerNorm
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # unbiased variance calculation since onnx does not support var()
        var_x = torch.sum((x - x.mean(dim=-1, keepdim=True)) ** 2, dim=-1) / (
            x.shape[-1] - 1
        )
        norm_x = (x - x.mean(dim=-1, keepdim=True)) / (
            var_x[..., None] + self.eps
        ) ** 0.5
        y = norm_x * self.weight + self.bias
        return y


def _get_norm(norm_type, *args, **kwargs):
    return NORM_DICT[norm_type](*args, **kwargs)
