import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, *args, **kwargs):
        return x


class Concatenation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, *args, **kwargs):
        dim = kwargs["dim"]
        return torch.cat((x, y), dim=dim)


class ResidualSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, *args, **kwargs):
        return x + y


class MinusSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, *args, **kwargs):
        y = self.sigmoid(y)
        x = x * (1 - y)
        return x


FUSION_DICT = {
    "none": Identity,
    "concat": Concatenation,
    "sum": ResidualSum,
    "msigmoid": MinusSigmoid,
}
