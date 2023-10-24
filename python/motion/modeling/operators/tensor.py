import torch
import torch.nn as nn


class TransposeModule(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x.transpose(*self.index)


class UnsqueezeModule(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x.unsqueeze(self.index)


class SqueezeModule(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x.squeeze(self.index)
