import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.layer = nn.Linear(200, z_dim)

    def forward(self, x):
        return self.layer(x)
