import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from motion.modeling.operators.builder import OPERATORS


@OPERATORS.register_module()
class Quantize(nn.Module):
    """
    A batch may only contain one code, leading to unstable training at early stage.
    We find training loss of average update is higher than cluster update

    Average update leads to high latent loss, where the ema of embed is unstable in converage stage
    This hurt reconstruction training

    For early training stage, we use average update, then we switch to cluster update
    """

    MAX_AVG_UPDATE_TIMES = -1

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(n_embed))
        self.register_buffer("embed_avg", embed.clone())

        self.update_times = nn.Parameter(torch.zeros(1), requires_grad=False)

    def extra_repr(self):
        str = f"embed_dim={self.dim}, n_embed={self.n_embed}, decay={self.decay}, eps={self.eps}"
        return str

    @torch.no_grad()
    def _update_buffers(self, inputs, embed_ind, act_mask=None):
        """
        Args:
            inputs (tensor): [B, C]
            embed_ind (int64 tensor): [B]
            act_mask (bool tensor, optional): [B, 1]. Defaults to None.

        """
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(inputs.dtype)
        if act_mask is not None:
            embed_onehot = embed_onehot * act_mask[..., None]

        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = inputs.transpose(0, 1) @ embed_onehot

        self.cluster_size.data.mul_(self.decay).add_(
            embed_onehot_sum, alpha=1 - self.decay
        )
        # Cluster update
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        prob = embed_onehot_sum / torch.sum(embed_onehot_sum)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8))
        used_curr = (embed_onehot_sum >= 1.0).sum()
        usage = torch.sum((self.cluster_size.view(self.n_embed, 1) >= 1.0).float())
        return dict(entropy=entropy, used_curr=used_curr, usage=usage)

    @torch.no_grad()
    def _update_embedding(self):
        dk_info = self._cluster_update_embedding()
        self.update_times += 1
        return dk_info

    def _cluster_update_embedding(self):
        # Cluster update
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        dk = torch.norm(self.embed - embed_normalized) / np.sqrt(
            np.prod(self.embed.shape)
        )
        self.embed.data.copy_(embed_normalized)
        return dict(dk=dk)

    @torch.no_grad()
    def compute_distance(self, inputs):
        """
        Args:
            inputs (tensor): [B, C]

        Returns:
            distance (tensor): [B, n_embed]
        """
        distance = (
            inputs.pow(2).sum(1, keepdim=True)
            - 2 * inputs @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        return distance

    @torch.no_grad()
    def calculate_index(self, distance):
        """
        Args:
            inputs (distance): [B, n_embed]

        Returns:
            embed_ind (int64 tensor): [B]
        """
        _, embed_ind = (-distance).max(1)
        return embed_ind

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        """
        Args:
            inputs (tensor): [B, C]

        Returns:
            embed_ind (int64 tensor): [B]
        """
        distance = self.compute_distance(inputs)
        embed_ind = self.calculate_index(distance)
        return embed_ind

    def forward(self, inputs, act_mask=None):
        """
        Args:
            inputs (tensor): [B, C]
            act_mask (bool tensor, optional): [B, 1] indicates whether these vectors should from the codebook
                some quantize vector might not be the selected action, we do not update these

        Returns:
            quantize (tensor): [B, C]
            diff (tensor): [B, C]
            embed_ind (tensor): [B, 1]
        """
        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1, inputs_shape[-1])
        embed_ind = self.find_nearest_embedding(inputs)
        if self.training:
            info = self._update_buffers(inputs, embed_ind, act_mask)
        else:
            info = {}

        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            info.update(self._update_embedding())

        diff = (quantize.detach() - inputs).pow(2)
        quantize = inputs + (quantize - inputs).detach()

        return (
            quantize.reshape(*inputs_shape[:-1], -1),
            diff.reshape(*inputs_shape, -1),
            embed_ind.reshape(*inputs_shape[:-1], 1),
            info,
        )

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


@OPERATORS.register_module()
class ProbQuantize(Quantize):
    @torch.no_grad()
    def calculate_index(self, distance):
        """
        Args:
            inputs (distance): [B, n_embed]

        Returns:
            embed_ind (int64 tensor): [B]
        """
        if self.training:
            distribution = Categorical(logits=-distance)
            embed_ind = distribution.sample()
        else:
            embed_ind = super().calculate_index(distance)
        return embed_ind


@OPERATORS.register_module()
class QuantizeONNX(Quantize):
    def forward(self, embed_ind):
        """
        Args:
            embed_ind (torch.int64): [B, 1]
        Returns:
            quantize (tensor): [B, C]
        """
        quantize = self.embed_code(embed_ind)
        return quantize
