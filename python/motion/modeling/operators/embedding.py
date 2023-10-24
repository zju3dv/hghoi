import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingONNX(nn.Module):
    # Convert to linear
    def __init__(self, embedding_n, embedding_dim, part_n):
        super().__init__()
        self.embedding_n = embedding_n
        self.embedding_dim = embedding_dim
        self.part_n = part_n
        self.quant_n = embedding_n // part_n

        for i in range(part_n):
            setattr(
                self,
                f"weight{i}",
                nn.Parameter(torch.randn(embedding_dim, self.quant_n)),
            )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "weight"
        w = state_dict[key].reshape(self.part_n, self.quant_n, self.embedding_dim)
        for i in range(self.part_n):
            weight = getattr(self, f"weight{i}")
            with torch.no_grad():
                weight.copy_(w[i].T)

    def forward(self, x):
        code_feat = []
        for i in range(self.part_n):
            w = getattr(self, f"weight{i}")
            code_embed = F.linear(x[:, i : i + 1, :], w)
            code_feat.append(code_embed)
        x = torch.cat(code_feat, dim=1)

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_n, embedding_dim, embedding_shape=-1, interval=1):
        super().__init__()
        self.embedding_n = embedding_n
        self.embedding_dim = embedding_dim
        self.embedding_shape = embedding_shape
        assert embedding_shape > -3, "Do not support embedding_shape <= -3"
        self.interval = interval
        self.embedding = nn.Embedding(embedding_n, embedding_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (tensor int64): [B, *]

        Returns:
            x (tensor float): [B, *, C]
        """
        if self.embedding_shape < -1:
            interval = torch.arange(x.shape[-1], device=x.device) * (
                self.embedding_n // x.shape[-1]
            )
            x += interval
        x = self.embedding(x)
        x = x.reshape(x.shape[: self.embedding_shape] + (-1,))
        return x
