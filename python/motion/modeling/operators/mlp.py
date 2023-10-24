import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        if norm is not None:
            norm = getattr(nn, norm)(out_channels)
        if activation is not None:
            activation = getattr(nn, activation)()
        if dropout is not None:
            assert isinstance(dropout, (float, int))
            dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm
        self.activation = activation
        self.dropout = dropout

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [*, in_channels]

        Returns:
            x (tensor): [*, out_channels]
        """
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResMLPBlock(MLPBlock):
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [*, in_channels]

        Returns:
            x (tensor): [*, out_channels]
        """
        res = x
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        x = res + x
        if self.activation is not None:
            x = self.activation(x)
        return x


class TimeMLPBlock(MLPBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
    ):
        super().__init__(in_channels, out_channels, norm, activation, dropout, bias)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))

    def forward(self, x, t):
        """_summary_

        Args:
            x (tensor): [*, in_channels]

        Returns:
            x (tensor): [*, out_channels]
        """
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        t = self.mlp(t)
        scale_shift = t.chunk(2, dim=-1)
        x = x * (scale_shift[0] + 1) + scale_shift[-1]
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResTimeMLPBlock(TimeMLPBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, time_dim, norm, activation, dropout, bias
        )
        self.res_mlp = None
        if in_channels != out_channels:
            self.res_mlp = nn.Linear(in_channels, out_channels)

    def forward(self, x, t):
        """_summary_

        Args:
            x (tensor): [*, in_channels]

        Returns:
            x (tensor): [*, out_channels]
        """
        res = x
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        t = self.mlp(t)
        scale_shift = t.chunk(2, dim=-1)
        x = x * (scale_shift[0] + 1) + scale_shift[-1]
        if self.res_mlp is not None:
            res = self.res_mlp(res)
        x = x + res
        if self.activation is not None:
            x = self.activation(x)
        return x


class MoEGating(nn.Module):
    def __init__(
        self,
        in_channels,
        num_experts,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.bias = bias
        self.fc = nn.Linear(in_channels, num_experts, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [B, in_channels]

        Returns:
            coef (tensor): [B, num_experts]
        """
        x = self.fc(x)
        coef = self.softmax(x)
        return coef


class MoEMLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_experts,
        norm=None,
        activation=None,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.bias = bias
        w_bound = (6.0 / (in_channels * out_channels)) ** 0.5
        w = torch.rand((num_experts, out_channels, in_channels)).uniform_(
            -w_bound, w_bound
        )
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(
            torch.zeros((num_experts, out_channels)), requires_grad=True
        )
        if norm is not None:
            norm = getattr(nn, norm)(out_channels)
        if activation is not None:
            activation = getattr(nn, activation)()
        self.norm = norm
        self.activation = activation

    def forward(self, x, coef):
        """_summary_

        Args:
            x (tensor): [B, in_channels]
            coef (tensor): [B, num_experts]

        Returns:
            x (tensor): [B, out_channels]
        """
        shape_x = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        coef = coef.reshape(-1, coef.shape[-1])
        w = torch.einsum("bi, ijk->bjk", coef, self.w)
        b = torch.einsum("bi, ik->bk", coef, self.b)
        x = torch.einsum("bk, bjk->bj", x, w) + b
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x.reshape(*shape_x, -1)
        return x


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        if norm is not None:
            norm = getattr(nn, norm)(out_channels)
        if activation is not None:
            activation = getattr(nn, activation)()
        if dropout is not None:
            assert isinstance(dropout, (float, int))
            dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = norm
        self.activation = activation
        self.dropout = dropout

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [N, C, L]

        Returns:
            x (tensor): [N, C, L]
        """
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Conv3dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        if norm is not None:
            norm = getattr(nn, norm)(out_channels)
        if activation is not None:
            activation = getattr(nn, activation)()
        if dropout is not None:
            assert isinstance(dropout, (float, int))
            dropout = nn.Dropout(dropout)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = norm
        self.activation = activation
        self.dropout = dropout

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [N, C, L]

        Returns:
            x (tensor): [N, C, L]
        """
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
