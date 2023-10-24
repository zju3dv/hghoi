import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kld(mu1, var1, mu2=None, var2=None):
    """_summary_
        calculate kl between N(mu1, var1) and N(mu2, var2)
        Default is kl between N(mu1, var1) and N(0, 1)

    Args:
        mu1 (tensor): [*, C]
        var1 (tensor): [*, C]
        mu2 (tensor, optional): [*, C]. Defaults to None.
        var2 (tensor, optional): [*, C]. Defaults to None.

    Returns:
        kld (tesnor): []
    """
    logvar1 = var1.log()
    if mu2 is not None:
        logvar2 = var2.log()
        kld = -0.5 * torch.mean(
            logvar1 - logvar2 - var1 / var2 - (mu1 - mu2).pow(2) / var2 + 1
        )
    else:
        kld = -0.5 * torch.mean(1 + logvar1 - mu1.pow(2) - var1)
    return kld


def cross_entropy(logits, targets):
    """_summary_

    Args:
        logits (tensor): [*, C]
        targets (tensor int64): [*]

    Returns:
        entropy (tensor): []
    """
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    entropy = F.cross_entropy(logits, targets)
    return entropy


def accuracy(pred, targets):
    """_summary_

    Args:
        pred (tensor int64): [*]
        targets (tensor int64): [*]

    Returns:
        acc (tensor): []
    """
    flags = pred == targets
    return flags.sum() / targets.shape.numel()
