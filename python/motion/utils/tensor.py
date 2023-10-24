import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths, device=torch.device("cpu")):
    """
    Args:
        lengths (List[int]):
        device (torch.device):

    Returns:
        Tensor (bool): [B, T]
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask
