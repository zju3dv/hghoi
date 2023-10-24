from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.utils.utils import denormalize, to_cpu_numpy
from motion.utils.traj import relative_trajvec2worldmat, absolute_trajvec2worldmat
import motion.utils.matrix as matrix
from motion.loss.builder import LOSSES, build_loss


class BaseLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = None

    def _forward_loss(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def forward_loss(self, *args, **kwargs):
        return self._forward_loss(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.weight != 0:
            loss = self.weight * self.forward_loss(*args, **kwargs)
        else:
            return 0.0
        if torch.isinf(loss).any():
            print("[ERROR] INF LOSS!")
            __import__("ipdb").set_trace()
        if torch.isnan(loss).any():
            print("[ERROR] NAN LOSS!")
            __import__("ipdb").set_trace()
        return loss


@LOSSES.register_module()
class crossentropyloss(BaseLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight)
        self.loss = nn.CrossEntropyLoss()


@LOSSES.register_module()
class binarycrossentropy(BaseLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight)
        self.loss = nn.BCELoss()


class ReductionLoss(BaseLoss):
    def __init__(self, weight, reduction="mean", average_dim=None):
        super().__init__(weight)
        self.reduction = reduction
        self.average_dim = average_dim

    def reduction_loss(self, loss, pred):
        if self.reduction == "sum" and self.average_dim is not None:
            avg_num = 1.0
            if isinstance(pred, list):
                pred_ = pred[0]
            elif isinstance(pred, dict):
                # Sometimes, dict can contains a list of tensor
                for k in pred.keys():
                    if isinstance(pred[k], torch.Tensor):
                        pred_ = pred[k]
                        break
            else:
                pred_ = pred

            for d in self.average_dim:
                avg_num *= pred_.shape[d]
            loss = loss / avg_num
        return loss

    def forward_loss(self, pred, gt, *args, **kwargs):
        # Sometimes we simply need a regularizer
        if gt is None and isinstance(pred, torch.Tensor):
            gt = torch.zeros_like(pred)
        loss = super().forward_loss(pred, gt, *args, **kwargs)
        loss = self.reduction_loss(loss, pred)
        return loss


@LOSSES.register_module()
class mseloss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.MSELoss(reduction=reduction)


@LOSSES.register_module()
class l1loss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.L1Loss(reduction=reduction)


class WeightedLoss(BaseLoss):
    def __init__(self, weight):
        super().__init__(weight)

    def forward_loss(self, pred, gt, loss_w, *args, **kwargs):
        # Sometimes we simply need a regularizer
        if gt is None and isinstance(pred, torch.Tensor):
            gt = torch.zeros_like(pred)
        loss = super().forward_loss(pred, gt, *args, **kwargs)
        loss = loss * loss_w
        return loss.mean()


@LOSSES.register_module()
class weightedmseloss(WeightedLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight, *args, **kwargs)
        self.loss = nn.MSELoss(reduction="none")


@LOSSES.register_module()
class weightedl1loss(WeightedLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight, *args, **kwargs)
        self.loss = nn.L1Loss(reduction="none")


@LOSSES.register_module()
class smoothl1loss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.SmoothL1Loss(reduction=reduction)


@LOSSES.register_module()
class kldloss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.reduction_func = getattr(torch, reduction)

    def _forward_loss(self, pred, gt, *args, **kwargs):
        var = pred["var"] + 1e-9  # Sometimes var is so small that could be zero
        logvar = pred["logvar"]
        mu = pred["mu"]
        if gt is None:
            loss = -0.5 * self.reduction_func(1 + logvar - mu.pow(2) - var)
        else:
            p_var = gt["prior_var"]
            p_mu = gt["prior_mu"]
            logvar = var.log()
            p_logvar = p_var.log()
            loss = -0.5 * self.reduction_func(
                logvar - p_logvar - var / p_var - (mu - p_mu).pow(2) / p_var + 1
            )

        return loss


@LOSSES.register_module()
class focalloss(BaseLoss):
    def __init__(self, weight, alpha=0.25, gamma=2.0):
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = gamma

    def _forward_loss(self, pred, gt, *args, **kwargs):
        """_summary_

        Args:
            pred (tensor): [B, C]
            gt (tensor int64): [B]
        """
        pred = F.softmax(pred, dim=-1)
        num_classes = pred.size(-1)
        target = F.one_hot(gt, num_classes=num_classes)
        target = target.type_as(pred)
        pt = (1 - pred) * target + pred * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(
            self.gamma
        )
        loss = F.binary_cross_entropy(pred, target, reduction="none") * focal_weight
        return loss.sum(dim=-1).mean()


@LOSSES.register_module()
class labelsmoothCSE(BaseLoss):
    def __init__(self, weight, label_smoothing=0.1):
        super().__init__(weight)
        self.label_smoothing = label_smoothing
        self.logsoftmax = nn.LogSoftmax()

    def _forward_loss(self, pred, gt, *args, **kwargs):
        """_summary_

        Args:
            pred (tensor): [B, C]
            gt (tensor int64): [B]
        """
        num_classes = pred.size(-1)
        target = F.one_hot(gt, num_classes=num_classes)
        target = target.type_as(pred)
        soft_target = (
            target * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        )
        loss = -soft_target * self.logsoftmax(pred)
        return loss.sum(dim=-1).mean()


@LOSSES.register_module()
class sparseloss(BaseLoss):
    def __init__(self, weight, margin=2.0):
        super().__init__(weight)
        self.margin = margin

    def _forward_loss(self, pred, *args, **kwargs):
        """_summary_

        Args:
            pred (tensor): [B, C]
        """
        distance = ((pred[None] - pred[:, None]) ** 2).sum(dim=-1)
        loss = torch.clamp(self.margin - distance, min=0.0)
        loss = loss.mean()
        return loss


@LOSSES.register_module()
class trajaccumulationloss(BaseLoss):
    def __init__(self, weight):
        super().__init__(weight)
        self.loss = nn.MSELoss()

    def _forward_loss(self, pred, gt, mask, mean, std, *args, **kwargs):
        """_summary_

        Args:
            pred (tensor): [B, L, 4]
        """
        if mean is not None or std is not None:
            mean = mean[..., 384:388]
            std = std[..., 384:388]

            pred = denormalize(pred, mean, std)
            gt = denormalize(gt, mean, std)

        pred_abs_mat = relative_trajvec2worldmat(pred)[..., 1:, :, :]
        pred_abs_vec = matrix.mat2vec_batch(pred_abs_mat)
        pred_abs_vec = matrix.project_vec(pred_abs_vec)

        gt_abs_mat = relative_trajvec2worldmat(gt)[..., 1:, :, :]
        gt_abs_vec = matrix.mat2vec_batch(gt_abs_mat)
        gt_abs_vec = matrix.project_vec(gt_abs_vec)
        if mask is not None:
            pred_abs_vec = pred_abs_vec[mask]
            gt_abs_vec = gt_abs_vec[mask]
        loss = self.loss(pred_abs_vec, gt_abs_vec)
        return loss


@LOSSES.register_module()
class tccloss(BaseLoss):
    def __init__(
        self,
        weight,
        loss_type="regression_mse",
        similarity_type="l2",
        temperature=0.1,
        variance_lambda=0.001,
        huber_delta=0.1,
        normalize_indices=True,
        normalize_embeddings=False,
        **kwargs
    ):
        super().__init__(weight)
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.temperature = temperature
        self.variance_lambda = variance_lambda
        self.huber_delta = huber_delta
        self.normalize_indices = normalize_indices
        self.normalize_embeddings = normalize_embeddings

    def _forward_loss(self, pred, gt, *args, **kwargs):
        steps = gt["frame_idxs"]
        seq_lens = gt["video_len"]

        embs = pred["embs"]
        batch_size, num_cc_frames = embs.shape[:2]
        loss = deterministic_tcc_loss(
            embs=embs,
            idxs=steps,
            seq_lens=seq_lens,
            num_cc=num_cc_frames,
            batch_size=batch_size,
            loss_type=self.loss_type,
            similarity_type=self.similarity_type,
            temperature=self.temperature,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices,
            normalize_dimension=(not self.normalize_embeddings),
        )
        return loss.mean()


def deterministic_tcc_loss(
    embs,
    idxs,
    seq_lens,
    num_cc,
    batch_size,
    loss_type,
    similarity_type,
    temperature,
    variance_lambda,
    huber_delta,
    normalize_indices,
    normalize_dimension,
):
    """Deterministic alignment between all pairs of sequences in a batch."""

    batch_size = embs.shape[0]

    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = align_sequence_pair(
                    embs[i],
                    embs[j],
                    similarity_type,
                    temperature,
                    normalize_dimension,
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(idxs[i : i + 1].expand(num_cc, -1))
                seq_lens_list.append(seq_lens[i : i + 1].expand(num_cc))

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    return regression_loss(
        logits,
        labels,
        steps,
        seq_lens,
        loss_type,
        normalize_indices,
        variance_lambda,
        huber_delta,
    )


def pairwise_l2_sq(
    x1,
    x2,
):
    """Compute pairwise squared Euclidean distances."""
    return torch.cdist(x1, x2).pow(2)


def get_scaled_similarity(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
    """Return pairwise similarity."""
    if similarity_type == "l2":
        similarity = -1.0 * pairwise_l2_sq(emb1, emb2)
        if normalize_dimension:
            similarity = similarity / emb1.shape[1]
    else:  # Cosine similarity.
        similarity = torch.mm(emb1, emb2.t())
    similarity = similarity / temperature
    return similarity


def align_sequence_pair(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
    """Align a pair of sequences."""
    max_num_steps = emb1.shape[0]
    sim_12 = get_scaled_similarity(
        emb1, emb2, similarity_type, temperature, normalize_dimension
    )
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)  # Row-wise softmax.
    nn_embs = torch.mm(softmaxed_sim_12, emb2)
    sim_21 = get_scaled_similarity(
        nn_embs, emb1, similarity_type, temperature, normalize_dimension
    )
    logits = sim_21
    labels = torch.arange(max_num_steps).to(logits.device)
    return logits, labels


def regression_loss(
    logits,
    labels,
    steps,
    seq_lens,
    loss_type,
    normalize_indices,
    variance_lambda,
    huber_delta,
):
    """Cycle-back regression loss."""
    if normalize_indices:
        steps = steps.float() / seq_lens[:, None].float()
    else:
        steps = steps.float()
    labels = one_hot(labels, logits.shape[1])
    beta = F.softmax(logits, dim=1)
    time_true = (steps * labels).sum(dim=1)
    time_pred = (steps * beta).sum(dim=1)
    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:  # Variance-aware regression.
            # Compute log of prediction variance.
            time_pred_var = (steps - time_pred.unsqueeze(1)).pow(2) * beta
            time_pred_var = torch.log(time_pred_var.sum(dim=1))
            err_sq = (time_true - time_pred).pow(2)
            loss = torch.exp(-time_pred_var) * err_sq + variance_lambda * time_pred_var
            return loss.mean()
        return F.mse_loss(time_pred, time_true)
    return huber_loss(time_pred, time_true, huber_delta)


def one_hot(y, K, smooth_eps=0):  # pylint: disable=invalid-name
    """One-hot encodes a tensor with optional label smoothing.

    Args:
      y: A tensor containing the ground-truth labels of shape (N,), i.e. one label
        for each element in the batch.
      K: The number of classes.
      smooth_eps: Label smoothing factor in [0, 1] range.

    Returns:
      A one-hot encoded tensor.
    """
    assert 0 <= smooth_eps <= 1
    assert y.ndim == 1, "Label tensor must be rank 1."
    y_hot = torch.eye(K)[y] * (1 - smooth_eps) + (smooth_eps / (K - 1))
    return y_hot.to(y.device)


def huber_loss(
    input,  # pylint: disable=redefined-builtin
    target,
    delta,
    reduction="mean",
):
    """Huber loss with tunable margin [1].

    This is a more general version of PyTorch's
    `torch.nn.functional.smooth_l1_loss` that allows the user to change the
    margin parameter.

    Args:
      input: A `FloatTensor` representing the model output.
      target: A `FloatTensor` representing the target values.
      delta: Given the tensor difference `diff`, delta is the value at which we
        incur a quadratic penalty if `diff` is at least delta and a linear penalty
        otherwise.
      reduction: The reduction strategy on the final loss tensor.

    Returns:
      If reduction is `none`, a 2D tensor.
      If reduction is `sum`, a 1D tensor.
      If reduction is `mean`, a scalar 1D tensor.

    References:
      [1]: Wikipedia Huber Loss,
      https://en.wikipedia.org/wiki/Huber_loss
    """
    assert isinstance(input, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(target, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert reduction in [
        "none",
        "mean",
        "sum",
    ], "reduction method is not supported"

    diff = target - input
    diff_abs = torch.abs(diff)
    cond = diff_abs <= delta
    loss = torch.where(cond, 0.5 * diff ** 2, (delta * diff_abs) - (0.5 * delta ** 2))
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    return loss.sum(dim=-1)  # reduction == "sum"
