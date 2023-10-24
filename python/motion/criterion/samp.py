import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.utils.loss import compute_kld, cross_entropy, accuracy
from motion.utils.utils import (
    denormalize,
    unnormalize_to_zero_to_one,
)
from motion.utils.traj import relative_trajvec2worldmat
import motion.utils.matrix as matrix
from .criterion import SimpleCriterion
from .builder import CRITERIONS


@CRITERIONS.register_module()
class PoseNetCriterion(SimpleCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_types = ["reconstruction", "rec_l1", "latent", "kld"]

    def _compute_kld(self, pred):
        if "mu" not in pred.keys():
            return 0.0
        mu = pred["mu"]
        var = pred["var"]
        kld = compute_kld(mu, var)
        return kld.item()

    def _compute_result(self, pred, gt):
        """
        Args:
            pred (dict):
                "y_hat": [1, C]
            gt (dict):
                "y": [1, C]

        Returns:
            errors(dict[float]):
                reconstruction(float)
                diff(float)
                kld(float)
            (processed pred and gt): here we do not need, set as None
        """
        p_pose = pred["y_hat"]
        gt_pose = gt["y"]

        pose_mean, pose_std = gt["pose_mean"], gt["pose_std"]
        if self.cfg.is_minmax:
            pose_max, pose_min = gt["pose_max"], gt["pose_min"]
            p_pose = unnormalize_to_zero_to_one(p_pose, pose_min, pose_max)
            gt_pose = unnormalize_to_zero_to_one(gt_pose, pose_min, pose_max)

        p_pose = denormalize(p_pose, pose_mean, pose_std)
        gt_pose = denormalize(gt_pose, pose_mean, pose_std)
        if "mask" in gt.keys():
            mask = gt["mask"]
            p_pose = p_pose[mask]
            gt_pose = gt_pose[mask]

        rec_error = F.mse_loss(p_pose, gt_pose, reduction="mean").item()  # element-wise
        rec_error_l1 = (p_pose - gt_pose).abs().sum(dim=-1).mean().item()  # data-wise
        errors = {"reconstruction": rec_error, "rec_l1": rec_error_l1}

        if "contact" in pred.keys():
            p_contact = pred["contact"]
            gt_contact = gt["contact"]
            contact_mean, contact_std = gt["contact_mean"], gt["contact_std"]
            if self.cfg.is_minmax:
                contact_max, contact_min = gt["contact_max"], gt["contact_min"]
                p_contact = unnormalize_to_zero_to_one(
                    p_contact, contact_min, contact_max
                )
                gt_contact = unnormalize_to_zero_to_one(
                    gt_contact, contact_min, contact_max
                )
            p_contact = denormalize(p_contact, contact_mean, contact_std)
            gt_contact = denormalize(gt_contact, contact_mean, contact_std)
            if "mask" in gt.keys():
                mask = gt["mask"]
                p_contact = p_contact[mask]
                gt_contact = gt_contact[mask]
            rec_error_contact_l1 = (
                (p_contact - gt_contact).abs().sum(dim=-1).mean().item()
            )  # data-wise
            errors["rec_contact_l1"] = rec_error_contact_l1

        if "diff" in pred.keys():
            latent_error = pred["diff"].mean().item()
            errors["latent"] = latent_error
        if "mu" in pred.keys():
            errors["kld"] = self._compute_kld(pred)
        return errors, (pred, gt)


@CRITERIONS.register_module()
class CodeNetCriterion(SimpleCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_types = ["nll"]

    def _compute_result(self, pred, gt):
        """
        Args:
            pred (dict):
                'logits' (tensor, float): [B, T, class] or [B, T, P, class]
                                          (sometimes we predict P parts for each sample)
            gt (dict):
                'y' (tensor, int): [B] or [B, P]

        Returns:
            errors (dict):
                'nll' (float)
        """
        nll = cross_entropy(pred["logits"], gt["y"])
        errors = {"nll": nll}
        pred_cls = pred["logits"].argmax(dim=-1)  # B, T, J
        pred["cls"] = pred_cls
        return errors, (pred, gt)


@CRITERIONS.register_module()
class KeyCodeNetCriterion(CodeNetCriterion):
    def _compute_result(self, pred, gt):
        """
        Args:
            pred (dict):
                'logits' (tensor, float): [B, T, class] or [B, T, P, class]
                                          (sometimes we predict P parts for each sample)
            gt (dict):
                'y' (tensor, int): [B] or [B, P]

        Returns:
            errors (dict):
                'nll' (float)
        """
        errors = {}
        for k, v in pred["logits"].items():
            nll = cross_entropy(v, gt["code"][k])
            pred_cls = v.argmax(dim=-1, keepdim=True)
            acc = accuracy(pred_cls, gt["code"][k])
            errors["nll_" + k] = nll
            errors["acc_" + k] = acc
        return errors, (pred, gt)


@CRITERIONS.register_module()
class TimeTrajCriterion(SimpleCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_types = ["nll"]

    def _compute_kld(self, pred):
        if "mu" not in pred.keys():
            return 0.0
        mu = pred["mu"]
        var = pred["var"]
        kld = compute_kld(mu, var)
        return kld.item()

    def _compute_result(self, pred, gt):
        """
        Args:
            pred (dict):
                'y_hat' (tensor, float): [B, T, class] or [B, T, P, class]
                                          (sometimes we predict P parts for each sample)
            gt (dict):
                'y' (tensor, int): [B] or [B, P]

        Returns:
            errors (dict):
                'nll' (float)
        """
        if "t" in pred.keys():
            nll = cross_entropy(pred["t"], gt["lengths"])
            errors = {"nll": nll}
            pred_cls = pred["t"].argmax(dim=-1)  # B, T, J
            pred["cls"] = pred_cls

        p_traj = pred["traj"]
        gt_traj = gt["traj"]

        p_win_traj = pred["y_hat"]
        gt_win_traj = gt["y"]

        win_traj_mean, win_traj_std = gt["trajstate_mean"], gt["trajstate_std"]

        mask = gt["mask"]

        pos_rec_error_l2_nonorm = (
            (p_traj[mask] - gt_traj[mask]).norm(2, dim=-1).mean().item()
        )  # data-wise no norm
        win_rec_error_l2_nonorm = (
            (p_win_traj[mask] - gt_win_traj[mask]).norm(2, dim=-1).mean().item()
        )  # data-wise no norm

        if self.cfg.is_minmax:
            traj_max, traj_min = gt["traj_max"], gt["traj_min"]
            p_traj = unnormalize_to_zero_to_one(p_traj, traj_min, traj_max)
            gt_traj = unnormalize_to_zero_to_one(gt_traj, traj_min, traj_max)
            win_max, win_min = gt["trajstate_max"], gt["trajstate_min"]
            p_win_traj = unnormalize_to_zero_to_one(p_win_traj, win_min, win_max)
            gt_win_traj = unnormalize_to_zero_to_one(gt_win_traj, win_min, win_max)

        p_pos_traj = p_traj[..., :4]
        gt_pos_traj = gt_traj[..., :4]

        p_traj_worldmat = relative_trajvec2worldmat(p_pos_traj)[..., 1:, :, :]
        gt_traj_worldmat = relative_trajvec2worldmat(gt_pos_traj)[..., 1:, :, :]
        p_abs_traj_vec = matrix.mat2vec_batch(p_traj_worldmat)
        p_pos_traj = matrix.project_vec(p_abs_traj_vec)  # [b, l, 4]
        gt_abs_traj_vec = matrix.mat2vec_batch(gt_traj_worldmat)
        gt_pos_traj = matrix.project_vec(gt_abs_traj_vec)  # [b, l, 4]

        p_win_traj = denormalize(p_win_traj, win_traj_mean, win_traj_std)
        gt_win_traj = denormalize(gt_win_traj, win_traj_mean, win_traj_std)

        p_pos_traj[~mask] = 0.0
        gt_pos_traj[~mask] = 0.0
        p_traj[~mask] = 0.0
        gt_traj[~mask] = 0.0
        p_win_traj[~mask] = 0.0
        gt_win_traj[~mask] = 0.0

        batch_num = mask.sum(dim=-1)

        # data-wise
        pos_rec_error_l1 = torch.mean(
            (p_pos_traj - gt_pos_traj).abs().sum(dim=-1).sum(dim=-1) / batch_num
        )

        win_rec_error_l1 = torch.mean(
            (p_win_traj - gt_win_traj).abs().sum(dim=-1).sum(dim=-1) / batch_num
        )
        errors = {
            "pos_rec_l1": pos_rec_error_l1.item(),
            "pos_rec_l2_nonorm": pos_rec_error_l2_nonorm,
            "win_rec_l1": win_rec_error_l1.item(),
            "win_rec_l2_nonorm": win_rec_error_l2_nonorm,
        }
        if self.cfg.is_pred_absolute_position:
            pos_abs_rec_error_l1 = torch.mean(
                (p_traj[..., 4:8] - gt_traj[..., 4:8]).abs().sum(dim=-1).sum(dim=-1)
                / batch_num
            )
            errors["pos_abs_rec_l1"] = pos_abs_rec_error_l1.item()
        if self.cfg.is_pred_invtraj:
            pos_abs_inv_rec_error_l1 = torch.mean(
                (p_traj[..., 8:12] - gt_traj[..., 8:12]).abs().sum(dim=-1).sum(dim=-1)
                / batch_num
            )
            errors["pos_abs_inv_rec_l1"] = pos_abs_inv_rec_error_l1.item()

        if "mu" in pred.keys():
            errors["kld"] = self._compute_kld(pred)
        return errors, (pred, gt)

    def _compute_process_result(self, process_output, split, *args, **kwargs):
        """_summary_
            Compute error of different kinds like interaction or idle
        B = 1

        Args:
            process_output (tuple(dict)):
                "key" (tensor): [B, C]

        Returns:
            errors (dict):
                "nll" (float)
                ... (float)
        """
        if "cls" not in process_output[0][0].keys():
            return {}, None
        logits = []
        pred_cls = []
        targets = []
        for pred, gt in process_output:
            logits.append(pred["t"])
            pred_cls.append(pred["cls"])
            targets.append(gt["lengths"])
        logits = torch.cat(logits, dim=0)
        pred_cls = torch.cat(pred_cls, dim=0)
        targets = torch.cat(targets, dim=0)

        # Calculate nll
        nll_total = cross_entropy(logits, targets)

        # Calculate acc
        acc_total = accuracy(pred_cls, targets)

        interval = (pred_cls - targets).abs().float().mean()
        interval_std = (pred_cls - targets).abs().float().std()

        errors = {
            "nll_total": nll_total.item(),
            "acc_total": acc_total.item(),
            "abs_interval": interval.item(),
            "abs_interval_std": interval_std.item(),
        }

        return errors, None
