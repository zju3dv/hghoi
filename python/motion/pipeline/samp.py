import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline import SLPipeline
from motion.utils.visualize import plot_code_histogram, vis_code_pca
from motion.utils.vq_vae import save_motioncode
from motion.utils.basics import DictwithEmpty

from .builder import PIPELINES


@PIPELINES.register_module()
class QContactPosePipeline(SLPipeline):
    def __init__(self, cfg, model, optimizer, *args, **kwargs):
        super().__init__(cfg, model, optimizer, *args, **kwargs)

    def output_loss(self, data, pred):
        loss_pairs = {
            "reconstruction_pose": [pred["y_hat"], data["y"]],
            "reconstruction_contact": [pred["contact"], data["contact"]],
            "latent": [pred["diff"]],
        }
        return self.calculate_loss(loss_pairs)

    def extract_code(
        self, results, cfg, split, vis, dataloader, dataset_save_dir, *args, **kwargs
    ):
        xs = []
        output_codes = DictwithEmpty([])
        quants = DictwithEmpty([])
        for res, data in results:
            for k, v in res["ind"].items():
                output_codes[k].append(v)
            for k, v in res["quant"].items():
                quants[k].append(v)  # [1, C]
        if vis:
            plot_code_histogram(output_codes)
        save_motioncode(
            xs,
            output_codes,
            quants,
            data.get("statistics", None),
            dataset_save_dir,
            split,
            *args,
            **kwargs,
        )


@PIPELINES.register_module()
class KeyCodeNetPipeline(SLPipeline):
    def __init__(self, cfg, model, optimizer, *args, **kwargs):
        super().__init__(cfg, model, optimizer, *args, **kwargs)

    def output_loss(self, data, pred):
        logits = pred["logits"]
        loss_pairs = {}
        for k, v in logits.items():
            c = v.shape[-1]
            loss_pairs[f"nll_{k}"] = [v.reshape(-1, c), data["code"][k].reshape(-1)]
        return self.calculate_loss(loss_pairs)


@PIPELINES.register_module()
class DDPMPipeline(SLPipeline):
    def output_loss(self, data, pred):
        loss_type = self.cfg.used_loss
        mask = data["mask"]
        y_hat = pred["y_hat"]
        y = pred["target"][mask]
        y_hat = y_hat[mask]

        # Predict mask=1 data
        loss_pairs = {
            f"{loss_type}_motion": [y_hat, y, pred["loss_w"][mask]],
        }
        return self.calculate_loss(loss_pairs)

    def _forward_step(self, data):
        output = self.model.sample(data)
        return output


@PIPELINES.register_module()
class DDPMTrajMilestonePipeline(DDPMPipeline):
    def output_loss(self, data, pred):
        loss_type = self.cfg.used_loss
        mask = data["mask"]
        y_hat = pred["pred_y_hat"]
        y = pred["target_y_hat"][mask]
        y_hat = y_hat[mask]

        t = data["lengths"]
        pred_t = pred["t"]

        gt_traj = pred["target_traj"]
        pred_traj = pred["pred_traj"]
        # Predict mask=1 data
        loss_pairs = {
            f"{loss_type}_state": [y_hat, y, pred["loss_w"][mask]],
            f"{loss_type}_traj": [pred_traj[mask], gt_traj[mask], pred["loss_w"][mask]],
            "nll": [pred_t, t],
        }
        return self.calculate_loss(loss_pairs)


@PIPELINES.register_module()
class DDPMTrajPipeline(DDPMPipeline):
    def output_loss(self, data, pred):
        loss_type = self.cfg.used_loss
        mask = data["mask"]
        y_hat = pred["pred_y_hat"]
        y = pred["target_y_hat"][mask]
        y_hat = y_hat[mask]

        gt_traj = pred["target_traj"]
        pred_traj = pred["pred_traj"]
        # Predict mask=1 data
        loss_pairs = {
            f"{loss_type}_state": [y_hat, y, pred["loss_w"][mask]],
            f"{loss_type}_traj": [pred_traj[mask], gt_traj[mask], pred["loss_w"][mask]],
        }
        return self.calculate_loss(loss_pairs)
