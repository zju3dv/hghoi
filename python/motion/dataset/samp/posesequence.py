import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from motion.utils.tensor import lengths_to_mask
from .utils import load_norm_data, load_norm_data_prefix, load_minmax_data_prefix
from motion.utils.utils import pd_load, to_tensor, pd_load2numpy, to_cpu_numpy

import motion.utils.utils as utils
from motion.dataset.dataset import BaseDataset
from motion.dataset.builder import DATASETS

from motion.utils.traj import gaussian_motion_samp
from motion.utils.matrix import (
    identity_mat,
    get_mat_BtoA,
    vec2mat,
    vec2mat_batch,
    mat2vec,
    mat2vec_batch,
    project_vec,
    get_position_from,
    get_relative_position_to,
    xzvec2mat,
)


@DATASETS.register_module()
class PoseSequencePositionData(BaseDataset):
    def _load_cfg(self, cfg):
        super()._load_cfg(cfg)
        self.data_dir = cfg.data_dir
        self.max_len = cfg.max_len
        self.predict_feature = cfg.predict_feature

        self.is_env = cfg.is_env
        self.predict_extraframes = cfg.predict_extraframes
        self.is_minmax = cfg.is_minmax

        self.pose_dim = cfg.pose_dim
        self.start_traj = cfg.start_traj
        self.end_traj = cfg.end_traj
        self.contact_dim = cfg.contact_dim
        self.env_dim = cfg.env_dim
        self.interaction_dim = cfg.interaction_dim
        self.traj_dim = self.end_traj - self.start_traj + self.contact_dim
        self.start_traj_now = self.start_traj + (self.traj_dim // 13) * 6
        self.start_style_now = self.start_traj + (self.traj_dim // 13) * 6 + 4
        self.end_style_now = self.start_traj + (self.traj_dim // 13) * 7
        self.style_dim = self.end_style_now - self.start_style_now

    def _load_data(self):
        print(f"Load input from {self.data_dir}...")
        # Traj and interaction
        self.input_data = pd_load(
            osp.join(self.data_dir, self.split, "Input.txt")
        ).to_numpy(np.float32)
        self.sequences = pd_load(
            osp.join(self.data_dir, self.split, "Sequences.txt")
        ).to_numpy()[:, 0]
        mask_path = osp.join(self.data_dir, self.split, "Mask.txt")
        if osp.exists(mask_path):
            self.mask = pd_load(mask_path).to_numpy()[:, 0]
        else:
            self.mask = None

        input_mean, input_std, _, _ = load_norm_data(self.data_dir)

        self.input_mean = to_tensor(input_mean, torch.float32)
        self.input_std = to_tensor(input_std, torch.float32)
        self.pose_mean = self.input_mean[: self.pose_dim]
        self.pose_std = self.input_std[: self.pose_dim]

        input_max, input_min = load_minmax_data_prefix(self.data_dir, prefix="Input")
        if not isinstance(input_max, np.ndarray):
            input_max = np.ones_like(input_mean)
            input_min = np.zeros_like(input_mean)
        self.pose_max = to_tensor(input_max[: self.pose_dim], torch.float32)
        self.pose_min = to_tensor(input_min[: self.pose_dim], torch.float32)
        self.pose_max = utils.normalize(self.pose_max, self.pose_mean, self.pose_std)
        self.pose_min = utils.normalize(self.pose_min, self.pose_mean, self.pose_std)
        self.contact_max = to_tensor(
            input_max[self.end_traj : self.start_traj + self.traj_dim]
        )
        self.contact_min = to_tensor(
            input_min[self.end_traj : self.start_traj + self.traj_dim]
        )
        self.contact_mean = self.input_mean[
            self.end_traj : self.start_traj + self.traj_dim
        ]
        self.contact_std = self.input_std[
            self.end_traj : self.start_traj + self.traj_dim
        ]
        self.contact_max = utils.normalize(
            self.contact_max, self.contact_mean, self.contact_std
        )
        self.contact_min = utils.normalize(
            self.contact_min, self.contact_mean, self.contact_std
        )

    def _preprocess_dataset(self):
        print("Start to preprocess data...")
        print(f"Start to expand data with max_len={self.max_len}...")
        seq_data = [
            self.input_data[self.sequences == i + 1]
            for i in range(self.sequences.max())
        ]
        seq_ind = [
            self.sequences[self.sequences == i + 1] for i in range(self.sequences.max())
        ]

        if self.mask is None:
            for i in tqdm(range(len(seq_data))):
                expand1 = seq_data[i][:1].repeat(self.max_len, axis=0)
                expand2 = seq_data[i][-1:].repeat(self.max_len, axis=0)
                seq_data[i] = np.concatenate((expand1, seq_data[i], expand2), axis=0)
                expand_ind = seq_ind[i][:1].repeat(self.max_len * 2, axis=0)
                seq_ind[i] = np.concatenate((seq_ind[i], expand_ind), axis=0)
            self.input_data = np.concatenate(seq_data, axis=0)
            self.sequences = np.concatenate(seq_ind, axis=0)

        print(f"Start to filter data with extraframes={self.predict_extraframes}...")
        valid_inds = []
        for i in range(
            self.predict_extraframes,
            len(self.sequences) - self.max_len - self.predict_extraframes,
        ):
            if (
                self.sequences[i] == self.sequences[i + self.max_len - 1]
                and self.sequences[i] == self.sequences[i - self.predict_extraframes]
                and self.sequences[i + self.max_len - 1]
                == self.sequences[i + self.max_len - 1 + self.predict_extraframes]
            ):
                if self.mask is not None:
                    mask_sum = self.mask[
                        i
                        - self.predict_extraframes : i
                        + self.max_len
                        + self.predict_extraframes
                    ].sum()
                    if mask_sum != self.max_len + 2 * self.predict_extraframes:
                        continue
            valid_inds.append(i)
        self.valid_inds = valid_inds

    def _load_data_instance(self, ind):
        if self.is_valid:
            ind = ind * self.max_len
        idx = self.valid_inds[ind]
        x = to_tensor(
            self.input_data[
                idx
                - self.predict_extraframes : idx
                + self.max_len
                + self.predict_extraframes
            ]
        )
        lengths = self.max_len + 2 * self.predict_extraframes
        sequences = to_tensor(
            self.sequences[
                idx
                - self.predict_extraframes : idx
                + self.max_len
                + self.predict_extraframes
            ],
            torch.int64,
        )
        return x, lengths, sequences

    def _load_statistics(self):
        return {
            "input_mean": self.input_mean.unsqueeze(0),
            "input_std": self.input_std.unsqueeze(0),
            "pose_mean": self.pose_mean.unsqueeze(0),
            "pose_std": self.pose_std.unsqueeze(0),
            "pose_max": self.pose_max.unsqueeze(0),
            "pose_min": self.pose_min.unsqueeze(0),
            "contact_mean": self.contact_mean.unsqueeze(0),
            "contact_std": self.contact_std.unsqueeze(0),
            "contact_max": self.contact_max.unsqueeze(0),
            "contact_min": self.contact_min.unsqueeze(0),
        }

    def _normalize_data(self, data):
        x, lengths, sequences = data
        x = utils.normalize(x, self.input_mean, self.input_std)
        return x, lengths, sequences

    def _process_data(self, data):
        x, lengths, sequences = data
        pose = x[:, : self.pose_dim]
        traj = x[:, self.start_traj : self.start_traj + self.traj_dim]
        I = x[:, -self.interaction_dim :]
        if self.is_env:
            env = x[:, -self.interaction_dim - self.env_dim : -self.interaction_dim]
        static_pose = torch.zeros_like(pose)
        static_pose[0 + self.predict_extraframes] = pose[0 + self.predict_extraframes]
        static_pose[-1 - self.predict_extraframes] = pose[-1 - self.predict_extraframes]
        static_extra_ind = -1
        if self.is_minmax:
            pose = utils.normalize_to_neg_one_to_one(
                pose, self.pose_min[None], self.pose_max[None]
            )
        data = {
            "traj": traj,
            "I": I,
            "pose": pose,
            "y": pose,
            "static_pose": static_pose,
            "static_extra_ind": static_extra_ind,
            "sequences": sequences,
            "lengths": lengths,
        }
        if self.is_env:
            data["env"] = env
        return data

    def __len__(self):
        if self.is_valid:
            return len(self.valid_inds[:: self.max_len])
        return len(self.valid_inds)

    def __getitem__(self, idx):
        data = self._load_data_instance(idx)
        data = self._normalize_data(data)
        data = self._process_data(data)
        data.update(self._load_statistics())
        return data

    def preprocess_data_socket(self, data):
        """_summary_

        Args:
            data (List(np.array)): [[T*C], [N, c]]

        Returns:
            dict (tensor):
                'traj': [B, T, traj_dim]
                'I': [B, T, self.interaction_dim]
                'static_pose': [B, T, pose_dim]
                'mask' (bool): [B, T]
                'lengths' (int): [B]
        """
        data_dim = self.pose_dim + self.traj_dim + self.interaction_dim
        if self.is_env:
            data_dim += self.env_dim
        data = data.reshape(-1, data_dim)
        data = to_tensor(data)
        N = (len(data) - 1) // (self.max_len - 1)
        split_input = []
        for i in range(N):
            split_data = data[
                i * (self.max_len - 1) : i * (self.max_len - 1) + self.max_len
            ]
            if self.predict_extraframes > 0:
                if i == 0:
                    expand1 = data[:1].repeat(self.predict_extraframes, 1)
                else:
                    expand1 = data[
                        i * (self.max_len - 1)
                        - self.predict_extraframes : i * (self.max_len - 1)
                    ]
                if i == N - 1:
                    expand2 = data[-1:].repeat(self.predict_extraframes, 1)
                else:
                    expand2 = data[
                        i * (self.max_len - 1)
                        + self.max_len : i * (self.max_len - 1)
                        + self.max_len
                        + self.predict_extraframes
                    ]
                split_data = torch.cat((expand1, split_data, expand2), dim=0)
            split_input.append(split_data)

        split_input = pad_sequence(split_input, batch_first=True)
        m = torch.cat(
            (
                self.input_mean[: self.pose_dim],
                self.input_mean[self.start_traj : self.start_traj + self.traj_dim],
                self.input_mean[-self.interaction_dim :],
            )
        )
        std = torch.cat(
            (
                self.input_std[: self.pose_dim],
                self.input_std[self.start_traj : self.start_traj + self.traj_dim],
                self.input_std[-self.interaction_dim :],
            )
        )
        if self.is_env:
            m = torch.cat(
                (
                    m,
                    self.input_mean[
                        -self.interaction_dim - self.env_dim : -self.interaction_dim
                    ],
                )
            )
            std = torch.cat(
                (
                    std,
                    self.input_std[
                        -self.interaction_dim - self.env_dim : -self.interaction_dim
                    ],
                )
            )

        split_input = utils.normalize(
            split_input,
            m.unsqueeze(0).unsqueeze(0),
            std.unsqueeze(0).unsqueeze(0),
        )

        pose = split_input[..., : self.pose_dim]
        static_pose = torch.zeros_like(pose)
        static_pose[..., 0 + self.predict_extraframes, :] = pose[
            ..., 0 + self.predict_extraframes, :
        ]
        static_pose[..., -1 - self.predict_extraframes, :] = pose[
            ..., -1 - self.predict_extraframes, :
        ]
        traj = split_input[..., self.pose_dim : self.pose_dim + self.traj_dim]
        I = split_input[
            ...,
            self.pose_dim
            + self.traj_dim : self.pose_dim
            + self.traj_dim
            + self.interaction_dim,
        ]

        data = {"traj": traj, "I": I, "static_pose": static_pose}
        if self.is_env:
            env = split_input[..., -self.env_dim :]
            data["env"] = env
        return data

    def postprocess_data_socket(self, result, data):
        """_summary_

        Args:
            result (dict):
                'pose': [T, C]

        Returns:
            result (np.array): [state_dim]
        """
        y_hat = result["y_hat"]
        if self.predict_extraframes > 0:
            expand_before = y_hat[1:, : self.predict_extraframes * 2 + 1]
            expand_after = y_hat[:-1, -(self.predict_extraframes * 2 + 1) :]
            weight = torch.linspace(0, 1, self.predict_extraframes * 2 + 1)[
                None, :, None
            ]
            expand_merge = expand_before * weight + expand_after * (1 - weight)
            for i in range(y_hat.shape[0]):
                if i != 0:
                    y_hat[i, : self.predict_extraframes * 2 + 1] = expand_merge[i - 1]
                if i != y_hat.shape[0] - 1:
                    y_hat[i, -(self.predict_extraframes * 2 + 1) :] = expand_merge[i]
            y_hat = y_hat[:, self.predict_extraframes : -self.predict_extraframes]
            assert y_hat.shape[1] == self.max_len
            for i in range(y_hat.shape[0] - 1):
                avg_y = (y_hat[i][-1] + y_hat[i + 1][0]) / 2
                y_hat[i][-1] = avg_y
                y_hat[i + 1][0] = avg_y
            y_hat_nofirst = y_hat[:, 1:].reshape(-1, y_hat.shape[-1])
            y_hat_nofirst = to_cpu_numpy(y_hat_nofirst)
            y_hat = to_cpu_numpy(y_hat)
            y_hat = np.concatenate((y_hat[0, :1], y_hat_nofirst), axis=0)
            y_hat = gaussian_motion_samp(y_hat)
            y_hat = y_hat.reshape(1, -1, y_hat.shape[-1])
            y_hat = to_tensor(y_hat, dtype=torch.float32)

        if self.is_minmax:

            y_hat = utils.unnormalize_to_zero_to_one(
                y_hat, self.pose_min[None, None], self.pose_max[None, None]
            )

        first_frames = self.predict_extraframes if self.predict_extraframes < 10 else 10

        first_pose = data["static_pose"][
            0, self.predict_extraframes : self.predict_extraframes + 1
        ]
        first_pose = first_pose.expand(first_frames + 1, -1)
        weight = torch.linspace(0, 1, first_frames + 1)[:, None]
        y_hat[0, : first_frames + 1] = (
            first_pose * (1 - weight) + y_hat[0, : first_frames + 1] * weight
        )
        last_pose = y_hat[-1, -1:]
        last_pose = last_pose.expand(first_frames + 1, -1)
        y_hat[-1, -(first_frames + 1) :] = last_pose * weight + y_hat[
            -1, -(first_frames + 1)
        ] * (1 - weight)

        result["pose"] = utils.denormalize(
            y_hat, self.pose_mean[None, None], self.pose_std[None, None]
        )

        result = to_cpu_numpy(result)  # [T, pose_dim]
        return result
