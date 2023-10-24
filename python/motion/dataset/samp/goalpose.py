import copy
import os
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .utils import load_norm_data, load_norm_data_prefix, load_minmax_data_prefix

import motion.utils.utils as utils
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy
import motion.utils.matrix as matrix
from motion.utils.quaternions import Quaternions
import smplx
from motion.dataset.dataset import BaseDataset
from motion.dataset.builder import DATASETS


@DATASETS.register_module()
class GoalPoseData(BaseDataset):
    def _load_cfg(self, cfg):
        self.data_dir = cfg.data_dir
        self.division = int(cfg.division)
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
        print(f"Load data from {self.data_dir}")

        if osp.exists(os.path.join(self.data_dir, self.split, "Goalpose.txt")):
            self.input_data = pd_load(
                osp.join(self.data_dir, self.split, "Goalpose.txt")
            ).to_numpy()
        else:
            self.input_data = pd_load(
                osp.join(self.data_dir, self.split, "Input.txt")
            ).to_numpy()
        self.sequences = pd_load(
            osp.join(self.data_dir, self.split, "Sequences.txt")
        ).to_numpy()[:, 0]

        input_mean, input_std, _, _ = load_norm_data(self.data_dir)
        self.input_mean = to_tensor(input_mean)
        self.input_std = to_tensor(input_std)
        self.input_data = to_tensor(self.input_data)
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

    def _preprocess_samp(self):
        Sit_label = 3
        LieDown_label = 4
        seq_num = int(self.sequences.max())
        style_data = [
            self.input_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        input_data_seq = [
            self.input_data[self.sequences == i + 1] for i in range(seq_num)
        ]

        interaction_flag = []
        for data in style_data:
            if data[:, LieDown_label].sum() > 0:
                interaction_flag.append(data[:, LieDown_label] == 1)
            else:
                interaction_flag.append(data[:, Sit_label] == 1)

        filter_input_data = []

        for i, flag in enumerate(interaction_flag):
            if flag.sum() > 0:
                ind_first = flag.nonzero()[0]
                length = flag.sum()
                division_len = length // self.division
                filter_input_data.append(
                    input_data_seq[i][
                        ind_first + division_len : ind_first + length - division_len
                    ]
                )
        self.input_data = torch.cat(filter_input_data, dim=0)

    def _preprocess_nsm(self):
        Sit_label = 5
        seq_num = int(self.sequences.max())
        style_data = [
            self.input_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        input_data_seq = [
            self.input_data[self.sequences == i + 1] for i in range(seq_num)
        ]

        interaction_flag = []
        for data in style_data:
            interaction_flag.append(data[:, Sit_label] == 1)

        filter_input_data = []

        for i, flag in enumerate(interaction_flag):
            if flag.sum() > 0:
                inds = flag.nonzero()
                ind_first = inds[0]
                ind_last = inds[-1]
                length = flag.sum()
                if ind_last - ind_first + 1 != length:
                    last_ind = inds[0]
                    for j in range(len(inds) - 1):
                        if inds[j] + 1 != inds[j + 1]:
                            inter_length = (inds[j] + 1 - last_ind) // self.division
                            pose_data = input_data_seq[i][
                                last_ind + inter_length : inds[j] - inter_length
                            ]
                            filter_input_data.append(pose_data)
                            last_ind = inds[j + 1]
                    inter_length = (inds[-1] + 1 - last_ind) // self.division
                    pose_data = input_data_seq[i][
                        last_ind + inter_length : inds[j] - inter_length
                    ]
                    filter_input_data.append(pose_data)

                else:
                    division_len = length // self.division
                    filter_input_data.append(
                        input_data_seq[i][
                            ind_first + division_len : ind_first + length - division_len
                        ]
                    )
        self.input_data = torch.cat(filter_input_data, dim=0)

    def _preprocess_couch(self):
        Sit_label = 2
        seq_num = int(self.sequences.max())
        style_data = [
            self.input_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        input_data_seq = [
            self.input_data[self.sequences == i + 1] for i in range(seq_num)
        ]

        interaction_flag = []
        for data in style_data:
            interaction_flag.append(data[:, Sit_label] == 1)

        filter_input_data = []

        for i, flag in enumerate(interaction_flag):
            if flag.sum() > 0:
                inds = flag.nonzero()
                ind_first = inds[0]
                ind_last = inds[-1]
                length = flag.sum()
                if ind_last - ind_first + 1 != length:
                    last_ind = inds[0]
                    for j in range(len(inds) - 1):
                        if inds[j] + 1 != inds[j + 1]:
                            inter_length = (inds[j] + 1 - last_ind) // self.division
                            pose_data = input_data_seq[i][
                                last_ind + inter_length : inds[j] - inter_length
                            ]
                            filter_input_data.append(pose_data)
                            last_ind = inds[j + 1]
                    inter_length = (inds[-1] + 1 - last_ind) // self.division
                    pose_data = input_data_seq[i][
                        last_ind + inter_length : inds[j] - inter_length
                    ]
                    filter_input_data.append(pose_data)

                else:
                    division_len = length // self.division
                    filter_input_data.append(
                        input_data_seq[i][
                            ind_first + division_len : ind_first + length - division_len
                        ]
                    )
        self.input_data = torch.cat(filter_input_data, dim=0)

    def _preprocess_dataset(self):
        if not osp.exists(os.path.join(self.data_dir, self.split, "Goalpose.txt")):
            if "samp" in self.data_dir:
                self._preprocess_samp()
            elif "nsm" in self.data_dir:
                self._preprocess_nsm()
            elif "couch" in self.data_dir:
                self._preprocess_couch()
            else:
                raise ValueError(f"do not support {self.data_dir}")
            np.savetxt(
                os.path.join(self.data_dir, self.split, "Goalpose.txt"),
                to_cpu_numpy(self.input_data),
            )
        print(f"The number of interaction data is {len(self.input_data)}")

    def _load_data_instance(self, idx):
        data = self.input_data[idx]
        return data

    def _load_statistics(self, L=1):
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "pose_mean": self.input_mean[: self.pose_dim],
            "pose_std": self.input_std[: self.pose_dim],
        }

    def _normalize_data(self, data):
        x = utils.normalize(data, self.input_mean, self.input_std)
        return x

    def _process_data(self, data):
        x = data
        pose = x[: self.pose_dim]  # pose
        contact = x[self.end_traj : self.start_traj + self.traj_dim]
        action = x[self.start_style_now : self.end_style_now]
        I = x[-self.interaction_dim :]

        return {"I": I, "y": pose, "contact": contact, "action": action}

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        data = self._load_data_instance(idx)
        data = self._normalize_data(data)
        data = self._process_data(data)
        data.update(self._load_statistics())
        return data

    def preprocess_data_socket(self, data):
        """_summary_

        Args:
            data (np.array): [C]

        Returns:
            dict (tensor):
                all: [1, c]
        """
        data = to_tensor(data.reshape(-1))  # [1, 5 + self.interaction_dim]
        action = data[: self.style_dim]
        action = utils.normalize(
            action,
            self.input_mean[self.start_style_now : self.end_style_now],
            self.input_std[self.start_style_now : self.end_style_now],
        )
        I = data[-self.interaction_dim :]
        I = utils.normalize(
            I,
            self.input_mean[-self.interaction_dim :],
            self.input_std[-self.interaction_dim :],
        )
        data = {"I": I[None], "action": action[None]}
        return data

    def postprocess_data_socket(self, result, data):
        """_summary_

        Args:
            result (dict):
                'traj': [1, L, C]

        Returns:
            result (dict):
                'traj_pos': [1, L, c1]
                'traj_window': [1, L, c2]
        """
        pose = result["y_hat"]
        if "contact" in result.keys():
            contact = result["contact"]
        else:
            contact = None
        if self.is_minmax:
            if contact is not None:
                contact = utils.unnormalize_to_zero_to_one(
                    contact, self.contact_min[None], self.contact_max[None]
                )

            pose = utils.unnormalize_to_zero_to_one(
                pose, self.pose_min[None], self.pose_max[None]
            )
        pose = utils.denormalize(
            pose,
            self.input_mean[None, : self.pose_dim],
            self.input_std[None, : self.pose_dim],
        )
        result["pose"] = pose
        if contact is not None:
            contact = utils.denormalize(
                contact,
                self.input_mean[None, self.end_traj : self.start_traj + self.traj_dim],
                self.input_std[None, self.end_traj : self.start_traj + self.traj_dim],
            )
            result["contact"] = contact
        result = to_cpu_numpy(result)
        return result


@DATASETS.register_module()
class GoalPoseEnvData(GoalPoseData):
    def _load_statistics(self, L=1):
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "pose_mean": self.pose_mean,
            "pose_std": self.pose_std,
            "pose_max": self.pose_max,
            "pose_min": self.pose_min,
            "contact_mean": self.contact_mean,
            "contact_std": self.contact_std,
            "contact_max": self.contact_max,
            "contact_min": self.contact_min,
        }

    def _process_data(self, data):
        x = data
        pose = x[: self.pose_dim]  # pose
        contact = x[self.end_traj : self.start_traj + self.traj_dim]
        action = x[self.start_style_now : self.end_style_now]
        env = x[-self.interaction_dim - self.env_dim : -self.interaction_dim]
        if self.is_minmax:
            pose = utils.normalize_to_neg_one_to_one(pose, self.pose_min, self.pose_max)
            contact = utils.normalize_to_neg_one_to_one(
                contact, self.contact_min, self.contact_max
            )

        return {"env": env, "y": pose, "contact": contact, "action": action}

    def preprocess_data_socket(self, data):
        """_summary_

        Args:
            data (np.array): [C]

        Returns:
            dict (tensor):
                all: [1, c]
        """
        data = to_tensor(
            data.reshape(-1)
        )  # [1, 5 + self.env_dim + self.env_dim + self.interaction_dim]
        action = data[: self.style_dim]
        action = utils.normalize(
            action,
            self.input_mean[self.start_style_now : self.end_style_now],
            self.input_std[self.start_style_now : self.end_style_now],
        )
        env = data[-self.interaction_dim - self.env_dim : -self.interaction_dim]

        I = data[-self.interaction_dim :]
        env = utils.normalize(
            env,
            self.input_mean[
                -self.interaction_dim - self.env_dim : -self.interaction_dim
            ],
            self.input_std[
                -self.interaction_dim - self.env_dim : -self.interaction_dim
            ],
        )
        I = utils.normalize(
            I,
            self.input_mean[-self.interaction_dim :],
            self.input_std[-self.interaction_dim :],
        )

        data = {"env": env[None], "action": action[None], "I": I[None]}
        return data


@DATASETS.register_module()
class GoalPoseEnvCodeData(GoalPoseEnvData):
    def _load_data(self):
        super()._load_data()
        code_list = [
            c for c in os.listdir(osp.join(self.data_dir, self.split)) if "Code" in c
        ]
        code_datas = {}
        for c in code_list:
            code_data = pd_load(osp.join(self.data_dir, self.split, c)).to_numpy()
            code_name = (
                c.split(".")[0].lower().split("_")[1]
            )  # remove '.txt' and "code_"
            code_datas[code_name] = to_tensor(code_data, torch.int64)
        self.code_data = code_datas

    def _load_data_instance(self, idx):
        x = super()._load_data_instance(idx)
        code_data = {k: v[idx] for k, v in self.code_data.items()}
        return x, code_data

    def _normalize_data(self, data):
        x, code_data = data
        x = utils.normalize(x, self.input_mean, self.input_std)
        return x, code_data

    def _process_data(self, data):
        x, code = data
        pose = x[: self.pose_dim]  # pose
        contact = x[self.end_traj : self.start_traj + self.traj_dim]
        action = x[self.start_style_now : self.end_style_now]
        env = x[-self.interaction_dim - self.env_dim : -self.interaction_dim]

        data = {
            "env": env,
            "y": pose,
            "contact": contact,
            "action": action,
            "code": code,
        }
        return data
