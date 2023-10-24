import copy
import os
import time
import os.path as osp
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from .utils import (
    load_norm_data,
    load_norm_data_prefix,
    get_priort,
    load_minmax_data_prefix,
    get_pred_t,
)

import motion.utils.utils as utils
from motion.utils.traj import interpolate_trajpoints_L
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy
from motion.utils.matrix import (
    identity_mat,
    get_mat_BtoA,
    get_mat_BfromA,
    vec2mat,
    vec2mat_batch,
    mat2vec,
    mat2vec_batch,
    project_vec,
    get_position_from,
    get_relative_position_to,
    xzvec2mat,
)
from motion.dataset.dataset import BaseDataset
from motion.dataset.builder import DATASETS


@DATASETS.register_module()
class TrajNetData(BaseDataset):
    def _load_cfg(self, cfg):
        super()._load_cfg(cfg)
        self.data_dir = cfg.data_dir
        self.division = int(cfg.division)

        self.is_pred_absolute_position = cfg.is_pred_absolute_position
        self.is_pred_invtraj = cfg.is_pred_invtraj
        self.pred_time_range = cfg.pred_time_range

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
        self.trajpoint_dim = self.end_style_now - self.start_traj_now

        self.is_goalpose = cfg.is_goalpose
        self.is_env = cfg.is_env
        self.is_minmax = cfg.is_minmax


    def _load_data(self):
        print(f"Load data from {self.data_dir}")
        self.motion_data = pd_load(
            osp.join(self.data_dir, self.split, "Input.txt")
        ).to_numpy(np.float32)
        self.input_data = pd_load(
            osp.join(self.data_dir, self.split, "World.txt")
        ).to_numpy(np.float32)
        self.sequences = pd_load(
            osp.join(self.data_dir, self.split, "Sequences.txt")
        ).to_numpy()[:, 0]

        motion_mean, motion_std, motion_out_mean, motion_out_std = load_norm_data(
            self.data_dir
        )
        world_mean, world_std = load_norm_data_prefix(self.data_dir, prefix="World")

        self.motion_mean = motion_mean
        self.motion_std = motion_std
        self.motion_out_mean = motion_out_mean
        self.motion_out_std = motion_out_std
        self.world_mean = world_mean
        self.world_std = world_std

        traj_mean, traj_std = load_norm_data_prefix(self.data_dir, prefix="Milestone")
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        traj_max, traj_min = load_minmax_data_prefix(self.data_dir, prefix="Milestone")
        self.traj_max = traj_max
        self.traj_min = traj_min
        motion_max, motion_min = load_minmax_data_prefix(self.data_dir, prefix="Input")
        self.motion_max = motion_max
        self.motion_min = motion_min
        self.motion_max = utils.normalize(
            self.motion_max, self.motion_mean, self.motion_std
        )
        self.motion_min = utils.normalize(
            self.motion_min, self.motion_mean, self.motion_std
        )

    def _get_interaction_flag_samp(self):
        LieDown_label = 4
        Sit_label = 3
        seq_num = int(self.sequences.max())
        style_data = [
            self.motion_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        interaction_flag = []
        for data in style_data:
            if data[:, LieDown_label].sum() > 0:
                interaction_flag.append(data[:, LieDown_label] == 1)
            else:
                interaction_flag.append(data[:, Sit_label] == 1)
        return interaction_flag

    def _get_interaction_flag_nsm(self):
        Sit_label = 5
        seq_num = int(self.sequences.max())
        style_data = [
            self.motion_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        interaction_flag = []
        for data in style_data:
            interaction_flag.append(data[:, Sit_label] == 1)
        return interaction_flag

    def _get_interaction_flag_couch(self):
        Sit_label = 2
        seq_num = int(self.sequences.max())
        style_data = [
            self.motion_data[:, self.start_style_now : self.end_style_now][
                self.sequences == i + 1
            ]
            for i in range(seq_num)
        ]
        interaction_flag = []
        for data in style_data:
            interaction_flag.append(data[:, Sit_label] == 1)
        return interaction_flag

    def _preprocess_dataset(self):
        if "samp" in self.data_dir:
            interaction_flag = self._get_interaction_flag_samp()
        elif "nsm" in self.data_dir:
            interaction_flag = self._get_interaction_flag_nsm()
        elif "couch" in self.data_dir:
            interaction_flag = self._get_interaction_flag_couch()
        else:
            raise ValueError(f"do not support {self.data_dir}")

        pred_time = []
        interaction_mask = []

        for i, flag in tqdm(enumerate(interaction_flag)):
            t, mask = get_pred_t(flag, self.pred_time_range, self.division)
            pred_time.append(np.array(t, dtype=np.int64))
            interaction_mask.append(np.array(mask, dtype=np.bool_))
        self.pred_time = np.concatenate(
            pred_time
        )  # contains first frame and last frame
        self.interaction_mask = np.concatenate(interaction_mask)

        self.train_inds = (self.pred_time > 1).nonzero()[0]
        for i in self.train_inds:
            assert (
                self.sequences[i] == self.sequences[i + self.pred_time[i] - 1]
            ), "Not in the same sequence data!"
        print(
            f"The max time range is {self.pred_time.max().item()} and "
            f"the min time range is {self.pred_time[self.pred_time > 1].min().item()}"
        )

    def _load_data_instance(self, idx):
        ind = self.train_inds[idx]
        t = self.pred_time[ind]
        x = self.motion_data[ind : ind + t]
        root = self.input_data[ind : ind + t, :12]
        goal = self.input_data[ind : ind + t, 12:24]
        I = self.input_data[ind : ind + t, 24:]
        mask = self.interaction_mask[ind : ind + t]
        return x, root, goal, I, t, None, mask

    def _load_statistics(self, L=1):
        try:
            return {
                "motion_mean": self.motion_mean[None],
                "motion_std": self.motion_std[None],
                "motion_out_mean": self.motion_out_mean[None],
                "motion_out_std": self.motion_out_std[None],
                "world_mean": self.world_mean[None],
                "world_std": self.world_std[None],
                "trajstate_mean": self.motion_mean[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
                "trajstate_std": self.motion_std[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
                "trajstate_max": self.motion_max[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
                "trajstate_min": self.motion_min[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
                "traj_mean": self.traj_mean[None],
                "traj_std": self.traj_std[None],
                "traj_max": self.traj_max[None],
                "traj_min": self.traj_min[None],
                "motion_max": self.motion_max[None],
                "motion_min": self.motion_min[None],
            }
        except:
            return {
                "motion_mean": self.motion_mean[None],
                "motion_std": self.motion_std[None],
                "motion_out_mean": self.motion_out_mean[None],
                "motion_out_std": self.motion_out_std[None],
                "world_mean": self.world_mean[None],
                "world_std": self.world_std[None],
                "trajstate_mean": self.motion_mean[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
                "trajstate_std": self.motion_std[
                    None, self.start_traj : self.end_traj + self.contact_dim
                ],
            }

    def _normalize_data(self, data):
        x, root, goal, I, t, y, mask = data
        x = utils.normalize(x, self.motion_mean[None], self.motion_std[None])
        I = utils.normalize(
            I,
            self.world_mean[None, -self.interaction_dim :],
            self.world_std[None, -self.interaction_dim :],
        )
        return x, root, goal, I, t, y, mask

    def _get_selected_ind(self, t, mask):
        selected_ind = np.arange(t)
        if not mask[0]:
            predict_len = np.random.randint(2, len(selected_ind) + 1)
            selected_ind = selected_ind[:predict_len]
        selected_ind = np.array(selected_ind, dtype=np.int64)
        return selected_ind

    def _process_data(self, data):
        x, root, goal, ItoG, t, _, mask = data
        selected_ind = self._get_selected_ind(t, mask)

        first_ind = selected_ind[0]
        last_ind = selected_ind[-1]
        x1 = x[first_ind, : self.pose_dim][None]  # first frame pose
        if self.is_goalpose:
            x1 = np.concatenate((x1, x[last_ind, : self.pose_dim][None]), axis=0)
        ItoR = x[first_ind, -self.interaction_dim :][None]  # [1, C]
        if self.is_env:
            envR = x[
                first_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            envG = x[
                last_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            env = np.concatenate((envR, envG), axis=0)
        ItoG = ItoG[first_ind][None]  # [1, C]

        root = root[selected_ind]
        root_mat = vec2mat_batch(root)
        goal2root_mat = get_mat_BtoA(root_mat[:1], root_mat[-1:])
        root2goal_mat = get_mat_BtoA(root_mat[-1:], root_mat[:1])

        traj_mat = get_mat_BtoA(root_mat[:-1], root_mat[1:])
        traj_mat = np.concatenate((traj_mat, identity_mat(is_numpy=True)[None]), axis=0)
        traj_vec = mat2vec_batch(traj_mat)
        traj = project_vec(traj_vec)

        if self.is_pred_absolute_position:
            traj_mat = get_mat_BtoA(root_mat[0], root_mat)
            abs_traj_vec = mat2vec_batch(traj_mat)
            abs_traj = project_vec(abs_traj_vec)  # [l, 4]
            abs_traj = abs_traj
            traj = np.concatenate((traj, abs_traj), axis=-1)
        if self.is_pred_invtraj:
            invtraj_mat = get_mat_BtoA(root_mat[-1], root_mat)
            abs_invtraj_vec = mat2vec_batch(invtraj_mat)
            abs_invtraj = project_vec(abs_invtraj_vec)  # [l, 4]
            abs_invtraj = abs_invtraj
            traj = np.concatenate((traj, abs_invtraj), axis=-1)

        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[:1, :6]  # [1, 6]
        goal = project_vec(goal)
        root = project_vec(root)
        root = root[:1]  # [1, 4]
        style0 = x[first_ind, self.start_style_now : self.end_style_now][None]
        goalstyle = x[last_ind, self.start_style_now : self.end_style_now][None]
        root = np.concatenate((root, style0), axis=-1)
        goal = np.concatenate((goal, goalstyle), axis=-1)

        y = x[
            selected_ind, self.start_traj : self.start_traj + self.traj_dim
        ]  # traj window
        t = len(selected_ind)

        if self.is_minmax:
            try:
                traj = utils.normalize_to_neg_one_to_one(
                    traj, self.traj_min[None], self.traj_max[None]
                )
                y = utils.normalize_to_neg_one_to_one(
                    y,
                    self.motion_min[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                    self.motion_max[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                )
            except Exception as e:
                pass
        data = {
            "pose": x1,
            "ItoR": ItoR,
            "ItoG": ItoG,
            "goal": goal,
            "root": root,
            "lengths": t,
            "traj": traj,
            "y": y,
        }
        if self.is_env:
            data["env"] = env
        return data

    def __len__(self):
        return (self.pred_time > 1).sum().item()

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
        data = to_tensor(
            data.reshape(1, -1)
        )  # [1, 264 + 264 + 2048 + 12 + 12 + 2048 + 315 + 315 + 5 + 5]
        pose_dims = self.pose_dim
        if self.is_goalpose:
            pose_dims = self.pose_dim + self.pose_dim
        pose = data[:, :pose_dims].reshape(-1, self.pose_dim)
        data = data[:, pose_dims:]
        ItoR = data[:, : self.interaction_dim]
        root = data[:, self.interaction_dim : self.interaction_dim + 12]
        goal = data[:, self.interaction_dim + 12 : self.interaction_dim + 24]
        ItoG = data[:, self.interaction_dim + 24 : self.interaction_dim * 2 + 24]
        data_dim = self.interaction_dim * 2 + 24 + 2 * self.style_dim + 2 * self.env_dim
        if data.shape[-1] != data_dim:
            x = data[
                :,
                self.interaction_dim * 2 + 24 : -2 * self.style_dim - 2 * self.env_dim,
            ]
            x = x.reshape(-1, 4)
            Milestone_mat = xzvec2mat(x)
            Milestone_mat_relative = get_mat_BtoA(Milestone_mat[:-1], Milestone_mat[1:])
            Milestone_mat_abs = get_mat_BtoA(Milestone_mat[:1], Milestone_mat)
            Milestone_mat_inv = get_mat_BtoA(Milestone_mat[-1:], Milestone_mat)
            Milestone_mat_relative = torch.cat(
                (Milestone_mat_abs[:1], Milestone_mat_relative), dim=0
            )
            control_x = project_vec(mat2vec_batch(Milestone_mat_relative))
            control_x_abs = project_vec(mat2vec_batch(Milestone_mat_abs))
            control_x_inv = project_vec(mat2vec_batch(Milestone_mat_inv))
            control_x = torch.cat((control_x, control_x_abs, control_x_inv), dim=-1)
            control_x = utils.normalize_to_neg_one_to_one(
                control_x, self.traj_min[None], self.traj_max[None]
            )[None]
            control_t = to_tensor(control_x.shape[1], dtype=torch.long).reshape(1)
        else:
            control_x = None
            control_t = None

        startstyle = data[:, -2 * self.style_dim : -self.style_dim]
        goalstyle = data[:, -self.style_dim :]
        pose = utils.normalize(
            pose,
            self.motion_mean[None, : self.pose_dim],
            self.motion_std[None, : self.pose_dim],
        )
        ItoR = utils.normalize(
            ItoR,
            self.motion_mean[None, -self.interaction_dim :],
            self.motion_std[None, -self.interaction_dim :],
        )
        ItoG = utils.normalize(
            ItoG,
            self.world_mean[None, -self.interaction_dim :],
            self.world_std[None, -self.interaction_dim :],
        )
        if self.is_env:
            envR = data[
                :,
                -self.env_dim * 2
                - self.style_dim * 2 : -self.env_dim
                - self.style_dim * 2,
            ]
            envG = data[:, -self.env_dim - self.style_dim * 2 : -self.style_dim * 2]
            envR = utils.normalize(
                envR,
                self.motion_mean[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
                self.motion_std[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
            )
            envG = utils.normalize(
                envG,
                self.motion_mean[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
                self.motion_std[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
            )
            env = torch.cat((envR, envG), dim=0)
        root_mat = vec2mat_batch(root)
        goal_mat = vec2mat_batch(goal)
        goal2root_mat = get_mat_BtoA(root_mat, goal_mat)
        root2goal_mat = get_mat_BtoA(goal_mat, root_mat)
        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[:1, :6]  # [1, 6]
        goal = project_vec(goal)
        root = project_vec(root)
        root = root[:1]  # [1, 4]
        style0 = utils.normalize(
            startstyle,
            self.motion_mean[None, self.start_style_now : self.end_style_now],
            self.motion_std[None, self.start_style_now : self.end_style_now],
        )
        goalstyle = utils.normalize(
            goalstyle,
            self.motion_mean[None, self.start_style_now : self.end_style_now],
            self.motion_std[None, self.start_style_now : self.end_style_now],
        )
        root = torch.cat((root, style0), dim=-1)
        goal = torch.cat((goal, goalstyle), dim=-1)
        distance = (goal[..., 0] ** 2 + goal[..., 1] ** 2) ** 0.5
        prior_t = get_priort(distance)

        data = {
            "pose": pose[None],
            "ItoR": ItoR[None],
            "ItoG": ItoG[None],
            "goal": goal[None],
            "root": root[None],
            "prior_t": prior_t,
            "control_x": control_x,
            "control_t": control_t,
        }
        if self.is_env:
            data["env"] = env[None]
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
        traj = result["traj"]
        traj_state = result["state"]
        D = 4
        if self.is_pred_absolute_position:
            D += 4
        if self.is_pred_invtraj:
            D += 4
        traj_pos = traj[..., :D]
        if self.is_minmax:
            traj_pos = utils.unnormalize_to_zero_to_one(
                traj_pos, self.traj_min[None], self.traj_max[None]
            )
            traj_state = utils.unnormalize_to_zero_to_one(
                traj_state,
                self.motion_min[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
                self.motion_max[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
            )

        # force the last state should be the goal
        goalstyle = data["goal"][0, :, -self.style_dim :]
        last_state = traj_state[0, -1, : -self.contact_dim].reshape(13, -1)
        last_state[:, -self.style_dim :] = goalstyle
        traj_state[0, -1, : -self.contact_dim] = last_state.reshape(-1)
        traj_state = utils.denormalize(
            traj_state,
            self.motion_mean[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
            self.motion_std[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
        )

        result["traj_pos"] = traj_pos
        result["traj_window"] = traj_state
        result = to_cpu_numpy(result)
        return result


@DATASETS.register_module()
class TrajMilestoneData(TrajNetData):
    def _load_cfg(self, cfg):
        super()._load_cfg(cfg)
        self.L = cfg.L

    def __len__(self):
        if self.is_valid:
            # return (self.pred_time >= self.L).sum().item() // self.L
            pass
        return (self.pred_time >= self.L).sum().item()

    def _preprocess_dataset(self):
        super()._preprocess_dataset()
        self.train_inds = (self.pred_time >= self.L).nonzero()[0]

    def _load_data_instance(self, idx):
        if self.is_valid:
            # idx = idx * self.L
            pass
        return super()._load_data_instance(idx)

    def _get_selected_ind(self, t, mask):
        selected_ind = np.arange(0, t, self.L - 1)
        if mask[0]:
            # this is interaction sequence
            if (t - 1) not in selected_ind:
                selected_ind = np.concatenate((selected_ind, np.array([t - 1])))
        else:
            predict_len = np.random.randint(2, len(selected_ind) + 1)
            selected_ind = selected_ind[:predict_len]
        return selected_ind


@DATASETS.register_module()
class TrajMilestoneWithPoseData(TrajMilestoneData):
    def _process_data(self, data):
        x, root, goal, ItoG, t, _, mask = data
        selected_ind = self._get_selected_ind(t, mask)

        first_ind = selected_ind[0]
        last_ind = selected_ind[-1]
        x1 = x[first_ind, : self.pose_dim][None]  # first frame pose
        if self.is_goalpose:
            x1 = np.concatenate((x1, x[last_ind, : self.pose_dim][None]), axis=0)
        ItoR = x[first_ind, -self.interaction_dim :][None]  # [1, C]
        if self.is_env:
            envR = x[
                first_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            envG = x[
                last_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            env = np.concatenate((envR, envG), axis=0)
        ItoG = ItoG[first_ind][None]  # [1, C]

        root = root[selected_ind]
        root_mat = vec2mat_batch(root)
        goal2root_mat = get_mat_BtoA(root_mat[:1], root_mat[-1:])
        root2goal_mat = get_mat_BtoA(root_mat[-1:], root_mat[:1])

        traj_mat = get_mat_BtoA(root_mat[:-1], root_mat[1:])
        traj_mat = np.concatenate((traj_mat, identity_mat(is_numpy=True)[None]), axis=0)
        traj_vec = mat2vec_batch(traj_mat)
        traj = project_vec(traj_vec)

        if self.is_pred_absolute_position:
            traj_mat = get_mat_BtoA(root_mat[0], root_mat)
            abs_traj_vec = mat2vec_batch(traj_mat)
            abs_traj = project_vec(abs_traj_vec)  # [l, 4]
            abs_traj = abs_traj
            traj = np.concatenate((traj, abs_traj), axis=-1)
        if self.is_pred_invtraj:
            invtraj_mat = get_mat_BtoA(root_mat[-1], root_mat)
            abs_invtraj_vec = mat2vec_batch(invtraj_mat)
            abs_invtraj = project_vec(abs_invtraj_vec)  # [l, 4]
            abs_invtraj = abs_invtraj
            traj = np.concatenate((traj, abs_invtraj), axis=-1)

        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[:1, :6]  # [1, 6]
        goal = project_vec(goal)
        root = project_vec(root)
        root = root[:1]  # [1, 4]
        style0 = x[first_ind, self.start_style_now : self.end_style_now][None]
        goalstyle = x[last_ind, self.start_style_now : self.end_style_now][None]
        root = np.concatenate((root, style0), axis=-1)
        goal = np.concatenate((goal, goalstyle), axis=-1)

        y = x[
            selected_ind, self.start_traj : self.start_traj + self.traj_dim
        ]  # traj window
        t = len(selected_ind)
        pose_y = x[selected_ind, : self.pose_dim]

        if self.is_minmax:
            try:
                traj = utils.normalize_to_neg_one_to_one(
                    traj, self.traj_min[None], self.traj_max[None]
                )
                y = utils.normalize_to_neg_one_to_one(
                    y,
                    self.motion_min[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                    self.motion_max[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                )
                pose_y = utils.normalize_to_neg_one_to_one(
                    pose_y,
                    self.motion_min[None, : self.pose_dim],
                    self.motion_max[None, : self.pose_dim],
                )
                y = np.concatenate((y, pose_y), axis=-1)
            except Exception as e:
                pass
        data = {
            "pose": x1,
            "ItoR": ItoR,
            "ItoG": ItoG,
            "goal": goal,
            "root": root,
            "lengths": t,
            "traj": traj,
            "y": y,
        }
        if self.is_env:
            data["env"] = env
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
        traj = result["traj"]
        traj_state = result["state"][..., : self.traj_dim]
        traj_pose = result["state"][..., self.traj_dim :]
        D = 4
        if self.is_pred_absolute_position:
            D += 4
        if self.is_pred_invtraj:
            D += 4
        traj_pos = traj[..., :D]
        if self.is_minmax:
            traj_pos = utils.unnormalize_to_zero_to_one(
                traj_pos, self.traj_min[None], self.traj_max[None]
            )
            traj_state = utils.unnormalize_to_zero_to_one(
                traj_state,
                self.motion_min[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
                self.motion_max[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
            )
            traj_pose = utils.unnormalize_to_zero_to_one(
                traj_pose,
                self.motion_min[None, : self.pose_dim],
                self.motion_max[None, : self.pose_dim],
            )

        # force the last state should be the goal
        goalstyle = data["goal"][0, :, -self.style_dim :]
        last_state = traj_state[0, -1, : -self.contact_dim].reshape(13, -1)
        last_state[:, -self.style_dim :] = goalstyle
        traj_state[0, -1, : -self.contact_dim] = last_state.reshape(-1)
        traj_state = utils.denormalize(
            traj_state,
            self.motion_mean[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
            self.motion_std[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
        )
        traj_pose = utils.denormalize(
            traj_pose,
            self.motion_mean[None, None, : self.pose_dim],
            self.motion_std[None, None, : self.pose_dim],
        )

        result["traj_pos"] = traj_pos
        result["traj_window"] = traj_state
        result["pose"] = traj_pose
        result = to_cpu_numpy(result)
        return result


@DATASETS.register_module()
class TrajNetWithPoseData(TrajNetData):
    def _process_data(self, data):
        x, root, goal, ItoG, t, _, mask = data
        selected_ind = self._get_selected_ind(t, mask)

        first_ind = selected_ind[0]
        last_ind = selected_ind[-1]
        x1 = x[first_ind, : self.pose_dim][None]  # first frame pose
        if self.is_goalpose:
            x1 = np.concatenate((x1, x[last_ind, : self.pose_dim][None]), axis=0)
        ItoR = x[first_ind, -self.interaction_dim :][None]  # [1, C]
        if self.is_env:
            envR = x[
                first_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            envG = x[
                last_ind, -self.interaction_dim - self.env_dim : -self.interaction_dim
            ][None]
            env = np.concatenate((envR, envG), axis=0)
        ItoG = ItoG[first_ind][None]  # [1, C]

        root = root[selected_ind]
        root_mat = vec2mat_batch(root)
        goal2root_mat = get_mat_BtoA(root_mat[:1], root_mat[-1:])
        root2goal_mat = get_mat_BtoA(root_mat[-1:], root_mat[:1])

        traj_mat = get_mat_BtoA(root_mat[:-1], root_mat[1:])
        traj_mat = np.concatenate((traj_mat, identity_mat(is_numpy=True)[None]), axis=0)
        traj_vec = mat2vec_batch(traj_mat)
        traj = project_vec(traj_vec)

        if self.is_pred_absolute_position:
            traj_mat = get_mat_BtoA(root_mat[0], root_mat)
            abs_traj_vec = mat2vec_batch(traj_mat)
            abs_traj = project_vec(abs_traj_vec)  # [l, 4]
            abs_traj = abs_traj
            traj = np.concatenate((traj, abs_traj), axis=-1)
        if self.is_pred_invtraj:
            invtraj_mat = get_mat_BtoA(root_mat[-1], root_mat)
            abs_invtraj_vec = mat2vec_batch(invtraj_mat)
            abs_invtraj = project_vec(abs_invtraj_vec)  # [l, 4]
            abs_invtraj = abs_invtraj
            traj = np.concatenate((traj, abs_invtraj), axis=-1)

        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[:1, :6]  # [1, 6]
        goal = project_vec(goal)
        root = project_vec(root)
        root = root[:1]  # [1, 4]
        style0 = x[first_ind, self.start_style_now : self.end_style_now][None]
        goalstyle = x[last_ind, self.start_style_now : self.end_style_now][None]
        root = np.concatenate((root, style0), axis=-1)
        goal = np.concatenate((goal, goalstyle), axis=-1)

        y = x[
            selected_ind, self.start_traj : self.start_traj + self.traj_dim
        ]  # traj window
        t = len(selected_ind)
        pose_y = x[selected_ind, : self.pose_dim]

        if self.is_minmax:
            try:
                traj = utils.normalize_to_neg_one_to_one(
                    traj, self.traj_min[None], self.traj_max[None]
                )
                y = utils.normalize_to_neg_one_to_one(
                    y,
                    self.motion_min[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                    self.motion_max[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                )
                pose_y = utils.normalize_to_neg_one_to_one(
                    pose_y,
                    self.motion_min[None, : self.pose_dim],
                    self.motion_max[None, : self.pose_dim],
                )
                y = np.concatenate((y, pose_y), axis=-1)
            except Exception as e:
                pass
        data = {
            "pose": x1,
            "ItoR": ItoR,
            "ItoG": ItoG,
            "goal": goal,
            "root": root,
            "lengths": t,
            "traj": traj,
            "y": y,
        }
        if self.is_env:
            data["env"] = env
        return data


@DATASETS.register_module()
class TrajCompletionData(TrajMilestoneData):
    def _load_cfg(self, cfg):
        super()._load_cfg(cfg)
        self.predict_extraframes = cfg.predict_extraframes

    def _load_data(self):
        super()._load_data()
        prefix = "Traj"
        traj_mean, traj_std = load_norm_data_prefix(self.data_dir, prefix=prefix)
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        traj_max, traj_min = load_minmax_data_prefix(self.data_dir, prefix=prefix)
        self.traj_max = traj_max
        self.traj_min = traj_min

        mask_path = osp.join(self.data_dir, self.split, "Mask.txt")
        if osp.exists(mask_path):
            self.mask = pd_load(mask_path).to_numpy()[:, 0]
        else:
            self.mask = None

    def _preprocess_dataset(self):
        print("Start to preprocess data...")
        if self.predict_extraframes > 0:
            print(f"Start to expand data with max_len={self.L}...")
            seq_data = [
                self.input_data[self.sequences == i + 1]
                for i in range(self.sequences.max())
            ]
            seq_motion_data = [
                self.motion_data[self.sequences == i + 1]
                for i in range(self.sequences.max())
            ]

            seq_ind = [
                self.sequences[self.sequences == i + 1]
                for i in range(self.sequences.max())
            ]
            for i in tqdm(range(len(seq_data))):
                expand1 = seq_data[i][:1].repeat(self.L, axis=0)
                expand2 = seq_data[i][-1:].repeat(self.L, axis=0)
                seq_data[i] = np.concatenate((expand1, seq_data[i], expand2), axis=0)
                expand_ind = seq_ind[i][:1].repeat(self.L * 2, axis=0)
                seq_ind[i] = np.concatenate((seq_ind[i], expand_ind), axis=0)

                expand1 = seq_motion_data[i][:1].repeat(self.L, axis=0)
                expand2 = seq_motion_data[i][-1:].repeat(self.L, axis=0)
                seq_motion_data[i] = np.concatenate(
                    (expand1, seq_motion_data[i], expand2), axis=0
                )
            self.input_data = np.concatenate(seq_data, axis=0)
            self.motion_data = np.concatenate(seq_motion_data, axis=0)
            self.sequences = np.concatenate(seq_ind, axis=0)

        print(f"Start to filter data with extraframes={self.predict_extraframes}...")
        valid_inds = []
        for i in range(
            self.predict_extraframes,
            len(self.sequences) - self.L - self.predict_extraframes,
        ):
            if (
                self.sequences[i] == self.sequences[i + self.L - 1]
                and self.sequences[i] == self.sequences[i - self.predict_extraframes]
                and self.sequences[i + self.L - 1]
                == self.sequences[i + self.L - 1 + self.predict_extraframes]
            ):
                if self.mask is not None:
                    mask_sum = self.mask[
                        i
                        - self.predict_extraframes : i
                        + self.L
                        + self.predict_extraframes
                    ].sum()
                    if mask_sum != self.L + 2 * self.predict_extraframes:
                        continue
                valid_inds.append(i)
        self.valid_inds = valid_inds
        self._frames_num = len(self.valid_inds)

    def _load_data_instance(self, ind):
        if self.is_valid:
            ind = ind * self.L
        idx = self.valid_inds[ind]
        start = idx - self.predict_extraframes
        end = idx + self.L + self.predict_extraframes
        x = self.motion_data[start:end]
        root = self.input_data[start:end, :12]
        goal = self.input_data[start:end, 12:24]
        I = self.input_data[start:end, 24:]
        if idx < self.L - 1 or self.sequences[idx] != self.sequences[idx - self.L + 1]:
            pre_ind = idx
        else:
            pre_ind = idx - self.L + 1
        if (
            idx > self._frames_num - self.L
            or self.sequences[idx] != self.sequences[idx + self.L - 1]
        ):
            aft_ind = idx
        else:
            aft_ind = idx + self.L - 1

        return x, root, goal, I, self.L + 2 * self.predict_extraframes, pre_ind, aft_ind

    def _process_data(self, data):
        x, root, _, _, t, pre_ind, aft_ind = data
        ItoR = x[:1, -self.interaction_dim :]  # [1, C]
        ItoG = x[-1:, -self.interaction_dim :]  # [1, C]
        if self.is_env:
            envR = x[:1, -self.interaction_dim - self.env_dim : -self.interaction_dim]
            envG = x[-1:, -self.interaction_dim - self.env_dim : -self.interaction_dim]
            env = np.concatenate((envR, envG), axis=0)

        root_mat = vec2mat_batch(root)
        goal2root_mat = get_mat_BtoA(root_mat[0], root_mat[-1])
        root2goal_mat = get_mat_BtoA(root_mat[-1], root_mat[0])

        traj_mat = get_mat_BtoA(root_mat[:-1], root_mat[1:])
        traj_mat = np.concatenate((traj_mat, identity_mat(is_numpy=True)[None]), axis=0)
        traj_vec = mat2vec_batch(traj_mat)
        traj = project_vec(traj_vec)

        if self.is_pred_absolute_position:
            traj_mat = get_mat_BtoA(root_mat[0], root_mat)
            abs_traj_vec = mat2vec_batch(traj_mat)
            abs_traj = project_vec(abs_traj_vec)  # [l, 4]
            traj = np.concatenate((traj, abs_traj), axis=-1)
        if self.is_pred_invtraj:
            invtraj_mat = get_mat_BtoA(root_mat[-1], root_mat)
            abs_invtraj_vec = mat2vec_batch(invtraj_mat)
            abs_invtraj = project_vec(abs_invtraj_vec)  # [l, 4]
            traj = np.concatenate((traj, abs_invtraj), axis=-1)
        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[None, :6]  # [1, 6]
        root = root[None, :6]
        goal = project_vec(goal)
        root = project_vec(root)
        style0 = x[:1, self.start_style_now : self.end_style_now]
        goalstyle = x[-1:, self.start_style_now : self.end_style_now]
        root = np.concatenate((root, style0), axis=-1)
        goal = np.concatenate((goal, goalstyle), axis=-1)

        y = x[:, self.start_traj : self.start_traj + self.traj_dim]  # traj window
        start_state = x[:1, self.start_traj : self.start_traj + self.traj_dim]
        end_state = x[-1:, self.start_traj : self.start_traj + self.traj_dim]

        if self.is_minmax:
            try:
                traj = utils.normalize_to_neg_one_to_one(
                    traj, self.traj_min[None], self.traj_max[None]
                )
                y = utils.normalize_to_neg_one_to_one(
                    y,
                    self.motion_min[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                    self.motion_max[
                        None, self.start_traj : self.start_traj + self.traj_dim
                    ],
                )
            except Exception as e:
                pass
        data = {
            "start_state": start_state,
            "end_state": end_state,
            "ItoR": ItoR,
            "ItoG": ItoG,
            "goal": goal,
            "root": root,
            "lengths": t,
            "traj": traj,
            "y": y,
        }
        if self.is_env:
            data["env"] = env
        return data

    def __len__(self):
        if self.is_valid:
            return len(self.valid_inds[:: self.L])
        return len(self.valid_inds)

    def preprocess_data_socket(self, data):
        """_summary_

        Args:
            data (np.array): [C]

        Returns:
            dict (tensor):
                all: [1, c]
        """

        state_dim = self.traj_dim
        D = state_dim + self.interaction_dim + 12
        if self.is_env:
            D += self.env_dim
        data = to_tensor(
            data.reshape(-1, D)
        )  # [N, 122 + self.interaction_dim + 12 + self.env_dim]

        start_state = data[:-1, None, : self.traj_dim]
        end_state = data[1:, None, : self.traj_dim]
        state_mean = self.motion_mean[
            None, None, self.start_traj : self.start_traj + self.traj_dim
        ]
        state_std = self.motion_std[
            None, None, self.start_traj : self.start_traj + self.traj_dim
        ]

        start_state = utils.normalize(start_state, state_mean, state_std)
        end_state = utils.normalize(end_state, state_mean, state_std)
        ItoR = data[:-1, None, state_dim : state_dim + self.interaction_dim]
        ItoG = data[1:, None, state_dim : state_dim + self.interaction_dim]

        if self.is_env:
            envR = data[:-1, None, -self.env_dim :]
            envG = data[1:, None, -self.env_dim :]
            envR = utils.normalize(
                envR,
                self.motion_mean[
                    None,
                    None,
                    -self.interaction_dim - self.env_dim : -self.interaction_dim,
                ],
                self.motion_std[
                    None,
                    None,
                    -self.interaction_dim - self.env_dim : -self.interaction_dim,
                ],
            )
            envG = utils.normalize(
                envG,
                self.motion_mean[
                    None,
                    None,
                    -self.interaction_dim - self.env_dim : -self.interaction_dim,
                ],
                self.motion_std[
                    None,
                    None,
                    -self.interaction_dim - self.env_dim : -self.interaction_dim,
                ],
            )
            env = torch.cat((envR, envG), dim=1)

        root_mat = vec2mat_batch(
            data[
                :,
                state_dim
                + self.interaction_dim : state_dim
                + self.interaction_dim
                + 12,
            ]
        )
        goal2root_mat = get_mat_BtoA(root_mat[:-1], root_mat[1:])
        root2goal_mat = get_mat_BtoA(root_mat[1:], root_mat[:-1])

        ItoR = utils.normalize(
            ItoR,
            self.motion_mean[None, -self.interaction_dim :],
            self.motion_std[None, -self.interaction_dim :],
        )
        ItoG = utils.normalize(
            ItoG,
            self.motion_mean[None, -self.interaction_dim :],
            self.motion_std[None, -self.interaction_dim :],
        )

        goal = mat2vec_batch(goal2root_mat)
        root = mat2vec_batch(root2goal_mat)
        goal = goal[:, None, :6]  # [N, 1, 6]
        goal = project_vec(goal)
        root = project_vec(root)
        root = root[:, None]  # [N, 1, 4]
        if state_dim == self.style_dim:
            start_style = start_state
            end_style = end_state
        else:
            start_style = start_state[
                ..., 6 * self.trajpoint_dim + 4 : 7 * self.trajpoint_dim
            ]
            end_style = end_state[
                ..., 6 * self.trajpoint_dim + 4 : 7 * self.trajpoint_dim
            ]
        root = torch.cat(
            (root, start_style),
            dim=-1,
        )
        goal = torch.cat(
            (goal, end_style),
            dim=-1,
        )
        data = {
            "start_state": start_state,
            "end_state": end_state,
            "ItoR": ItoR,
            "ItoG": ItoG,
            "goal": goal,
            "root": root,
        }
        if self.is_env:
            data["env"] = env
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
        traj = result["traj"]
        traj_state = result["y_hat"]
        D = 4
        if self.is_pred_absolute_position:
            D += 4
        if self.is_pred_invtraj:
            D += 4
        traj_pos = traj[..., :D]

        if self.is_minmax:
            traj_pos = utils.unnormalize_to_zero_to_one(
                traj_pos, self.traj_min[None], self.traj_max[None]
            )
            traj_state = utils.unnormalize_to_zero_to_one(
                traj_state,
                self.motion_min[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
                self.motion_max[
                    None, self.start_traj : self.start_traj + self.traj_dim
                ],
            )

        traj_state = utils.denormalize(
            traj_state,
            self.motion_mean[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
            self.motion_std[
                None, None, self.start_traj : self.start_traj + self.traj_dim
            ],
        )
        result["traj_pos"] = traj_pos
        result["traj_window"] = traj_state
        result = to_cpu_numpy(result)
        return result


@DATASETS.register_module()
class PoseMilestoneData(TrajMilestoneData):
    def _load_statistics(self, L=1):
        return {
            "motion_mean": self.motion_mean[None],
            "motion_std": self.motion_std[None],
            "motion_out_mean": self.motion_out_mean[None],
            "motion_out_std": self.motion_out_std[None],
            "world_mean": self.world_mean[None],
            "world_std": self.world_std[None],
            "pose_mean": self.motion_mean[None, : self.pose_dim],
            "pose_std": self.motion_std[None, : self.pose_dim],
            "pose_max": self.motion_max[None, : self.pose_dim],
            "pose_min": self.motion_min[None, : self.pose_dim],
        }

    def _process_data(self, data):
        x, _, _, _, t, _, mask = data
        selected_ind = self._get_selected_ind(t, mask)

        first_ind = selected_ind[0]
        last_ind = selected_ind[-1]
        x1 = x[first_ind, : self.pose_dim][None]  # first frame pose
        if self.is_goalpose:
            x1 = np.concatenate((x1, x[last_ind, : self.pose_dim][None]), axis=0)
        I = x[selected_ind, -self.interaction_dim :]
        traj = x[
            selected_ind, self.start_traj : self.start_traj + self.traj_dim
        ]  # [L, C]

        if self.is_env:
            env = x[
                selected_ind,
                -self.interaction_dim - self.env_dim : -self.interaction_dim,
            ]

        t = len(selected_ind)
        y = x[selected_ind, : self.pose_dim]  # [L, C]
        if self.is_minmax:
            y = utils.normalize_to_neg_one_to_one(
                y,
                self.motion_min[None, : self.pose_dim],
                self.motion_max[None, : self.pose_dim],
            )
        data = {
            "pose": x1,
            "I": I,
            "traj": traj,
            "lengths": t,
            "y": y,
        }
        if self.is_env:
            data["env"] = env
        return data

    def preprocess_data_socket(self, data):
        """_summary_

        Args:
            data (np.array): [C]

        Returns:
            dict (tensor):
                all: [1, c]
        """
        data = to_tensor(data)
        pose_dim = self.pose_dim
        if self.is_goalpose:
            pose_dim = self.pose_dim * 2
        pose = data[:pose_dim].reshape(-1, self.pose_dim)
        condition_dim = self.traj_dim + self.interaction_dim
        if self.is_env:
            condition_dim += self.env_dim
        data = data[pose_dim:].reshape(-1, condition_dim)  # [T, self.traj_dim + 2048]
        traj = data[:, : self.traj_dim]
        I = data[:, self.traj_dim : self.traj_dim + self.interaction_dim]
        pose = utils.normalize(
            pose,
            self.motion_mean[None, : self.pose_dim],
            self.motion_std[None, : self.pose_dim],
        )
        traj = utils.normalize(
            traj,
            self.motion_mean[None, self.start_traj : self.start_traj + self.traj_dim],
            self.motion_std[None, self.start_traj : self.start_traj + self.traj_dim],
        )
        I = utils.normalize(
            I,
            self.motion_mean[None, -self.interaction_dim :],
            self.motion_std[None, -self.interaction_dim :],
        )
        if self.is_env:
            env = data[:, -self.env_dim :]
            env = utils.normalize(
                env,
                self.motion_mean[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
                self.motion_std[
                    None, -self.interaction_dim - self.env_dim : -self.interaction_dim
                ],
            )
        data = {"pose": pose[None], "traj": traj[None], "I": I[None]}
        if self.is_env:
            data["env"] = env[None]
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
        y_hat = result["y_hat"]
        if self.is_minmax:
            y_hat = utils.unnormalize_to_zero_to_one(
                y_hat,
                self.motion_min[None, None, : self.pose_dim],
                self.motion_max[None, None, : self.pose_dim],
            )
        pose = y_hat
        pose[:, 0] = data["pose"][:, 0]
        if self.is_goalpose:
            pose[:, -1] = data["pose"][:, 1]
        pose = utils.denormalize(
            pose,
            self.motion_mean[None, None, : self.pose_dim],
            self.motion_std[None, None, : self.pose_dim],
        )
        result["pose"] = pose
        result = to_cpu_numpy(result)
        return result
