from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import socket
import struct
from threading import Thread
import time
import motion.utils.matrix as matrix
from motion.utils.traj import (
    relative_trajvec2worldmat,
    absolute_trajvec2worldmat,
    abstrajvec2points,
    trajvec2points,
    invtrajvec2points,
    merge_abs_inv_traj,
    interpolate_trajpoint,
    gaussian_traj,
    filter_traj,
    interpolate_trajpoints_L,
)
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy

from motion.comm.builder import SERVERS


class Server:
    def __init__(
        self,
        socket_func,
        cfg,
        hostname: str = "localhost",
        port: int = 3456,
        client_type: str = "c",
    ):
        self.socket_func = socket_func
        self.cfg = cfg
        self.port = port
        self.hostname = hostname
        print(f"{hostname}:{port}")
        self.client_type = client_type
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.hostname, port))
        self.receive_times = 0
        headtbeats_thread = Thread(target=self._heartbeats, daemon=True)
        headtbeats_thread.start()
        self._load_cfg()

    def _load_cfg(self):
        pass

    def convert_and_send(self, data, client):
        """_summary_

        Args:
            data (np.array): [*]

        Returns:
            send_l: _description_
            send_data: _description_
        """
        if isinstance(data, torch.Tensor):
            data = to_cpu_numpy(data)
        data = data.reshape(-1)
        print(f"Send {len(data)}")
        l = np.array([len(data)], dtype=np.float32)
        if self.client_type == "c":
            send_l = struct.pack(f"{len(l)}f", *l)
            send_data = struct.pack(f"{len(data)}f", *data)
        else:
            send_l = l.tobytes()
            send_data = data.tobytes()
        client.sendall(send_l)
        client.sendall(send_data)

    def receive_data(self):
        client, address = self.server.accept()
        self.receive_times += 1
        print(
            f"Connected from {address}, receive {self.receive_times} messages in total!"
        )
        length = client.recv(4)
        length = np.frombuffer(length, dtype=np.int32)[0]
        print(f"Receive {length} data!")

        if length * 4 <= 1024:
            total_data = client.recv(length * 4)
        else:
            total_data = client.recv(1024)
            total_num = length * 4
            left_num = total_num - len(total_data)
            while left_num > 0:
                if left_num > 1024:
                    total_data += client.recv(1024)
                else:
                    total_data += client.recv(left_num)
                left_num = total_num - len(total_data)
        total_data = np.frombuffer(total_data, dtype=np.float32)
        print(f"Get {len(total_data)} now!")
        return client, total_data

    def receive_data_with_str(self, already_connect=True):
        client, address = self.server.accept()
        self.receive_times += 1
        print(
            f"Connected from {address}, receive {self.receive_times} messages in total!"
        )
        length = client.recv(4)
        length = np.frombuffer(length, dtype=np.int32)[0]
        print(f"Receive {length} data!")

        if length * 4 <= 1024:
            total_data = client.recv(length * 4)
        else:
            total_data = client.recv(1024)
            total_num = length * 4
            left_num = total_num - len(total_data)
            while left_num > 0:
                if left_num > 1024:
                    total_data += client.recv(1024)
                else:
                    total_data += client.recv(left_num)
                left_num = total_num - len(total_data)
        total_data = np.frombuffer(total_data, dtype=np.float32)
        print(f"Get float {len(total_data)} now!")

        length = client.recv(4)
        length = np.frombuffer(length, dtype=np.int32)[0]
        print(f"Receive {length} data!")

        if length * 4 <= 1024:
            str_data = client.recv(length * 4)
        else:
            str_data = client.recv(1024)
            total_num = length * 4
            left_num = total_num - len(str_data)
            while left_num > 0:
                if left_num > 1024:
                    str_data += client.recv(1024)
                else:
                    str_data += client.recv(left_num)
                left_num = total_num - len(str_data)
        str_data = np.frombuffer(str_data, dtype=np.dtype(f"<U{length}"))
        print(f"Get str {len(str_data)} now!")
        return client, total_data, str_data

    def receive_data_with_ind(self, already_connect=True):
        client, address = self.server.accept()
        self.receive_times += 1
        print(
            f"Connected from {address}, receive {self.receive_times} messages in total!"
        )
        length = client.recv(4)
        length = np.frombuffer(length, dtype=np.int32)[0]
        print(f"Receive {length} data!")

        if length * 4 <= 1024:
            total_data = client.recv(length * 4)
        else:
            total_data = client.recv(1024)
            total_num = length * 4
            left_num = total_num - len(total_data)
            while left_num > 0:
                if left_num > 1024:
                    total_data += client.recv(1024)
                else:
                    total_data += client.recv(left_num)
                left_num = total_num - len(total_data)
        total_data = np.frombuffer(total_data, dtype=np.float32)
        print(f"Get float {len(total_data)} now!")

        length = client.recv(4)
        length = np.frombuffer(length, dtype=np.int32)[0]
        print(f"Receive {length} data!")

        if length * 4 <= 1024:
            str_data = client.recv(length * 4)
        else:
            str_data = client.recv(1024)
            total_num = length * 4
            left_num = total_num - len(str_data)
            while left_num > 0:
                if left_num > 1024:
                    str_data += client.recv(1024)
                else:
                    str_data += client.recv(left_num)
                left_num = total_num - len(str_data)
        ind_data = np.frombuffer(str_data, dtype=np.float32)
        print(f"Get ind {len(ind_data)} now!")
        return client, total_data, ind_data

    def listen(self):
        print("Start to listen...")
        self.server.listen()
        while True:
            try:
                client, total_data = self.receive_data()
                print(f"Receive {total_data}")
                output = self.socket_func(total_data)
                self.convert_and_send(output, client)
                client.close()
            except Exception as e:
                print(e)

    def _heartbeats(self):
        while True:
            print(
                f"[HeartBeats] {self.hostname}:{self.port} Receive {self.receive_times} times!"
            )
            time.sleep(120)


class BaseServer(Server):
    def __init__(
        self,
        socket_func,
        cfg,
        hostname: str = "localhost",
        port: int = 3456,
        client_type: str = "c",
    ):
        super().__init__(socket_func, cfg, hostname, port, client_type)
        self.send_keys = []
        self.network_time = 0.0
        self.inference_time = 0.0

    def listen(self):
        print("Start to listen...")
        self.server.listen()
        while True:
            try:
                client, total_data = self.receive_data()
                print(f"Receive {total_data}")
                start_time = time.time()
                try:
                    output = self.socket_func(total_data)
                except Exception as e:
                    print(e)
                    print("Fail at socket func")
                    continue
                if self.receive_times > 10:
                    self.network_time += time.time() - start_time
                output = self.postprocess_output(output, total_data)
                if self.receive_times > 10:
                    self.inference_time += time.time() - start_time
                self.send_output(output, client)
                client.close()
            except Exception as e:
                print(e)
            if self.receive_times > 10:
                print(f"Receive: {self.receive_times}")
                print(
                    f"Avg network time: {self.network_time * 1000 / (self.receive_times - 10)}ms;"
                )
                print(
                    f"Avg inference time: {self.inference_time * 1000 / (self.receive_times - 10)}ms;"
                )

    def postprocess_output(self, output, total_data):
        return output

    def send_output(self, output, client):
        for k in self.send_keys:
            if k in output.keys():
                self.convert_and_send(output[k], client)


@SERVERS.register_module()
class MotionServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["pose"]

    def postprocess_output(self, output, total_data):
        pose = output["pose"]
        for i in range(pose.shape[0] - 1):
            avg_pose = (pose[i][-1] + pose[i + 1][0]) / 2
            pose[i][-1] = avg_pose
            pose[i + 1][0] = avg_pose

        pose_nofirst = pose[:, 1:].reshape(-1, pose.shape[-1])
        pose = np.concatenate((pose[0, :1], pose_nofirst), axis=0)
        if "original_n" in output.keys():
            pose = pose[: output["original_n"]]

        # contact = output["contact"]
        # for i in range(contact.shape[0] - 1):
        #     avg_contact = (contact[i][-1] + contact[i + 1][0]) / 2
        #     contact[i][-1] = avg_contact
        #     contact[i + 1][0] = avg_contact
        # contact_nofirst = contact[:, 1:].reshape(-1, contact.shape[-1])
        # contact = np.concatenate((contact[0, :1], contact_nofirst), axis=0)
        # pose = np.concatenate((pose, contact), axis=-1)
        output["pose"] = pose
        return output


@SERVERS.register_module()
class MoEServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["y_hat", "omega"]


@SERVERS.register_module()
class MilestoneServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["T", "traj", "traj_state"]

    def _load_cfg(self):
        super()._load_cfg()
        cfg = self.cfg.DATASET.cfg

        self.is_goalpose = cfg.is_goalpose
        self.is_env = cfg.is_env
        self.L = cfg.L
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
        self.trajpoint_dim = self.end_style_now - self.start_traj_now
        self.style_dim = self.end_style_now - self.start_style_now

    def postprocess_output(self, output, total_data):
        data = to_tensor(total_data.reshape(1, -1))  # [1, 647 + 2048 + 12 + 12 + 2048]

        # For only goal version
        # root = data[:, 2695:2707]
        # goal = data[:, 2707:2719]

        # For goal pose version
        pose_dims = self.pose_dim
        if self.is_goalpose:
            pose_dims = self.pose_dim * 2

        root = data[
            :, pose_dims + self.interaction_dim : pose_dims + self.interaction_dim + 12
        ]
        goal = data[
            :,
            pose_dims
            + self.interaction_dim
            + 12 : pose_dims
            + self.interaction_dim
            + 24,
        ]

        root_mat = matrix.vec2mat_batch(root)[0]
        goal_mat = matrix.vec2mat_batch(goal)[0]
        Milestone_pred = output["traj_pos"].reshape(1, -1, 12)
        traj_vec = Milestone_pred[..., :4]
        abstraj_vec = Milestone_pred[..., 4:8]
        invtraj_vec = Milestone_pred[..., 8:12]
        traj = trajvec2points(traj_vec)
        abs_traj = abstrajvec2points(abstraj_vec)
        invtraj = invtrajvec2points(invtraj_vec, goal_mat, root_mat)
        merge_traj_1 = merge_abs_inv_traj(traj, abs_traj)
        T = traj_vec.shape[-2]

        # merge_weight = 1 - 1 / (1 + np.exp(merge_weight))
        # merge_weight[0] = 0.0
        # merge_weight[-1] = 1.0

        # sub_T = T // 2
        # sub_merge_weight = np.arange(sub_T) - (sub_T - 1) / 2.0
        # sub_merge_weight = 1 - 1 / (1 + np.exp(sub_merge_weight))
        # sub_merge_weight[0] = 0.0
        traj_window = output["traj_window"]
        style = traj_window[..., : -self.contact_dim].reshape(
            -1, 13, self.trajpoint_dim
        )[..., :, -self.style_dim :]
        style = np.clip(style, 0, 1)
        traj_state = traj_window[..., : -self.contact_dim].reshape(
            -1, 13, self.trajpoint_dim
        )[..., :, : -self.style_dim]
        contact = np.clip(traj_window[..., -self.contact_dim :], 0, 1)
        clip_traj_window = np.concatenate((traj_state, style), axis=-1).reshape(
            1, T, -1
        )
        clip_traj_window = np.concatenate((clip_traj_window, contact), axis=-1)

        start_action = data[0, -2 * self.style_dim : -self.style_dim]
        start_action_ind = start_action.argmax()
        goal_action = data[0, -self.style_dim :]
        goal_action_ind = goal_action.argmax()
        traj_style = clip_traj_window[
            0,
            :,
            self.start_style_now
            - self.start_traj : self.end_style_now
            - self.start_traj,
        ]

        threshold = 0.7
        merge_weight = np.linspace(0, 1, T)
        interaction_T = 0
        for i in range(len(merge_weight) - 1):
            if (
                traj_style[i].argmax() == goal_action_ind
                and traj_style[i][goal_action_ind] > threshold
            ):
                interaction_T += 1

        inter = (T - interaction_T) // 3 + 1
        merge_weight[:inter] = 0.0
        merge_weight[-(inter + interaction_T) :] = 1.0

        merge_traj = merge_abs_inv_traj(abs_traj, invtraj, merge_weight)
        # merge_traj = merge_abs_inv_traj(merge_traj_1, invtraj)
        merge_traj = np.stack(merge_traj, axis=-1)

        new_traj = [merge_traj[0]]
        for i in range(1, len(merge_traj) - 1):
            point_style = traj_style[i]
            if (
                point_style.argmax() == start_action_ind
                and point_style[start_action_ind] > threshold
                and i <= 2
            ):
                new_traj.append(
                    interpolate_trajpoint(merge_traj[i], merge_traj[0], w1=1.0, w2=0.0)
                )
            elif (
                point_style.argmax() == goal_action_ind
                and point_style[goal_action_ind] > threshold
                and i > 2
            ):
                new_traj.append(
                    interpolate_trajpoint(merge_traj[i], merge_traj[-1], w1=1.0, w2=0.0)
                )
            else:
                new_traj.append(merge_traj[i])
        new_traj.append(merge_traj[-1])
        merge_traj = np.stack(new_traj, axis=0)

        T = np.array([i * self.L + 1 for i in range(len(merge_traj))])

        output["T"] = T
        output["traj"] = merge_traj
        output["traj_state"] = clip_traj_window
        return output


@SERVERS.register_module()
class MilestoneWithPoseServer(MilestoneServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["T", "traj", "traj_state", "pose"]


@SERVERS.register_module()
class TrajServer(MilestoneServer):
    def postprocess_output(self, output, total_data):
        data = to_tensor(total_data.reshape(1, -1))  # [1, 647 + 2048 + 12 + 12 + 2048]

        # For goal pose version
        pose_dims = self.pose_dim
        if self.is_goalpose:
            pose_dims = self.pose_dim * 2

        root = data[
            :, pose_dims + self.interaction_dim : pose_dims + self.interaction_dim + 12
        ]
        goal = data[
            :,
            pose_dims
            + self.interaction_dim
            + 12 : pose_dims
            + self.interaction_dim
            + 24,
        ]

        root_mat = matrix.vec2mat_batch(root)[0]
        goal_mat = matrix.vec2mat_batch(goal)[0]
        Milestone_pred = output["traj_pos"].reshape(1, -1, 12)
        traj_vec = Milestone_pred[..., :4]
        abstraj_vec = Milestone_pred[..., 4:8]
        invtraj_vec = Milestone_pred[..., 8:12]
        traj = trajvec2points(traj_vec)
        abs_traj = abstrajvec2points(abstraj_vec)
        invtraj = invtrajvec2points(invtraj_vec, goal_mat, root_mat)
        merge_traj = merge_abs_inv_traj(abs_traj, invtraj)
        merge_traj = np.stack(merge_traj, axis=-1)
        T = traj_vec.shape[-2]

        traj_state = output["traj_window"][0]
        mergetraj = merge_traj

        # modify the traj state
        mergetraj = to_tensor(mergetraj)
        traj_state = to_tensor(traj_state)
        mergetraj_expand = torch.cat(
            (
                mergetraj[:1].repeat(30, 1),
                mergetraj,
                mergetraj[-1:].repeat(30, 1),
            ),
            dim=0,
        )
        mergetraj_mat = matrix.xzvec2mat(mergetraj)
        mergetraj_expand_mat = matrix.xzvec2mat(mergetraj_expand)
        state_expand = torch.cat(
            (
                traj_state[:1].repeat(30, 1),
                traj_state,
                traj_state[-1:].repeat(30, 1),
            ),
            dim=0,
        )

        reltraj_num = 13
        pivot = 7
        step_size = 5
        for i in range(reltraj_num):
            relind = [
                j + (i - pivot + 1) * step_size + 30 for j in range(len(traj_state))
            ]
            relind = to_tensor(relind, torch.int64)
            relmat = matrix.get_mat_BtoA(mergetraj_mat, mergetraj_expand_mat[relind])
            relxz = matrix.project_vec(matrix.mat2vec_batch(relmat))
            traj_state[:, i * self.trajpoint_dim : i * self.trajpoint_dim + 4] = relxz
            relstyle = state_expand[relind][
                ..., (pivot - 1) * self.trajpoint_dim + 4 : pivot * self.trajpoint_dim
            ]
            relstyle = torch.clamp(relstyle, 0, 1)  # for style
            traj_state[
                :,
                i * self.trajpoint_dim
                + 4 : i * self.trajpoint_dim
                + self.trajpoint_dim,
            ] = relstyle
        traj_state[:, -self.contact_dim :] = torch.clamp(
            traj_state[:, -self.contact_dim :], 0, 1
        )  # for contact

        T = np.array([i + 1 for i in range(len(merge_traj))])

        output["T"] = T
        output["traj"] = mergetraj
        output["traj_state"] = traj_state
        return output


@SERVERS.register_module()
class TrajCompletionServer(MilestoneServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["traj", "traj_state"]

    def _load_cfg(self):
        super()._load_cfg()
        cfg = self.cfg.DATASET.cfg

    def postprocess_output(self, output, total_data):
        state_dim = self.traj_dim
        D = state_dim + self.interaction_dim + 12 + self.env_dim
        data = to_tensor(total_data.reshape(-1, D))  # [N, 122 + 2048 + 12]
        root_mat = matrix.vec2mat_batch(
            data[
                :,
                state_dim
                + self.interaction_dim : state_dim
                + self.interaction_dim
                + 12,
            ]
        )
        traj_pred = output["traj_pos"].reshape(-1, self.L, 12)
        traj_vec = traj_pred[..., :4]
        abstraj_vec = traj_pred[..., 4:8]
        invtraj_vec = traj_pred[..., 8:12]

        traj_style = output["traj_window"][
            ...,
            self.start_style_now
            - self.start_traj : self.end_style_now
            - self.start_traj,
        ]
        target_style = data[
            -1,
            self.start_style_now
            - self.start_traj : self.end_style_now
            - self.start_traj,
        ]
        style_ind = target_style.argmax()

        mergetraj_all = []

        for i, (traj_vec_seq, abstraj_vec_seq, invtraj_vec_seq) in enumerate(
            zip(traj_vec, abstraj_vec, invtraj_vec)
        ):
            relative_start_mat = matrix.get_mat_BtoA(
                root_mat[0],
                root_mat[i],
            )
            traj = trajvec2points(traj_vec_seq, relative_start_mat)
            abstraj = abstrajvec2points(abstraj_vec_seq, relative_start_mat)
            invtraj = invtrajvec2points(
                invtraj_vec_seq,
                root_mat[i + 1],
                root_mat[i],
                relative_start_mat,
            )
            # mergetraj = merge_abs_inv_traj(traj, abstraj)
            # mergetraj = merge_abs_inv_traj(mergetraj, invtraj)

            mergetraj = merge_abs_inv_traj(abstraj, invtraj)

            # mergetraj = merge_abs_inv_traj(abstraj, invtraj)
            # mergetraj = merge_abs_inv_traj(traj, mergetraj)
            mergetraj = np.stack(mergetraj, axis=-1)
            traj_style_seq = traj_style[i]
            # if traj_style_seq[0, style_ind] > 0.9 and traj_style_seq[-1, style_ind] > 0.9:
            if (
                root_mat[i, :-1, -1] - root_mat[i + 1, :-1, -1]
            ).abs().sum() == 0 or i == traj_vec.shape[0] - 1:
                mergetraj = interpolate_trajpoints_L(
                    mergetraj[:1], mergetraj[-1:], L=mergetraj.shape[0]
                )

            if i > 0:
                mergetraj = mergetraj[1:]
            mergetraj_all.append(mergetraj)
        mergetraj = np.concatenate(mergetraj_all, axis=0)

        mergetraj = gaussian_traj(mergetraj)
        mergetraj = filter_traj(mergetraj)

        traj_state = output["traj_window"]
        for i in range(traj_state.shape[0] - 1):
            avg_state = (traj_state[i][-1] + traj_state[i + 1][0]) / 2
            traj_state[i][-1] = avg_state
            traj_state[i + 1][0] = avg_state

        traj_state_nofirst = traj_state[:, 1:].reshape(-1, traj_state.shape[-1])
        traj_state = np.concatenate((traj_state[0, :1], traj_state_nofirst), axis=0)

        # modify the traj state
        mergetraj = to_tensor(mergetraj)
        traj_state = to_tensor(traj_state)
        mergetraj_expand = torch.cat(
            (
                mergetraj[:1].repeat(30, 1),
                mergetraj,
                mergetraj[-1:].repeat(30, 1),
            ),
            dim=0,
        )
        mergetraj_mat = matrix.xzvec2mat(mergetraj)
        mergetraj_expand_mat = matrix.xzvec2mat(mergetraj_expand)
        state_expand = torch.cat(
            (
                traj_state[:1].repeat(30, 1),
                traj_state,
                traj_state[-1:].repeat(30, 1),
            ),
            dim=0,
        )

        reltraj_num = 13
        pivot = 7
        step_size = 5
        for i in range(reltraj_num):
            relind = [
                j + (i - pivot + 1) * step_size + 30 for j in range(len(traj_state))
            ]
            relind = to_tensor(relind, torch.int64)
            relmat = matrix.get_mat_BtoA(mergetraj_mat, mergetraj_expand_mat[relind])
            relxz = matrix.project_vec(matrix.mat2vec_batch(relmat))
            traj_state[:, i * self.trajpoint_dim : i * self.trajpoint_dim + 4] = relxz
            relstyle = state_expand[relind][
                ..., (pivot - 1) * self.trajpoint_dim + 4 : pivot * self.trajpoint_dim
            ]
            relstyle = torch.clamp(relstyle, 0, 1)  # for style
            traj_state[
                :,
                i * self.trajpoint_dim
                + 4 : i * self.trajpoint_dim
                + self.trajpoint_dim,
            ] = relstyle
        traj_state[:, -self.contact_dim :] = torch.clamp(
            traj_state[:, -self.contact_dim :], 0, 1
        )  # for contact

        output["traj"] = mergetraj
        output["traj_state"] = traj_state
        return output


@SERVERS.register_module()
class TrajMotionServer(TrajCompletionServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["traj", "traj_state", "pose"]

    def postprocess_output(self, output, total_data):
        if self.is_only_action:
            state_dim = self.style_dim
        else:
            state_dim = self.traj_dim
        D = state_dim + self.interaction_dim + 12 + self.env_dim + self.pose_dim
        data = to_tensor(total_data.reshape(-1, D))  # [N, 122 + 2048 + 12]
        root_mat = matrix.vec2mat_batch(
            data[
                :,
                state_dim
                + self.interaction_dim : state_dim
                + self.interaction_dim
                + 12,
            ]
        )
        traj_pred = output["traj_pos"].reshape(-1, self.L, 12)
        traj_vec = traj_pred[..., :4]
        abstraj_vec = traj_pred[..., 4:8]
        invtraj_vec = traj_pred[..., 8:12]

        mergetraj_all = []

        for i, (traj_vec_seq, abstraj_vec_seq, invtraj_vec_seq) in enumerate(
            zip(traj_vec, abstraj_vec, invtraj_vec)
        ):
            relative_start_mat = matrix.get_mat_BtoA(
                root_mat[0],
                root_mat[i],
            )
            traj = trajvec2points(traj_vec_seq, relative_start_mat)
            abstraj = abstrajvec2points(abstraj_vec_seq, relative_start_mat)
            invtraj = invtrajvec2points(
                invtraj_vec_seq,
                root_mat[i + 1],
                root_mat[i],
                relative_start_mat,
            )
            mergetraj = merge_abs_inv_traj(abstraj, invtraj)
            mergetraj = merge_abs_inv_traj(traj, mergetraj)
            mergetraj = np.stack(mergetraj, axis=-1)
            if i > 0:
                mergetraj = mergetraj[1:]
            mergetraj_all.append(mergetraj)
        mergetraj = np.concatenate(mergetraj_all, axis=0)

        traj_state = output["traj_window"]
        for i in range(traj_state.shape[0] - 1):
            avg_state = (traj_state[i][-1] + traj_state[i + 1][0]) / 2
            traj_state[i][-1] = avg_state
            traj_state[i + 1][0] = avg_state

        traj_state_nofirst = traj_state[:, 1:].reshape(-1, traj_state.shape[-1])
        traj_state = np.concatenate((traj_state[0, :1], traj_state_nofirst), axis=0)

        # modify the traj state
        mergetraj = to_tensor(mergetraj)
        traj_state = to_tensor(traj_state)
        mergetraj_expand = torch.cat(
            (
                mergetraj[:1].repeat(30, 1),
                mergetraj,
                mergetraj[-1:].repeat(30, 1),
            ),
            dim=0,
        )
        mergetraj_mat = matrix.xzvec2mat(mergetraj)
        mergetraj_expand_mat = matrix.xzvec2mat(mergetraj_expand)
        state_expand = torch.cat(
            (
                traj_state[:1].repeat(30, 1),
                traj_state,
                traj_state[-1:].repeat(30, 1),
            ),
            dim=0,
        )

        reltraj_num = 13
        pivot = 7
        step_size = 5
        for i in range(reltraj_num):
            relind = [
                j + (i - pivot + 1) * step_size + 30 for j in range(len(traj_state))
            ]
            relind = to_tensor(relind, torch.int64)
            relmat = matrix.get_mat_BtoA(mergetraj_mat, mergetraj_expand_mat[relind])
            relxz = matrix.project_vec(matrix.mat2vec_batch(relmat))
            traj_state[:, i * self.trajpoint_dim : i * self.trajpoint_dim + 4] = relxz
            relstyle = state_expand[relind][
                ..., (pivot - 1) * self.trajpoint_dim + 4 : pivot * self.trajpoint_dim
            ]
            relstyle = torch.clamp(relstyle, 0, 1)  # for style
            traj_state[
                :,
                i * self.trajpoint_dim
                + 4 : i * self.trajpoint_dim
                + self.trajpoint_dim,
            ] = relstyle
        traj_state[:, -self.contact_dim :] = torch.clamp(
            traj_state[:, -self.contact_dim :], 0, 1
        )  # for contact

        output["traj"] = mergetraj
        output["traj_state"] = traj_state
        return output


@SERVERS.register_module()
class PoseServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["pose"]


@SERVERS.register_module()
class PoseContactServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["pose", "contact"]

    def postprocess_output(self, output, total_data):
        contact = output["contact"]
        contact = np.clip(contact, 0, 1)
        output["contact"] = contact
        return output


@SERVERS.register_module()
class POSAServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["pose", "root"]

    def listen(self):
        print("Start to listen...")
        self.server.listen()
        while True:
            try:
                # client, total_data, str_data = self.receive_data_with_str()
                client, total_data, ind_data = self.receive_data_with_ind()
                print(f"Receive float {total_data}")
                # print(f"Receive str {str_data}")
                print(f"Receive ind {ind_data}")
                start_time = time.time()
                try:
                    # output = self.socket_func((total_data, str_data))
                    output = self.socket_func((total_data, ind_data))
                except Exception as e:
                    print(e)
                    print("Fail at socket func")
                    continue
                if self.receive_times > 10:
                    self.network_time += time.time() - start_time
                output = self.postprocess_output(output, total_data)
                if self.receive_times > 10:
                    self.inference_time += time.time() - start_time
                self.send_output(output, client)
                client.close()
            except Exception as e:
                print(e)
            if self.receive_times > 10:
                print(f"Receive: {self.receive_times}")
                print(
                    f"Avg network time: {self.network_time * 1000 / (self.receive_times - 10)}ms;"
                )
                print(
                    f"Avg inference time: {self.inference_time * 1000 / (self.receive_times - 10)}ms;"
                )


@SERVERS.register_module()
class TrajRefineServer(TrajCompletionServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["traj"]

    def postprocess_output(self, output, total_data):
        state_dim = self.style_dim
        D = state_dim + self.interaction_dim + 12 + self.env_dim
        data = to_tensor(total_data.reshape(-1, D))  # [N, 122 + 2048 + 12]
        root_mat = matrix.vec2mat_batch(
            data[
                :,
                state_dim
                + self.interaction_dim : state_dim
                + self.interaction_dim
                + 12,
            ]
        )
        traj_pred = output["traj_pos"].reshape(-1, self.L, 4)
        traj_vec = traj_pred[..., :4]

        mergetraj_all = []

        for i, abs_traj_seq in enumerate(traj_vec):
            relative_start_mat = matrix.get_mat_BtoA(
                root_mat[0],
                root_mat[i],
            )
            abstraj = abstrajvec2points(abs_traj_seq, relative_start_mat)
            mergetraj = np.stack(abstraj, axis=-1)
            if i > 0:
                mergetraj = mergetraj[1:]
            mergetraj_all.append(mergetraj)
        mergetraj = np.concatenate(mergetraj_all, axis=0)
        output["traj"] = mergetraj
        return output


@SERVERS.register_module()
class MapperServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_keys = ["h"]
