import argparse
from turtle import st
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import socket
import time
import motion.utils.matrix as matrix
from motion.utils.traj import (
    relative_trajvec2worldmat,
    absolute_trajvec2worldmat,
    abstrajvec2points,
    trajvec2points,
    invtrajvec2points,
    merge_abs_inv_traj,
)
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy


class Client:
    def __init__(self, servername: str, port: int, encode_type="utf-8"):
        self.encode_type = encode_type
        self.servername = servername
        self.port = port
        self.load_traj_data()
        # self.load_motion_data()

    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client.connect((self.servername, self.port))
                print(f"Successfully connect to {self.servername}:{self.port}")
                break
            except ConnectionRefusedError:
                print(f"Faild to connect {self.servername}:{self.port} !")
                time.sleep(1)
                pass

    def receive_data(self):
        l = self.client.recv(4)
        l = int(np.frombuffer(l, dtype=np.float32)[0])
        print(f"Receive {l} data!")
        if l * 4 > 1024:
            total_data = self.client.recv(1024)
            left = l * 4 - 1024
            while True:
                if left > 1024:
                    total_data += self.client.recv(1024)
                else:
                    total_data += self.client.recv(left)
                if len(total_data) == l * 4:
                    print(f"Get {len(total_data)} now!")
                    break
                left -= 1024
        else:
            total_data = self.client.recv(l * 4)
            print(f"Get {len(total_data)} now!")
        total_data = np.frombuffer(total_data, dtype=np.float32)
        return total_data

    def send_motion(self):
        self.connect()
        data = self.motion_input_data[0]
        msg = data
        msg = msg.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        self.receive_data()
        self.receive_data()
        self.client.close()

    def load_motion_data(self):
        split = "test"
        data_path = "./datasets/nsm/Motion"
        print(f"Load data from {data_path}/{split}")
        input_path = f"{data_path}/{split}/Input.txt"
        output_path = f"{data_path}/{split}/Output.txt"
        sequence_path = f"{data_path}/{split}/Sequences.txt"
        self.motion_input_data = pd_load(input_path).to_numpy()
        self.motion_output_data = pd_load(output_path).to_numpy()
        self.motion_seqind_data = pd_load(sequence_path).to_numpy()[:, 0]

    def load_traj_data(self):
        split = "test"
        data_path = "./datasets/samp/MotionObject"
        print(f"Load data from {data_path}/{split}")
        input_path = f"{data_path}/{split}/Input.txt"
        output_path = f"{data_path}/{split}/Output.txt"
        world_path = f"{data_path}/{split}/World.txt"
        sequence_path = f"{data_path}/{split}/Sequences.txt"
        obj_path = f"{data_path}/{split}/ObjFile.txt"
        self.input_data = pd_load(input_path).to_numpy()
        self.output_data = pd_load(output_path).to_numpy()
        self.world_data = pd_load(world_path).to_numpy()
        self.seqind_data = pd_load(sequence_path).to_numpy()[:, 0]
        self.obj_data = pd_load(obj_path).to_numpy()

    def send_goalpose(self, msg="biu"):
        self.connect()
        t = 500
        I = self.input_data[t, -2048:]
        # I = self.input_data[t, -2048 - 315 : -2048]
        action = self.input_data[t, 388:393]
        env = self.input_data[t, -315 - 2048 : -2048]
        msg = np.concatenate((action, env, I), axis=0)
        msg = msg.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        pose = self.receive_data()
        # self.receive_data()
        self.client.close()

    def send_posa(self, msg="biu"):
        self.connect()
        t = 500
        pose = self.input_data[t, :264]
        obj_pos = self.obj_data[t, :3]
        obj_forward = self.obj_data[t, 3:6]
        world_pos = self.world_data[t, :3]
        world_forward = self.world_data[t, 3:6]
        scale = np.ones_like(pose)[..., :6]
        pose = np.concatenate(
            (pose, world_pos, world_forward, scale, obj_pos, obj_forward)
        )
        msg = pose.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        objname = np.array([0.0], dtype=np.float32).reshape(-1)
        msg = objname
        l = len(objname)
        print(l)
        l = np.array(l, dtype=np.int32).tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        pose = self.receive_data()
        self.client.close()

    def send_slt_optim(self, N=1, msg="biu"):
        self.connect()
        t = N * 60
        pose = self.input_data[:t, :264]
        obj_pos = self.obj_data[:t, :3]
        obj_forward = self.obj_data[:t, 3:6]
        world_pos = self.world_data[:t, :3]
        world_forward = self.world_data[:t, 3:6]
        scale = np.ones_like(pose)[..., :6]
        pose = np.concatenate(
            (pose, world_pos, world_forward, scale, obj_pos, obj_forward), axis=-1
        )
        msg = pose.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        objname = np.array([0.0], dtype=np.float32).reshape(-1)
        msg = objname
        l = len(objname)
        print(l)
        l = np.array(l, dtype=np.int32).tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        pose = self.receive_data()
        self.client.close()

    def send_staticpose(self, msg="biu"):
        self.connect()
        pose = self.input_data[0, :264]
        trajI = self.input_data[:3]
        traj = trajI[:, 330:452]
        I = trajI[:, -2048:]
        msg = np.concatenate((traj, I), axis=1)
        msg = msg.reshape(-1)
        msg = np.concatenate((pose, msg), axis=0)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        self.receive_data()
        self.client.close()

    def send_staticpose_goalpose(self, N=10, msg="biu"):
        self.connect()
        pose = self.input_data[0, :264]
        goalpose = self.input_data[2, :264]
        trajI = self.input_data[:N]
        traj = trajI[:, 330:452]
        I = trajI[:, -2048:]
        env = trajI[:, -2048 - 315 : -2048]
        msg = np.concatenate((traj, I, env), axis=1)
        msg = msg.reshape(-1)
        msg = np.concatenate((pose, goalpose, msg), axis=0)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        self.receive_data()
        self.client.close()

    def send_actorpose(self, N=2, msg="biu"):
        self.connect()
        T = (61 - 1) * N + 1
        pose = self.input_data[:T, :264]
        traj = self.input_data[:T, 330:452]
        I = self.input_data[:T, -2048:]
        env = self.input_data[:T, -2048 - 315 : -2048]
        msg = np.concatenate((pose, traj, I, env), axis=1)
        msg = msg.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        pred = self.receive_data()
        self.client.close()

    def send_traj_absolute(self):
        self.connect()
        start_t = 0
        msg = np.concatenate(
            (
                self.input_data[start_t : start_t + 1],
                self.world_data[start_t : start_t + 1],
            ),
            axis=-1,
        )
        msg = msg.astype(np.float32)

        T = msg.shape[0]
        l = msg.shape[0] * msg.shape[1]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        traj_vec = self.receive_data()
        traj_window = self.receive_data()
        prob = self.receive_data()

        plt.subplot(1, 2, 1)
        plt.bar(range(len(prob)), prob)

        traj_vec = traj_vec.reshape(1, -1, 12)
        abstraj_vec = traj_vec[..., 4:8]
        invtraj_vec = traj_vec[..., 8:12]
        traj_vec = traj_vec[..., :4]
        T = traj_vec.shape[1]
        style = self.input_data[self.seqind_data == 1][:, 388:393]
        gt_t = (style[:, 3] == 1).nonzero()[0] + (style[:, 3] == 1).sum() // 4

        print(f"predict T = {T}, while gt = {gt_t[0] - start_t}")

        traj_winodw = traj_window.reshape(T, -1)

        world_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, :12]))
        goal_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, 12:24]))
        # gt_traj_vec = to_tensor(self.output_data[start_t : start_t + T, 384:388])
        # traj_world_mat = relative_trajvec2worldmat(gt_traj_vec)
        # traj_vec = matrix.mat2vec_batch(traj_world_mat)
        # traj_vec = matrix.project_vec(traj_vec)  # [l, 4]

        traj_x, traj_y = trajvec2points(traj_vec)
        abs_traj_x, abs_traj_y = abstrajvec2points(abstraj_vec)
        invtraj_x, invtraj_y = invtrajvec2points(
            invtraj_vec, goal_mat[start_t], world_mat[start_t]
        )

        world_xyz = to_tensor(self.world_data[start_t : start_t + T, :3])
        new_xyz = matrix.get_relative_position_to(world_xyz, world_mat[start_t])
        new_xyz = new_xyz - new_xyz[:1]
        new_xyz = to_cpu_numpy(new_xyz)

        plt.subplot(1, 2, 2)
        plt.plot(new_xyz[:, 0], new_xyz[:, 2], "o-", alpha=0.5, label="gt")
        plt.plot(traj_x, traj_y, "o-", alpha=0.5, label="pred")
        plt.plot(abs_traj_x, abs_traj_y, "o-", alpha=0.5, label="pred-abs")
        plt.plot(invtraj_x, invtraj_y, "o-", alpha=0.5, label="pred-inv")

        goal_xyz = to_tensor(self.input_data[200:201, 570:573])
        goal_xyz = matrix.get_position_from(goal_xyz, world_mat[200])

        goal_xyz = to_tensor(self.world_data[start_t : start_t + 1, 12:15])
        goal_xyz = matrix.get_relative_position_to(goal_xyz, world_mat[start_t])
        plt.plot(goal_xyz[:, 0], goal_xyz[:, 2], "x", label="target")
        plt.legend()
        plt.show()

        # error = np.abs(actor_data.reshape(T, -1) - msg[:, :330]).sum(axis=-1).mean()
        # print(f"Error: {error}")
        self.client.close()

    def send_Milestone(self, N=10):
        self.connect()
        # start_t = 5279
        start_t = 0
        start_pose = self.input_data[start_t][:264]
        start_I = self.input_data[start_t][-2048:]
        goalpose = self.input_data[start_t + 472][:264]
        world_data = self.world_data[start_t]
        startaction = self.input_data[start_t][388:393]
        goalaction = self.input_data[start_t + 472][388:393]
        envR = self.input_data[start_t][-2048 - 315 : -2048]
        envG = self.input_data[start_t + 472][-2048 - 315 : -2048]

        msg = np.concatenate(
            (
                start_pose,
                goalpose,
                start_I,
                world_data,
                envR,
                envG,
                startaction,
                goalaction,
            ),
            axis=-1,
        )
        msg = msg.astype(np.float32)

        l = len(msg)
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        traj_vec = self.receive_data()
        traj_window = self.receive_data()
        prob = self.receive_data()

        plt.subplot(1, 2, 1)
        plt.bar(range(len(prob)), prob)

        traj_vec = traj_vec.reshape(1, -1, 12)
        abstraj_vec = traj_vec[..., 4:8]
        invtraj_vec = traj_vec[..., 8:12]
        traj_vec = traj_vec[..., :4]
        T = traj_vec.shape[1]
        style = self.input_data[self.seqind_data == 1][:, 388:393]
        gt_t = (style[:, 3] == 1).nonzero()[0] + (style[:, 3] == 1).sum() // 4

        print(f"predict T = {T}, while gt = {(gt_t[0] - start_t) // 60 + 1}")

        traj_winodw = traj_window.reshape(T, -1)

        world_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, :12]))
        goal_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, 12:24]))
        # gt_traj_vec = to_tensor(self.output_data[start_t : start_t + T, 384:388])
        # traj_world_mat = relative_trajvec2worldmat(gt_traj_vec)
        # traj_vec = matrix.mat2vec_batch(traj_world_mat)
        # traj_vec = matrix.project_vec(traj_vec)  # [l, 4]

        traj = trajvec2points(traj_vec)
        abs_traj = abstrajvec2points(abstraj_vec)
        invtraj = invtrajvec2points(invtraj_vec, goal_mat[start_t], world_mat[start_t])
        # merge_traj_1 = merge_abs_inv_traj(abs_traj, invtraj)
        # merge_traj = merge_abs_inv_traj(traj, merge_traj_1)
        # merge_weight = np.ones(T)
        # if T > 5:
        #     merge_weight[:3] = np.linspace(0, 1, 3)
        merge_traj_1 = merge_abs_inv_traj(abs_traj, invtraj)
        merge_traj = merge_abs_inv_traj(traj, merge_traj_1)

        world_xyz = to_tensor(self.world_data[start_t : start_t + T * 61, :3])
        new_xyz = matrix.get_relative_position_to(world_xyz, world_mat[start_t])
        new_xyz = new_xyz - new_xyz[:1]
        new_xyz = to_cpu_numpy(new_xyz)

        plt.subplot(1, 2, 2)
        plt.plot(new_xyz[:, 0], new_xyz[:, 2], "o-", alpha=0.2, label="gt")

        plt.plot(*traj[:2], "o-", alpha=0.5, label="pred")
        plt.quiver(*traj[:4], alpha=0.3)

        plt.plot(*abs_traj[:2], "o-", alpha=0.5, label="pred-abs")
        plt.quiver(*abs_traj[:4], alpha=0.3)

        plt.plot(*invtraj[:2], "o-", alpha=0.5, label="pred-inv")
        plt.quiver(*invtraj[:4], alpha=0.3)

        plt.plot(*merge_traj_1[:2], "o-", alpha=0.5, label="pred-merge-absinv")
        plt.quiver(*merge_traj_1[:4], alpha=0.3)

        # plt.plot(*merge_traj[:2], "o-", alpha=0.5, label="pred-merge-all")
        # plt.quiver(*merge_traj[:4], alpha=0.3)

        goal_xyz = to_tensor(self.world_data[start_t : start_t + 1, 12:15])
        goal_xyz = matrix.get_relative_position_to(goal_xyz, world_mat[start_t])
        plt.plot(goal_xyz[:, 0], goal_xyz[:, 2], "x", label="target")
        plt.legend()
        plt.show()

        # error = np.abs(actor_data.reshape(T, -1) - msg[:, :330]).sum(axis=-1).mean()
        # print(f"Error: {error}")
        self.client.close()

    def send_completetraj(self, N=10):
        self.connect()
        start_t = 0
        # start_t = 5279
        T_ = N
        T_interval = 60
        target_t = start_t + T_interval * T_ + 1
        traj_data = self.input_data[start_t:target_t, 330:452]
        I_data = self.input_data[start_t:target_t, -2048:]
        env_data = self.world_data[start_t:target_t, -315:]
        world_data = self.world_data[start_t:target_t, :12]
        msg = np.concatenate((traj_data, I_data, world_data, env_data), axis=-1)[
            ::T_interval
        ]
        msg = msg.astype(np.float32)

        T = msg.shape[0]
        l = msg.shape[0] * msg.shape[1]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        traj_vec = self.receive_data()
        traj_window = self.receive_data()
        return

        traj_vec = traj_vec.reshape(-1, 60, 12)
        abstraj_vec = traj_vec[..., 4:8]
        invtraj_vec = traj_vec[..., 8:12]
        traj_vec = traj_vec[..., :4]

        world_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, :12]))
        goal_mat = matrix.vec2mat_batch(to_tensor(self.world_data[:, 12:24]))
        # gt_traj_vec = to_tensor(self.output_data[start_t : start_t + T, 384:388])
        # traj_world_mat = relative_trajvec2worldmat(gt_traj_vec)
        # traj_vec = matrix.mat2vec_batch(traj_world_mat)
        # traj_vec = matrix.project_vec(traj_vec)  # [l, 4]

        traj_all = []
        abstraj_all = []
        invtraj_all = []
        mergetraj_all = []

        for i, (traj_vec_seq, abstraj_vec_seq, invtraj_vec_seq) in enumerate(
            zip(traj_vec, abstraj_vec, invtraj_vec)
        ):
            relative_start_mat = matrix.get_mat_BtoA(
                world_mat[start_t],
                world_mat[start_t + T_interval * i],
            )
            traj = trajvec2points(traj_vec_seq, relative_start_mat)
            abstraj = abstrajvec2points(abstraj_vec_seq, relative_start_mat)
            invtraj = invtrajvec2points(
                invtraj_vec_seq,
                world_mat[start_t + T_interval * (i + 1)],
                world_mat[start_t + T_interval * i],
                relative_start_mat,
            )
            mergetraj = merge_abs_inv_traj(abstraj, invtraj)
            mergetraj = merge_abs_inv_traj(traj, mergetraj)
            traj_all.append(traj)
            abstraj_all.append(abstraj)
            invtraj_all.append(invtraj)
            mergetraj_all.append(mergetraj)

        world_xyz = to_tensor(self.world_data[start_t:target_t, :3])
        new_xyz = matrix.get_relative_position_to(world_xyz, world_mat[start_t])
        new_xyz = new_xyz - new_xyz[:1]
        new_xyz = to_cpu_numpy(new_xyz)

        for i, (traj, abstraj, invtraj, mergetraj) in enumerate(
            zip(traj_all, abstraj_all, invtraj_all, mergetraj_all)
        ):
            plt.plot(new_xyz[:, 0], new_xyz[:, 2], "o-", alpha=0.2, label="gt")

            plt.plot(*traj[:2], "o-", alpha=0.2, label="pred")
            # plt.quiver(*traj[:4], alpha=0.3)

            plt.plot(*abstraj[:2], "o-", alpha=0.2, label="pred-abs")
            # plt.quiver(*abstraj[:4], alpha=0.3)

            plt.plot(*invtraj[:2], "o-", alpha=0.2, label="pred-inv")
            # plt.quiver(*invtraj[:4], alpha=0.3)

            plt.plot(*mergetraj[:2], "o-", alpha=0.2, label="pred-merge")
            plt.quiver(*mergetraj[:4], alpha=0.3)

            start_xyz = matrix.mat2vec(world_mat[start_t + i * T_interval])[None, :3]
            goal_xyz = matrix.mat2vec(world_mat[start_t + (i + 1) * T_interval])[
                None, :3
            ]
            start_xyz = matrix.get_relative_position_to(start_xyz, world_mat[start_t])
            goal_xyz = matrix.get_relative_position_to(goal_xyz, world_mat[start_t])
            plt.plot(
                start_xyz[:, 0], start_xyz[:, 2], "x", label="start", markersize=30
            )
            plt.plot(goal_xyz[:, 0], goal_xyz[:, 2], "x", label="target", markersize=30)

            plt.legend()
            plt.show()
