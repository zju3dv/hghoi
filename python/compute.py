from multiprocessing import Pool
import os
import numpy as np
import time
import datetime
import torch
import argparse
import matplotlib.pyplot as plt

from motion.utils.basics import DictwithEmpty
from motion.utils.utils import pd_load, to_tensor, to_cpu_numpy, to_cuda
from motion.utils.fid import calculate_fid
from motion.utils.distance import frechet_dist
import motion.utils.matrix as matrix

Idle_label = 0
INTERVAL = 30

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "--type",
    type=str,
    required=True,
    help="nsm or samp",
)
parser.add_argument(
    "--func",
    default="FD",
    type=str,
    help="Modify config options from command line",
)
parser.add_argument(
    "--j",
    default=8,
    type=int,
    help="number of multiprocess",
)
parser.add_argument(
    "--kind",
    type=str,
    required=False,
    default="",
    help="armchair",
)
parser.add_argument(
    "--nostyle",
    action="store_true",
)

args = parser.parse_args()

PREDICTION_PATH = args.path
DATA_TYPE = args.type
kind = args.kind
kind_ = ""
if kind == "armchair":
    kind_ = kind + "_"
if args.type == "nsm":
    FID_PATH = "./datasets/nsm/MotionWorld/train"
    FID_PATH = "./datasets/nsm/OriginalData/train"
    GT_PATH = "./datasets/nsm/MotionWorld/test"
    OBJ_NUM = 1
    POSE_DIM = 276
    LF_IND = 22
    RF_IND = 18
    STYLE_DIM = 7
    SIT_LABEL = 5
    LIEDOWN_LABEL = -1
elif args.type == "couch":
    FID_PATH = f"./datasets/couch/MotionSequence/train"
    GT_PATH = "./datasets/couch/MotionSequence/test"
    OBJ_NUM = 1
    POSE_DIM = 264
    LF_IND = 4
    RF_IND = 8
    STYLE_DIM = 3
    START_TRAJ = 330
    SIT_LABEL = 2
    LIEDOWN_LABEL = -1
elif args.type == "samp":
    FID_PATH = "./datasets/samp/MotionWorldIEnvCuboid/train"
    FID_PATH = "./datasets/samp/OriginalData/train"
    GT_PATH = "./datasets/samp/MotionWorldIEnvCuboid/test"
    OBJ_NUM = 3
    POSE_DIM = 264
    LF_IND = 4
    RF_IND = 8
    STYLE_DIM = 5
    SIT_LABEL = 3
    LIEDOWN_LABEL = 4
    START_TRAJ = 330
else:
    raise ValueError("dataset type should be in samp or nsm!")

P_T = 60  # TDNS on samp
A_T = 90  # TDNS on samp
# P_T = 90 # mypred on SAMP not affect
# A_T = 30 # mypred on SAMP not affect


pred = pd_load(os.path.join(PREDICTION_PATH, "Pred.txt")).to_numpy()
pred_id = pd_load(os.path.join(PREDICTION_PATH, "Sequences.txt")).to_numpy()[:, 0]
if os.path.exists(os.path.join(PREDICTION_PATH, "Penetration.txt")):
    penetration_total = pd_load(
        os.path.join(PREDICTION_PATH, "Penetration.txt")
    ).to_numpy()
    print("Load from penetration!")
else:
    penetration_total = None
if kind == "armchair":
    N = penetration_total.shape[0]
    pred = pred[:N]
    pred_id = pred_id[:N]
    kind = ""

gt = pd_load(os.path.join(GT_PATH, "Input.txt")).to_numpy()
gt_root = pd_load(os.path.join(GT_PATH, "World.txt")).to_numpy()
gt_id = pd_load(os.path.join(GT_PATH, "Sequences.txt")).to_numpy()[:, 0]

ALL_POSE = {"pos": [], "rot": []}
ALL_TRAJ = {"pos": [], "rot": []}
ALL_MOTION = {"pos-rot": []}
ERROR = {"pos": [], "angle": [], "pene_ratio": [], "penetration": [], "sliding": []}
FD = {
    "pose_pos": {},
    "pose_rot": {},
    "traj_pos": [],
    "traj_rot": [],
    "pred_sliding": [],
    "gt_sliding": [],
    "pene_ratio": [],
    "penetration": [],
}
FID = {}
SPLIT_FID = DictwithEmpty({})
TEST_N = pred_id.max() // 1000
SAMPLE_NUM = (pred_id.max() - TEST_N * 1000) // OBJ_NUM
NOSTYLE_PRED = args.nostyle
if args.type == "couch":
    TEST_N = 30
    SAMPLE_NUM = 5
    OBJ_NUM = 1


def find_interaction_t(style, interaction_label, threshold=0.7):
    max_val = style[:, interaction_label].max()
    interaction_inds = (style[:, interaction_label] >= threshold * max_val).nonzero()[0]
    interaction_t = interaction_inds[0]
    interaction_last_t = interaction_inds[-1]
    interaction_middle_t = (interaction_last_t + interaction_t) // 2
    if len(interaction_inds) > 240:
        interaction_middle_t = interaction_last_t - 120
    last_val = style[-1, interaction_label]
    print(max_val)
    print(last_val)
    return interaction_middle_t, max_val > 0.7 and last_val < 0.3


def mvec2cmvec(vec, pose_dim):
    """_summary_

    Args:
        vec (tensor): [N, 264+4] pose (264) + root (2 + 2)
    """
    pose = vec[..., :pose_dim]
    N = pose_dim // 12
    pose = pose.reshape(-1, N, 12)
    cm_pose = pose[..., :3] * 100
    cm_pose = cm_pose.reshape(-1, N * 3)
    rot_pose = pose[..., 3:9].reshape(-1, N * 6)
    root = vec[..., -4:]
    root_cm = root[..., -4:-2] * 100
    root_deg = np.arctan2(root[..., -2], root[..., -1])
    root_deg = root_deg[..., np.newaxis] / np.pi * 180
    # return np.concatenate((cm_pose, root_cm, root_deg), axis=-1)
    return np.concatenate((cm_pose, rot_pose, root_cm, root[..., -2:]), axis=-1)


def process_predgtvec(
    pred_seq,
    penetration_seq,
    gt_root_seq,
    gt_seq,
    pose_dim,
    style_dim,
    lf_ind,
    rf_ind,
    data_type,
    sit_label,
    liedown_label,
):
    ind = 0
    gt_root_seq = gt_root_seq[:, :6]
    gt_rot_mat = matrix.get_rot_mat_from_forward(gt_root_seq[:, 3:])
    gt_mat = matrix.get_TRS(gt_rot_mat, gt_root_seq[:, :3])
    gt_root_vec = matrix.get_mat_BtoA(gt_mat[ind], gt_mat)
    gt_root_vec = matrix.mat2vec_batch(gt_root_vec)
    gt_root_vec = matrix.project_vec(gt_root_vec)

    pred_pose_seq = pred_seq[:, :pose_dim]
    n_dim = pose_dim + style_dim
    pred_rot = matrix.get_rot_mat_from_forward(pred_seq[:, n_dim + 3 : n_dim + 6])
    pred_mat = matrix.get_TRS(pred_rot, pred_seq[:, n_dim : n_dim + 3])
    pred_root_vec = matrix.get_mat_BtoA(gt_mat[ind], pred_mat)
    pred_root_vec = matrix.mat2vec_batch(pred_root_vec)
    pred_root_vec = matrix.project_vec(pred_root_vec)

    gt_pose_seq = gt_seq[:, :pose_dim]

    gt_vec = np.concatenate((gt_pose_seq, gt_root_vec), axis=-1)
    pred_vec = np.concatenate((pred_pose_seq, pred_root_vec), axis=-1)
    gt_vec = mvec2cmvec(gt_vec, pose_dim)
    pred_vec = mvec2cmvec(pred_vec, pose_dim)

    pred_style_vec = pred_seq[:, pose_dim : pose_dim + style_dim]
    start_style = pose_dim // 12 * 15 + 6 * (style_dim + 4) + 4
    gt_style_vec = gt_seq[:, start_style : start_style + style_dim]
    interaction_label = sit_label
    if liedown_label >= 0:
        if gt_style_vec[:, liedown_label].sum() > 0:
            interaction_label = liedown_label

    pred_pose_dict = split_sequence(
        pred_pose_seq, pred_style_vec, interaction_label, 0.9, False
    )
    gt_pose_dict = split_sequence(
        gt_pose_seq, gt_style_vec, interaction_label, 0.9, False
    )

    pred_pose_seq = np.concatenate(
        (pred_pose_dict["Approach"], pred_pose_dict["Leaving"]), axis=0
    )
    gt_pose_seq = np.concatenate(
        (gt_pose_dict["Approach"], gt_pose_dict["Leaving"]), axis=0
    )

    pred_mat = split_sequence(pred_mat, pred_style_vec, interaction_label, 0.9, False)
    pred_mat = np.concatenate((pred_mat["Approach"], pred_mat["Leaving"]), axis=0)

    gt_mat = split_sequence(gt_mat, gt_style_vec, interaction_label, 0.9, False)
    gt_mat = np.concatenate((gt_mat["Approach"], gt_mat["Leaving"]), axis=0)

    gt_sliding = get_foot_sliding(gt_pose_seq, gt_mat, pose_dim, lf_ind, rf_ind)
    pred_sliding = get_foot_sliding(pred_pose_seq, pred_mat, pose_dim, lf_ind, rf_ind)
    pred_seq_dict = split_sequence(pred_seq, pred_style_vec, interaction_label, 0.9)
    pred_seq_vec = np.concatenate(
        (pred_seq_dict["Approach"], pred_seq_dict["Leaving"]), axis=0
    )
    pred_seq_vec = penetration_seq
    penetration_ratio = (
        1.0 * (pred_seq_vec[..., -1] > 0).sum() / pred_seq_vec[..., -1].shape[0]
    )
    penetration = pred_seq_vec[..., -1].mean()
    return (
        pred_vec,
        gt_vec,
        pred_sliding,
        gt_sliding,
        pred_style_vec,
        gt_style_vec,
        penetration_ratio,
        penetration,
    )


def calculate_sliding(foot_pos, root, data_type):
    pos = matrix.get_position_from(foot_pos[..., None, :3], root)[..., 0, :]
    vel = pos[1:] - pos[:-1]
    vel = torch.cat((torch.zeros(1, 3, device=vel.device), vel), dim=0)
    vel_g = torch.cat((vel[:, :1], vel[:, -1:]), dim=-1)
    vel_y = vel[:, 1]
    idle_y = pos[0, 1]
    pos_y = pos[..., 1]
    # make the idle as the ground
    pos_y = pos_y - idle_y
    # make negative ones as zero
    pos_y[pos_y < 0] = 0.0
    h_threshold = 0.025
    contact_flag = pos_y < h_threshold
    distance = torch.norm(vel_g[contact_flag], dim=-1)
    distance = distance * (2 - 2 ** (pos_y[contact_flag] / h_threshold))
    return (distance.sum() / vel.shape[0]).item()


def calculate_sliding2(foot_pos, root):
    pos = matrix.get_position_from(foot_pos[..., None, :3], root)[..., 0, :]
    vel = pos[1:] - pos[:-1]
    vel = np.concatenate((np.zeros((1, 3)), vel), axis=0)
    vel_g = np.concatenate((vel[:, :1], vel[:, -1:]), axis=-1)
    pos_y = pos[..., 1]
    distance = np.linalg.norm(vel_g, ord=2, axis=-1)
    return pos_y, distance


def get_foot_sliding(pose, root, pose_dim, lf_ind, rf_ind):
    N = pose_dim // 12
    pose = pose.reshape(-1, N, 12)
    # samp: 4, 8; nsm: 22, 18
    lf = pose[:, lf_ind]
    rf = pose[:, rf_ind]
    # lf_sliding = calculate_sliding(lf, root, data_type)
    # rf_sliding = calculate_sliding(rf, root, data_type)
    pos_lf, lf_sliding = calculate_sliding2(lf, root)
    pos_rf, rf_sliding = calculate_sliding2(rf, root)
    mask = pos_lf > pos_rf
    sliding = rf_sliding * mask + lf_sliding * ~mask
    return 100 * sliding.mean().item()


def calculate_apd(pose):
    apd = torch.norm(pose[None] - pose[:, None], dim=-1)
    mask = torch.triu(torch.ones_like(apd), diagonal=1).bool()
    apd = apd[mask].sum() / mask.sum()
    return apd


def calculate_pose(
    gt_seq,
    pred_seq,
    penetration_seq,
    pose_dim,
    style_dim,
    sit_label,
    liedown_label,
    lf_ind,
    rf_ind,
):
    N = pose_dim // 12
    start_style = pose_dim + N * 3 + 6 * (style_dim + 4) + 4
    if gt_seq is not None:
        gt_style = gt_seq[:, start_style : start_style + style_dim]
        interaction_label = sit_label
        if liedown_label >= 0:
            if gt_style[:, liedown_label].sum() > 0:
                interaction_label = liedown_label
    else:
        interaction_label = sit_label
    pred_style = pred_seq[:, pose_dim : pose_dim + style_dim]
    if NOSTYLE_PRED:
        pred_style = np.zeros_like(pred_style)
        v = (pred_seq[1:] - pred_seq[:-1]).sum(axis=-1)
        v0 = np.where(v == 0)[0]
        interact_t = pred_style.shape[0] // 2
        for i in range(v0.shape[0] - 30):
            if v0[i] + 30 == v0[i + 30]:
                interact_t = v0[i]
                break
        pred_style[: interact_t - P_T, 1] = 1
        pred_style[interact_t - P_T : interact_t + A_T, interaction_label] = 1
        pred_style[interact_t + A_T :, 1] = 1
    pred_t, success_flag = find_interaction_t(
        pred_style, interaction_label, threshold=0.9
    )
    # if NOSTYLE_PRED:
    #     pred_t = interact_t

    pred_pose = pred_seq[:, :pose_dim]
    root_pos = pred_pose[:, :3]
    pose_rot = pred_pose.reshape(-1, N, 12)[:, :, 3:9]
    pose_rot = pose_rot.reshape(-1, N * 6)
    pred_motion_vec = np.concatenate((root_pos, pose_rot), axis=-1)

    n_dim = pose_dim + style_dim
    pred_pos = pred_seq[..., n_dim : n_dim + 3]
    pred_forward = pred_seq[..., n_dim + 3 : n_dim + 6]
    goal_pos = pred_seq[..., n_dim + 6 : n_dim + 9]
    # we only compare the position and orientation on floor, since pelvis is calculated according pose
    # pred_pos_ = pred_pos + pred_seq[..., :3]
    pred_pos_ = pred_pos
    goal_pos[..., 1] = 0
    pos_error = np.linalg.norm(goal_pos - pred_pos_, ord=2, axis=-1) * 100
    goal_pos[..., 1] = 0
    goal_forward = pred_seq[..., n_dim + 9 : n_dim + 12]
    pred_rot = matrix.get_rot_mat_from_forward(pred_forward)

    pred_mat = matrix.get_TRS(pred_rot, pred_pos)
    pred_mat_rel = matrix.get_mat_BtoA(pred_mat[:1], pred_mat)
    pred_root_vec = matrix.mat2vec_batch(pred_mat_rel)
    pred_root_vec = matrix.project_vec(pred_root_vec)
    selected_ind = [i for i in range(0, pred_root_vec.shape[0], INTERVAL)]
    # if pred_root_vec.shape[0] - 1 not in selected_ind:
    # selected_ind.append(pred_root_vec.shape[0] - 1)
    pred_root_vec = pred_root_vec[selected_ind]
    pred_motion_vec = pred_motion_vec[selected_ind]
    # pred_motion_vec = matrix.get_relative_pose_from_vec(
    #     pred_pose[selected_ind], pred_root_vec, N
    # )
    pred_motion_vec[:, 0] = 0.0
    pred_motion_vec[:, 2] = 0.0

    goal_rot = matrix.get_rot_mat_from_forward(goal_forward)
    delta_rot = matrix.get_mat_BtoA(goal_rot, pred_rot)
    delta_angle = np.arctan2(delta_rot[..., 0, 2], delta_rot[..., 2, 2])
    angle_error = np.abs(delta_angle) / np.pi * 180
    pos_error = pos_error[pred_t]
    angle_error = angle_error[pred_t]

    pred_pose = pred_seq[pred_t, :pose_dim]
    pred_pose = pred_pose.reshape(-1, N, 12)
    pred_pose_pos = pred_pose[..., :3] * 100
    pred_pose_rot = pred_pose[..., 3:9]

    pred_pose_dict = split_sequence(
        pred_seq[..., :pose_dim], pred_style, interaction_label, 0.9, False
    )
    pred_pose_seq = np.concatenate(
        (pred_pose_dict["Approach"], pred_pose_dict["Leaving"]), axis=0
    )
    pred_mat = split_sequence(pred_mat, pred_style, interaction_label, 0.9, False)
    pred_mat = np.concatenate((pred_mat["Approach"], pred_mat["Leaving"]), axis=0)

    pred_sliding = get_foot_sliding(pred_pose_seq, pred_mat, pose_dim, lf_ind, rf_ind)

    pred_seq_dict = split_sequence(
        pred_seq, pred_style, interaction_label, threshold=0.9
    )
    pred_seq_vec = np.concatenate(
        (pred_seq_dict["Approach"], pred_seq_dict["Leaving"]), axis=0
    )
    pred_seq_vec = penetration_seq
    penetration_ratio = (
        1.0 * (pred_seq_vec[..., -1] > 0).sum() / pred_seq_vec[..., -1].shape[0]
    )
    penetration = pred_seq_vec[..., -1].mean()

    ratio = 1.0 * pred_t / len(pred_seq) * 100
    print_str = f"Error at {pred_t}/{len(pred_seq)} - ({ratio}%)\n"
    print_str += f"    pos: {pos_error}; angle: {angle_error};\n"
    print_str += f"    pene_ratio: {penetration_ratio}; penetration: {penetration};"
    print(print_str)
    if success_flag:
        ERROR["pos"].append(pos_error)
        ERROR["angle"].append(angle_error)
    else:
        print("Fail!")
        ERROR["pos"].append(pos_error)
        ERROR["angle"].append(angle_error)
        # ERROR["pos"].append(np.inf)
        # ERROR["angle"].append(np.inf)
    ERROR["pene_ratio"].append(penetration_ratio)
    ERROR["penetration"].append(penetration)
    ERROR["sliding"].append(pred_sliding)
    return (
        pred_pose_pos.reshape(1, -1),
        pred_pose_rot.reshape(1, -1),
        pred_motion_vec,
        pred_root_vec,
        pred_style,
        interaction_label,
        success_flag,
    )


def calculate_fid_feat(pred_seq, pose_dim, style_dim):
    N = pose_dim // 12
    n_dim = pose_dim + style_dim
    pred_pose = pred_seq[..., :pose_dim]

    pred_pos = pred_seq[..., n_dim : n_dim + 3]
    pred_forward = pred_seq[..., n_dim + 3 : n_dim + 6]
    pred_rot = matrix.get_rot_mat_from_forward(pred_forward)
    pred_mat = matrix.get_TRS(pred_rot, pred_pos)

    relxz = matrix.get_window_traj(pred_mat)
    relpose, _, _ = matrix.get_window_pose(pred_pose, relxz, N)

    relpose = relpose[30:-30]
    relxz = relxz[30:-30]
    feat = np.concatenate((relpose, relxz), axis=-1)

    return {"pose": relpose, "traj": relxz, "all": feat}


def split_sequence(sequence, style, interaction_label, threshold=0.5, is_expand=True):
    max_val = style[:, interaction_label].max()
    interaction_inds = (style[:, interaction_label] >= threshold * max_val).nonzero()[0]
    if interaction_label == 4:
        interaction_key = "Liedown"
    else:
        interaction_key = "Sit"
    last_val = style[-1, interaction_label]

    t1 = interaction_inds[0]
    t2 = interaction_inds[-1]
    if is_expand:
        if t1 > 30:
            t1 -= 30
        if t2 < sequence.shape[0] - 30:
            t2 += 30
    if args.type == "couch":
        MIN_FRAMES = 30
    else:
        MIN_FRAMES = 60
    if t2 > sequence.shape[0] - MIN_FRAMES:
        t2 = sequence.shape[0] - MIN_FRAMES
    if t1 >= t2:
        t2 = t1 + 1
    if len(interaction_inds) == 0 or max_val < 0.7:
        return {
            "Approach": sequence,
            interaction_key: sequence,
            "Leaving": sequence,
        }
    if args.type == "couch":
        return {
            "Approach": sequence[:t1],
            interaction_key: sequence[t1:t2],
            "Leaving": sequence[t2:],
        }
    if last_val > 0.3:
        return {
            "Approach": sequence[:t1],
            interaction_key: sequence[t1:],
            "Leaving": sequence[t1:],
        }

    return {
        "Approach": sequence[:t1],
        interaction_key: sequence[t1:t2],
        "Leaving": sequence[t2:],
    }


def sequence_test(
    gt_seq,
    gt_root_seq,
    pred_seq,
    penetration_seq,
    pose_dim,
    style_dim,
    lf_ind,
    rf_ind,
    data_type,
    sit_label,
    liedown_label,
    fd,
):
    (
        pred_vec,
        gt_vec,
        pred_sliding,
        gt_sliding,
        pred_style,
        gt_style,
        penetration_ratio,
        penetration,
    ) = process_predgtvec(
        pred_seq,
        penetration_seq,
        gt_root_seq,
        gt_seq,
        pose_dim,
        style_dim,
        lf_ind,
        rf_ind,
        data_type,
        sit_label,
        liedown_label,
    )
    interaction_label = sit_label
    if liedown_label >= 0:
        if gt_style[:, liedown_label].sum() > 0:
            interaction_label = liedown_label

    N = pose_dim // 12
    pred_vec_dict = split_sequence(pred_vec, pred_style, interaction_label)
    gt_vec_dict = split_sequence(gt_vec, gt_style, interaction_label)
    if liedown_label >= 0:
        fd_keys = ["Approach", "Leaving"]
        pred_vec_dict = split_sequence(pred_vec, pred_style, interaction_label)
        gt_vec_dict = split_sequence(gt_vec, gt_style, interaction_label)
    else:
        fd_keys = ["Approach", "Sit", "Leaving"]
        pred_vec_dict = split_sequence(pred_vec, pred_style, interaction_label)
        gt_vec_dict = split_sequence(gt_vec, gt_style, interaction_label)
        fd_keys = ["All"]
        pred_vec_dict = {"All": pred_vec}
        gt_vec_dict = {"All": gt_vec}
    print_str = "FD:\n"
    for k in fd_keys:
        pose_pos_fd = frechet_dist(
            pred_vec_dict[k][:, : N * 3], gt_vec_dict[k][:, : N * 3]
        )
        pose_rot_fd = frechet_dist(
            pred_vec_dict[k][:, N * 3 : N * 9], gt_vec_dict[k][:, N * 3 : N * 9]
        )
        print_str += f"    {k}: pos-{pose_pos_fd}; rot-{pose_rot_fd};\n"
        if k not in fd["pose_pos"].keys():
            fd["pose_pos"][k] = []
            fd["pose_rot"][k] = []

        fd["pose_pos"][k].append(pose_pos_fd)
        fd["pose_rot"][k].append(pose_rot_fd)

    f_dist_pos = frechet_dist(pred_vec[:, -4:-2], gt_vec[:, -4:-2])
    f_dist_rot = frechet_dist(pred_vec[:, -2:], gt_vec[:, -2:])
    print_str += f"    traj: pos-{f_dist_pos}; rot-{f_dist_rot};\n"
    print_str += f"Sliding: {pred_sliding}; GT: {gt_sliding}\n"
    print_str += f"Pene_ratio: {penetration_ratio}; Penetration: {penetration}"

    fd["traj_pos"].append(f_dist_pos)
    fd["traj_rot"].append(f_dist_rot)
    fd["pred_sliding"].append(pred_sliding)
    fd["gt_sliding"].append(gt_sliding)
    fd["pene_ratio"].append(penetration_ratio)
    fd["penetration"].append(penetration)
    return print_str, fd


def calculate_traj(
    pred_seq,
    pose_dim,
    style_dim,
):
    N = pose_dim // 12
    n_dim = pose_dim + style_dim
    pred_pos = pred_seq[..., n_dim : n_dim + 3]
    pred_forward = pred_seq[..., n_dim + 3 : n_dim + 6]
    goal_pos = pred_seq[..., n_dim + 6 : n_dim + 9]
    goal_forward = pred_seq[..., n_dim + 9 : n_dim + 12]
    goal_rot = matrix.get_rot_mat_from_forward(goal_forward)
    goal_mat = matrix.get_TRS(goal_rot, goal_pos)

    pred_rot = matrix.get_rot_mat_from_forward(pred_forward)

    pred_mat = matrix.get_TRS(pred_rot, pred_pos)
    pred_mat_rel = matrix.get_mat_BtoA(goal_mat[:1], pred_mat)
    pred_root_vec = matrix.mat2vec_batch(pred_mat_rel)
    pred_root_vec = matrix.project_vec(pred_root_vec)
    goal_vec = matrix.project_vec(matrix.mat2pose_batch(goal_mat))

    return pred_root_vec, goal_vec


def calculate_time(
    gt_seq,
    pred_seq,
):
    N = POSE_DIM // 12
    start_style = POSE_DIM + N * 3 + 6 * (STYLE_DIM + 4) + 4
    if gt_seq is not None:
        gt_style = gt_seq[:, start_style : start_style + STYLE_DIM]
        interaction_label = SIT_LABEL
        if LIEDOWN_LABEL >= 0:
            if gt_style[:, LIEDOWN_LABEL].sum() > 0:
                interaction_label = LIEDOWN_LABEL
    else:
        interaction_label = SIT_LABEL
    if NOSTYLE_PRED:
        v = (pred_seq[1:] - pred_seq[:-1]).sum(axis=-1)
        v0 = np.where(v == 0)[0]
        interact_t = pred_seq.shape[0] // 2
        for i in range(v0.shape[0] - 30):
            if v0[i] + 30 == v0[i + 30]:
                interact_t = v0[i]
                break
    else:
        pred_style = pred_seq[:, POSE_DIM : POSE_DIM + STYLE_DIM]
        interact_t = np.where(
            pred_style[:, interaction_label]
            > 0.95 * pred_style[:, interaction_label].max()
        )[0][0]
    return interact_t


def test_FD(args):
    pred_ind, gt_ind = args
    start_time = time.time()
    test_n = 0
    fd = {
        "pose_pos": {},
        "pose_rot": {},
        "traj_pos": [],
        "traj_rot": [],
        "pred_sliding": [],
        "gt_sliding": [],
        "pene_ratio": [],
        "penetration": [],
    }

    for p_i, g_i in zip(pred_ind, gt_ind):
        pred_seq = pred[pred_id == p_i]
        gt_seq = gt[gt_id == g_i]
        gt_root_seq = gt_root[gt_id == g_i]
        if penetration_total is not None:
            penetration_seq = penetration_total[pred_id == p_i]
        else:
            penetration_seq = pred_seq[..., -1:]

        seq_i = p_i // 1000
        sample_i = (p_i - seq_i * 1000 - 1) // OBJ_NUM + 1
        obj_i = (p_i - seq_i * 1000) % OBJ_NUM + 1

        print(
            f"Process {os.getpid()}: Sequence {seq_i}/{TEST_N} - Sample {sample_i}/{SAMPLE_NUM} - OBJ {obj_i}/{OBJ_NUM}:"
        )
        print_str, fd = sequence_test(
            gt_seq,
            gt_root_seq,
            pred_seq,
            penetration_seq,
            POSE_DIM,
            STYLE_DIM,
            LF_IND,
            RF_IND,
            DATA_TYPE,
            SIT_LABEL,
            LIEDOWN_LABEL,
            fd,
        )
        test_n += 1
        used_time = time.time() - start_time
        eta_time = 1.0 * used_time / test_n * (len(pred_ind) - test_n)
        used_time = str(datetime.timedelta(seconds=int(used_time)))
        eta_time = str(datetime.timedelta(seconds=int(eta_time)))
        print(
            f"Process {os.getpid()} used_time: {used_time}, eta: {eta_time}\n"
            + print_str
        )
    return fd


def measure_FD():
    print(f"Start to test {PREDICTION_PATH} FD")
    start_time = time.time()

    total_n = SAMPLE_NUM * TEST_N * OBJ_NUM
    pred_ind = []
    gt_ind = []
    for i in range(TEST_N):
        n = 0
        for j in range(i):
            n += OBJ_NUM

        for k in range(SAMPLE_NUM):
            for j in range(OBJ_NUM):
                pred_ind.append((i + 1) * 1000 + k * OBJ_NUM + j + 1)
                gt_ind.append(n + j + 1)
    p = Pool(args.j)
    N = args.j
    p_N = total_n // N
    split_args = [
        (pred_ind[i * p_N : (i + 1) * p_N], gt_ind[i * p_N : (i + 1) * p_N])
        for i in range(N - 1)
    ]
    split_args.append((pred_ind[(N - 1) * p_N :], gt_ind[(N - 1) * p_N :]))

    fds = p.map(test_FD, split_args)
    p.close()
    p.join()
    for fd in fds:
        for k, v in fd.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    if k_ not in FD[k]:
                        FD[k][k_] = v_
                    else:
                        FD[k][k_].extend(v_)
            else:
                FD[k].extend(v)
    used_time = time.time() - start_time
    used_time = str(datetime.timedelta(seconds=int(used_time)))
    print(f"Costs {used_time} in total...")

    print(f"{PREDICTION_PATH} FD:")
    pos_all = []
    rot_all = []
    print("Pose:")
    for k_, v_ in FD["pose_pos"].items():
        if len(k_) < 8:
            print(f"{k_}\t\tPos\tRot")
        else:
            print(f"{k_}\tPos\tRot")
        print(
            f"Mean\t\t{np.array(v_).mean():.2f}\t{np.array(FD['pose_rot'][k_]).mean():.2f}"
        )
        print(
            f"Std\t\t{np.array(v_).std():.2f}\t{np.array(FD['pose_rot'][k_]).std():.2f}"
        )
        pos_all.extend(v_)
        rot_all.extend(FD["pose_rot"][k_])
    print("Average\t\tPos\tRot")
    print(f"Mean\t\t{np.array(pos_all).mean():.2f}\t{np.array(rot_all).mean():.2f}")
    print(f"Std\t\t{np.array(pos_all).std():.2f}\t{np.array(rot_all).std():.2f}")
    for k, v in FD.items():
        if "pose" not in k:
            print(f"{k}\tMean\tStd")
            if "penetration" in k:
                print(f"\t\t{np.array(v).mean():.4f}\t{np.array(v).std():.4f}")
            else:
                print(f"\t\t{np.array(v).mean():.2f}\t{np.array(v).std():.2f}")


def measure_APD():
    start_time = time.time()
    print(f"Start to test {PREDICTION_PATH} APD")
    test_n = 0
    TEST_N = pred_id.max() // 1000
    SAMPLE_NUM = (pred_id.max() - TEST_N * 1000) // OBJ_NUM
    total_n = SAMPLE_NUM * TEST_N * OBJ_NUM
    if args.type == "couch":
        TEST_N = 30
        SAMPLE_NUM = 5
        total_n = TEST_N * SAMPLE_NUM
    fid_feats = DictwithEmpty([])
    fid_split_feats = DictwithEmpty(DictwithEmpty([]))
    for i in range(TEST_N):
        n = 0
        for j in range(i):
            n += OBJ_NUM
        all_pose = {
            "pos": [[] for _ in range(OBJ_NUM)],
            "rot": [[] for _ in range(OBJ_NUM)],
        }
        trajs = {
            "pos": [[] for _ in range(OBJ_NUM)],
            "rot": [[] for _ in range(OBJ_NUM)],
        }
        motions = {
            "pos-rot": [[] for _ in range(OBJ_NUM)],
        }

        for k in range(SAMPLE_NUM):
            for j in range(OBJ_NUM):
                matching_ind = (i + 1) * 1000 + k * OBJ_NUM + j + 1
                if args.type == "couch":
                    matching_ind = i * SAMPLE_NUM + k + 1
                    # if NOSTYLE_PRED:
                    #     matching_ind -= 1

                pred_seq = pred[pred_id == matching_ind]

                if args.type == "couch":
                    gt_seq = None
                else:
                    gt_seq = gt[gt_id == n + j + 1]
                if penetration_total is not None:
                    penetration_seq = penetration_total[pred_id == matching_ind]
                else:
                    penetration_seq = pred_seq[..., -1:]

                print(
                    f"Sequence {i + 1}/{TEST_N} - Sample {k + 1}/{SAMPLE_NUM} - OBJ {j + 1}/{OBJ_NUM}:"
                )
                (
                    pos_pose,
                    rot_pose,
                    motion_vec,
                    root_vec,
                    pred_style,
                    interaction_label,
                    success_flag,
                ) = calculate_pose(
                    gt_seq,
                    pred_seq,
                    penetration_seq,
                    POSE_DIM,
                    STYLE_DIM,
                    SIT_LABEL,
                    LIEDOWN_LABEL,
                    LF_IND,
                    RF_IND,
                )

                fid_feat = calculate_fid_feat(pred_seq, POSE_DIM, STYLE_DIM)
                for k_, v_ in fid_feat.items():
                    feat_split_dict = split_sequence(v_, pred_style, interaction_label)
                    for k__, v__ in feat_split_dict.items():
                        fid_split_feats[k__][k_].append(v__)

                    fid_feats[k_].append(v_)

                if success_flag:
                    all_pose["pos"][j].append(pos_pose)
                    all_pose["rot"][j].append(rot_pose)
                    trajs["pos"][j].append(root_vec[..., :2] * 100.0)
                    trajs["rot"][j].append(root_vec[..., 2:])
                    motions["pos-rot"][j].append(motion_vec)
                test_n += 1
                used_time = time.time() - start_time
                eta_time = 1.0 * used_time / test_n * (total_n - test_n)
                used_time = str(datetime.timedelta(seconds=int(used_time)))
                eta_time = str(datetime.timedelta(seconds=int(eta_time)))
                print(f"used_time: {used_time}, eta: {eta_time}")
        for j in range(OBJ_NUM):
            for k_, v_ in trajs.items():
                fd = 0.0
                apd = 0.0
                for i1 in range(len(v_[j])):
                    for j1 in range(i1 + 1, len(v_[j])):
                        fd += frechet_dist(v_[j][i1], v_[j][j1])
                        apd += np.linalg.norm(all_pose[k_][j][i1] - all_pose[k_][j][j1])

                fd = 2.0 * fd / ((len(v_[j]) * (len(v_[j]) - 1)) + 1e-9)
                apd = 2.0 * apd / ((len(v_[j]) * (len(v_[j]) - 1)) + 1e-9)
                ALL_TRAJ[k_].append(fd)
                ALL_POSE[k_].append(apd)
            for k_, v_ in motions.items():
                fd = 0.0
                for i1 in range(len(v_[j])):
                    for j1 in range(i1 + 1, len(v_[j])):
                        fd += frechet_dist(v_[j][i1], v_[j][j1])

                fd = 2.0 * fd / ((len(v_[j]) * (len(v_[j]) - 1)) + 1e-9)
                ALL_MOTION[k_].append(fd)

    used_time = time.time() - start_time
    used_time = str(datetime.timedelta(seconds=int(used_time)))
    print(f"Costs {used_time} in total...")

    print(f"{PREDICTION_PATH} APD:")
    print(f"APD\tPose\tTraj\tMotion")
    for k, v_ in ALL_POSE.items():
        pose_apd = np.array(ALL_POSE[k]).mean()
        fd_apd = np.array(ALL_TRAJ[k]).mean()
        mfd_apd = np.array(ALL_MOTION["pos-rot"]).mean()
        print(f"{k}\t{pose_apd:.2f}\t{fd_apd:.2f}\t{mfd_apd:.2f}")

    used_time = time.time() - start_time
    used_time = str(datetime.timedelta(seconds=int(used_time)))

    print(f"{PREDICTION_PATH} Mean Error:")
    print(f"Error\t\tMean\tStd\tMin\tMax")
    for k, v in ERROR.items():
        if len(k) < 8:
            print(
                f"{k}\t\t{np.array(v).mean():.2f}\t{np.array(v).std():.2f}"
                f"\t{np.array(v).min():.2f}\t{np.array(v).max():.2f}"
            )
        else:
            if "pene" in k:
                print(
                    f"{k}\t{np.array(v).mean():.4f}\t{np.array(v).std():.4f}"
                    f"\t{np.array(v).min():.4f}\t{np.array(v).max():.4f}"
                )
            else:
                print(
                    f"{k}\t{np.array(v).mean():.2f}\t{np.array(v).std():.2f}"
                    f"\t{np.array(v).min():.2f}\t{np.array(v).max():.2f}"
                )
    # raise AssertionError
    acc_channel = 0
    for k in ["pose", "traj"]:
        v = np.concatenate(fid_feats[k], axis=0)
        FID[k] = calculate_fid(
            v, FID_PATH, acc_channel, acc_channel + v.shape[-1], gt_key=kind
        )
        acc_channel += v.shape[-1]
    v = np.concatenate(fid_feats["all"], axis=0)
    FID["all"] = calculate_fid(v, FID_PATH, gt_key=kind)
    used_time = time.time() - start_time
    used_time = str(datetime.timedelta(seconds=int(used_time)))
    print(f"Costs {used_time}...")
    print("Finish calculate FID on all data")

    for k, v in fid_split_feats.items():
        accu_c = 0
        for k_ in ["pose", "traj"]:
            v_ = np.concatenate(v[k_], axis=0)
            SPLIT_FID[k][k_] = calculate_fid(
                v_, FID_PATH, accu_c, accu_c + v_.shape[-1], kind_ + k
            )
            accu_c += v_.shape[-1]
        v_ = np.concatenate(v["all"], axis=0)
        SPLIT_FID[k]["all"] = calculate_fid(v_, FID_PATH, gt_key=kind_ + k)
    used_time = time.time() - start_time
    used_time = str(datetime.timedelta(seconds=int(used_time)))
    print(f"Costs {used_time}...")
    print("Finish calculate FID on different splits")

    print(f"{PREDICTION_PATH} FID:")
    print(f"FID\t\tValue")
    for k, v in FID.items():
        if len(k) < 8:
            print(f"{k}\t\t{v:.2f}")
        else:
            print(f"{k}\t{v:.2f}")
    Average_fid = DictwithEmpty([])
    print(f"Split\t\tPose\tTraj\tAll")
    for k, v in SPLIT_FID.items():
        print_str = ""
        if len(k) < 8:
            print_str += f"{k}\t\t"
        else:
            print_str += f"{k}\t"
        for k_, v_ in v.items():
            print_str += f"{v_:.2f}\t"
            Average_fid[k_].append(v_)
        print(print_str)
    print_str = "Average\t\t"
    for k, v in Average_fid.items():
        if len(v) == 4:
            print_str += f"{(v[0] + v[2] + (v[3] + v[1] * 5) / 6 ) / 3:.2f}\t"
        else:
            print_str += f"{(v[0] + v[1]+ v[2]) / 3.:.2f}\t"

    print(print_str)


def draw_traj():
    start_time = time.time()
    test_n = 0
    TEST_N = pred_id.max() // 1000
    SAMPLE_NUM = (pred_id.max() - TEST_N * 1000) // OBJ_NUM
    total_n = SAMPLE_NUM * TEST_N * OBJ_NUM
    if args.type == "couch":
        TEST_N = 30
        SAMPLE_NUM = 5
        total_n = TEST_N * SAMPLE_NUM
    for i in range(TEST_N):
        n = 0
        for j in range(i):
            n += OBJ_NUM
        trajs = {
            "pos": [[] for _ in range(OBJ_NUM)],
            "rot": [[] for _ in range(OBJ_NUM)],
            "goal": [[] for _ in range(OBJ_NUM)],
        }
        for k in range(SAMPLE_NUM):
            for j in range(OBJ_NUM):
                matching_ind = (i + 1) * 1000 + k * OBJ_NUM + j + 1
                if args.type == "couch":
                    matching_ind = i * SAMPLE_NUM + k + 1

                pred_seq = pred[pred_id == matching_ind]

                if args.type == "couch":
                    gt_seq = None
                else:
                    gt_seq = gt[gt_id == n + j + 1]
                if penetration_total is not None:
                    penetration_seq = penetration_total[pred_id == matching_ind]
                else:
                    penetration_seq = pred_seq[..., -1:]

                print(
                    f"Sequence {i + 1}/{TEST_N} - Sample {k + 1}/{SAMPLE_NUM} - OBJ {j + 1}/{OBJ_NUM}:"
                )
                root_vec, goal_vec = calculate_traj(pred_seq, POSE_DIM, STYLE_DIM)

                trajs["pos"][j].append(root_vec[..., :2])
                trajs["rot"][j].append(root_vec[..., 2:])
                trajs["goal"][j].append(goal_vec[..., :2])
                test_n += 1
                used_time = time.time() - start_time
                eta_time = 1.0 * used_time / test_n * (total_n - test_n)
                used_time = str(datetime.timedelta(seconds=int(used_time)))
                eta_time = str(datetime.timedelta(seconds=int(eta_time)))
                print(f"used_time: {used_time}, eta: {eta_time}")
        for j in range(OBJ_NUM):
            pos = trajs["pos"]  # [[np.array(N, 2)]]
            goal = trajs["goal"]
            for k in range(len(pos[j])):
                plt.plot(pos[j][k][:, 0], pos[j][k][:, 1])
                plt.plot(pos[j][k][0, 0], pos[j][k][0, 1], marker="^", color="black")
                plt.plot(0, 0, marker="o", color="black")
                plt.plot(pos[j][k][-1, 0], pos[j][k][-1, 1], marker="^", color="black")
            x = 3
            plt.xticks(np.arange(-x, x + 1, 1))
            plt.yticks(np.arange(-x, x + 1, 1))
            plt.show()
            raise NotImplementedError
            break


def draw_time():
    start_time = time.time()
    test_n = 0
    TEST_N = pred_id.max() // 1000
    SAMPLE_NUM = (pred_id.max() - TEST_N * 1000) // OBJ_NUM
    total_n = SAMPLE_NUM * TEST_N * OBJ_NUM
    if args.type == "couch":
        TEST_N = 30
        SAMPLE_NUM = 5
        total_n = TEST_N * SAMPLE_NUM
    all_time = []
    for i in range(TEST_N):
        n = 0
        for j in range(i):
            n += OBJ_NUM
        for k in range(SAMPLE_NUM):
            for j in range(OBJ_NUM):
                matching_ind = (i + 1) * 1000 + k * OBJ_NUM + j + 1
                if args.type == "couch":
                    matching_ind = i * SAMPLE_NUM + k + 1

                pred_seq = pred[pred_id == matching_ind]
                print(
                    f"Sequence {i + 1}/{TEST_N} - Sample {k + 1}/{SAMPLE_NUM} - OBJ {j + 1}/{OBJ_NUM}:"
                )
                if args.type == "couch":
                    gt_seq = None
                else:
                    gt_seq = gt[gt_id == n + j + 1]
                t = calculate_time(gt_seq, pred_seq)
                all_time.append((t // 60 + 1) * 2)
                test_n += 1
                used_time = time.time() - start_time
                eta_time = 1.0 * used_time / test_n * (total_n - test_n)
                used_time = str(datetime.timedelta(seconds=int(used_time)))
                eta_time = str(datetime.timedelta(seconds=int(eta_time)))
                print(f"used_time: {used_time}, eta: {eta_time}")
    all_time = np.array(all_time)
    train_gt = pd_load(os.path.join(FID_PATH, "Output.txt")).to_numpy()
    train_gt_id = pd_load(os.path.join(FID_PATH, "Sequences.txt")).to_numpy()[:, 0]
    train_gt_style = train_gt[
        :, START_TRAJ + (4 + STYLE_DIM) * 6 + 4 : START_TRAJ + (4 + STYLE_DIM) * 7
    ]
    train_gt_style = [
        train_gt_style[train_gt_id == i + 1] for i in range(train_gt_id.max())
    ]
    gt_time = []
    for gt_s in train_gt_style:
        interaction_label = SIT_LABEL
        if LIEDOWN_LABEL != -1:
            if gt_s[:, LIEDOWN_LABEL].sum() != 0:
                interaction_label = LIEDOWN_LABEL
        if gt_s[:, interaction_label].sum() == 0:
            continue
        gt_t = np.where(gt_s[:, interaction_label] == 1)[0][0]
        gt_time.append((gt_t // 60 + 1) * 2)
    gt_time = np.array(gt_time)
    max_t = max(all_time.max(), gt_time.max())
    all_t_distribution = np.array(
        [(all_time == i).sum() / all_time.shape[0] for i in range(2, max_t + 1, 2)]
    )
    gt_t_distribution = np.array(
        [(gt_time == i).sum() / gt_time.shape[0] for i in range(2, max_t + 1, 2)]
    )
    # plt.bar(np.arange(2, max_t+1, 2) + 1, gt_t_distribution, label="Dataset")
    # plt.bar(np.arange(2, max_t+1, 2), all_t_distribution, label="Prediction")
    plt.plot(np.arange(2, max_t + 1, 2), gt_t_distribution, label="Dataset")
    plt.plot(np.arange(2, max_t + 1, 2), all_t_distribution, label="Prediction")
    plt.legend()
    plt.show()


def measure_Pene():
    print(f"Start to test {PREDICTION_PATH} penetration")
    N = penetration_total.shape[0]
    pred_id_ = pred_id[:N]
    pe = [penetration_total[pred_id_ == i + 1000] for i in range(1, 21)]
    print("Threshold\tPenetration")
    for i in range(16):
        pene = np.array([(p > 0.01 * i).sum() / p.shape[0] for p in pe]).mean()
        print(f"{0.01*i}\t\t{pene:.3f}")


if args.func == "FD":
    measure_FD()
elif args.func == "APD":
    measure_APD()
elif args.func == "traj":
    draw_traj()
elif args.func == "time":
    draw_time()
else:
    measure_Pene()

    # raise ValueError("Should be FD or APD")
