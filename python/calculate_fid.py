import os
import numpy as np
from motion.utils.basics import DictwithEmpty
from motion.utils.utils import pd_load
from motion.utils.fid import calculate_fid
import motion.utils.matrix as matrix
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    required=True,
    help="nsm or samp or couch",
)
parser.add_argument(
    "--split",
    type=str,
    required=False,
    default="test",
    help="test or tran",
)

parser.add_argument(
    "--kind",
    type=str,
    required=False,
    default="",
    help="armchair",
)

args = parser.parse_args()

dataset = args.type
split = args.split
kind = args.kind
if dataset == "samp":
    FID_PATH = f"./datasets/samp/MotionWorldIEnvCuboid/"
    FID_PATH = f"./datasets/samp/OriginalData/"
    SAVE_PATH = FID_PATH + split
    POSE_DIM = 264
    STYLE_DIM = 5
    START_TRAJ = 330
    SIT_LABEL = 3
    LIEDOWN_LABEL = 4
elif dataset == "couch":
    FID_PATH = f"./datasets/couch/MotionSequence/"
    FID_PATH = f"./datasets/couch/MotionSequence/"
    SAVE_PATH = FID_PATH + split
    POSE_DIM = 264
    STYLE_DIM = 3
    START_TRAJ = 330
    SIT_LABEL = 2
    LIEDOWN_LABEL = -1
elif dataset == "nsm":
    FID_PATH = f"./datasets/nsm/MotionWorld/"
    FID_PATH = f"./datasets/nsm/OriginalData/"
    SAVE_PATH = FID_PATH + split
    POSE_DIM = 276
    STYLE_DIM = 7
    START_TRAJ = 345
    SIT_LABEL = 5
    LIEDOWN_LABEL = -1
else:
    raise ValueError
TRAJ_DIM = (4 + STYLE_DIM) * 13
START_STYLE = START_TRAJ + 6 * (4 + STYLE_DIM) + 4
N = POSE_DIM // 12


def split_sequence(sequence, style, interaction_label, threshold=0.5, is_expand=True):
    max_val = style[:, interaction_label].max()
    interaction_inds = (style[:, interaction_label] >= threshold * max_val).nonzero()[0]
    if len(interaction_inds) == 0:
        return {}
    t1 = interaction_inds[0]
    t2 = interaction_inds[-1]
    if is_expand:
        if t1 > 30:
            t1 -= 30
        if t2 < sequence.shape[0] - 30:
            t2 += 30
    MIN_FRAMES = 120
    # MIN_FRAMES = 60
    if style.shape[-1] == 7:
        MIN_FRAMES = 60
    if style.shape[-1] == 3:
        MIN_FRAMES = 30
    if t2 > sequence.shape[0] - MIN_FRAMES:
        t2 = sequence.shape[0] - MIN_FRAMES
    if t1 >= t2:
        t2 = t1 + 1
    if interaction_label == 4:
        return {
            "Approach": sequence[:t1],
            "Liedown": sequence[t1:t2],
            "Leaving": sequence[t2:],
        }

    else:
        return {
            "Approach": sequence[:t1],
            "Sit": sequence[t1:t2],
            "Leaving": sequence[t2:],
        }


if os.path.exists(os.path.join(SAVE_PATH, "Mask.txt")):
    mask = pd_load(os.path.join(SAVE_PATH, "Mask.txt")).to_numpy()[..., 0]
else:
    mask = None
data = pd_load(os.path.join(SAVE_PATH, "Input.txt")).to_numpy()
seq_inds = pd_load(os.path.join(SAVE_PATH, "Sequences.txt")).to_numpy()[..., 0]
seq_num = int(seq_inds.max())
if kind == "armchair":
    seq_num = 18
    kind += "_"

pose = data[..., :POSE_DIM]
traj = data[..., START_TRAJ : START_TRAJ + TRAJ_DIM]
traj = traj.reshape(data.shape[:-1] + (13, (4 + STYLE_DIM)))
traj = traj[..., :4]
traj = traj.reshape(data.shape[:-1] + (52,))
window_pose_total = []
traj_total = []
mask_total = []
for i in tqdm(range(seq_num)):
    pose_seq = pose[seq_inds == (i + 1)]
    traj_seq = traj[seq_inds == (i + 1)]
    if mask is not None:
        mask_seq = mask[seq_inds == (i + 1)]
    else:
        mask_seq = None
    window_pose, traj_seq, mask_seq = matrix.get_window_pose(
        pose_seq, traj_seq, N, mask_seq
    )
    window_pose_total.append(window_pose)
    traj_total.append(traj_seq)
    mask_total.append(mask_seq)
window_pose = np.concatenate(window_pose_total, axis=0)
traj = np.concatenate(traj_total, axis=0)
if mask is not None:
    mask_total = np.concatenate(mask_total, axis=0)
x_dict = {"pose": window_pose, "traj": traj}
new_x = np.concatenate((window_pose, traj), axis=-1)
mu = np.mean(new_x, axis=0)
sigma = np.cov(new_x, rowvar=False)
np.savetxt(os.path.join(SAVE_PATH, f"FID_{kind}mu.txt"), mu)
np.savetxt(os.path.join(SAVE_PATH, f"FID_{kind}sigma.txt"), sigma)
print("Finish save all data FID")

style = data[..., START_STYLE : START_STYLE + STYLE_DIM]
split_xs = DictwithEmpty([])
if mask is not None:
    seq_inds = seq_inds[: new_x.shape[0]]
    style = style[: new_x.shape[0]]
else:
    seq_inds = seq_inds[mask_total]
    style = style[mask_total]
    seq_inds = seq_inds[mask_total]
for i in range(int(seq_inds.max())):
    x_seq = new_x[seq_inds == (i + 1)]
    style_seq = style[seq_inds == (i + 1)]
    interaction_label = SIT_LABEL
    if LIEDOWN_LABEL >= 0:
        if style_seq[:, LIEDOWN_LABEL].sum() > 0:
            interaction_label = LIEDOWN_LABEL
    seq_x_dict = split_sequence(x_seq, style_seq, interaction_label)
    for k, v in seq_x_dict.items():
        split_xs[k].append(v)
for k, v in split_xs.items():
    x = np.concatenate(v, axis=0)
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    np.savetxt(os.path.join(SAVE_PATH, f"FID_{kind}{k}_mu.txt"), mu)
    np.savetxt(os.path.join(SAVE_PATH, f"FID_{kind}{k}_sigma.txt"), sigma)
print("Finish save all splits FID")

if split == "test":
    accu_c = 0
    print("FID\t\tValue")
    for k, v in x_dict.items():
        fid = calculate_fid(v, FID_PATH + "train", accu_c, accu_c + v.shape[-1])
        accu_c += v.shape[-1]
        print(f"{k}\t\t{fid:.2f}")
    fid = calculate_fid(new_x, FID_PATH + "train")
    print(f"All\t\t{fid:.2f}\n")
    print("Split\t\tPose\tTraj\tAll")
    for k, v in split_xs.items():
        x = np.concatenate(v, axis=0)
        M = 3 * 13 + 6 * 13 * N
        fid_pose = calculate_fid(x[..., :M], FID_PATH + "train", 0, M, gt_key=k)
        fid_traj = calculate_fid(x[..., M:], FID_PATH + "train", M, M + 52, gt_key=k)
        fid_all = calculate_fid(x, FID_PATH + "train", gt_key=k)
        if len(k) < 8:
            print(f"{k}\t\t{fid_pose:.2f}\t{fid_traj:.2f}\t{fid_all:.2f}")
        else:
            print(f"{k}\t{fid_pose:.2f}\t{fid_traj:.2f}\t{fid_all:.2f}")
