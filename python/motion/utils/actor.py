import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from motion.utils.basics import DictwithEmpty

NAMES = ["Idle", "Walk", "Run", "Sit", "Liedown"]


def extract_style(sequence_input, start_inv_traj, start_traj=0, THRESHOLD=1.0):
    """_summary_

    Args:
        sequence_input (tensor): [T, C]
        start_inv_traj (int): _description_

    Returns:
        stylestr_sequence (List[str]): length = T
    """
    style_sequence = sequence_input[:, start_traj:start_inv_traj].reshape(-1, 13, 9)[
        :, 6, 4:
    ]
    stylestr_sequence = [style2str(style, THRESHOLD) for style in style_sequence]
    return stylestr_sequence


def style2str(style, THRESHOLD=1.0):
    """
    Args:
        style (tensor): [5]
    Returns:
        name (str): the name of the style
    """
    for i in range(style.shape[0]):
        if style[i] >= THRESHOLD:
            return NAMES[i]
    return "Transition"


def split_data_per_sequence(
    sequence_data, max_len, start_inv_traj, start_traj=0, THRESHOLD=1.0
):
    """_summary_

    Args:
        sequence_data (List(np.array)): List[[T, C]]
        max_len (int):
        start_inv_traj (int):

    Returns:
        split_data_total (List(List(np.array))): List[List[(T, C)]]
        lengths (List(int))
    """
    N = len(sequence_data)
    sequence_input = sequence_data[0]
    style_sequence = extract_style(
        sequence_input, start_inv_traj, start_traj, THRESHOLD
    )
    split_data_total = [[] for _ in range(N)]
    split_data = [[] for _ in range(N)]
    style_previous = None
    static_pose_count = 0
    for i, style_name in enumerate(style_sequence):
        if style_name != style_previous:
            # If not transition, we need this frame as end
            # Sometimes, the first frame is transition, no idle here
            if style_name != "Transition" or i == 0:
                for j in range(N):
                    split_data[j].append(sequence_data[j][i])
            static_pose_count += 1
            if static_pose_count == 2:
                # Aggregate 2 static pose, collect as a sequence
                for j in range(N):
                    split_data_total[j].append(np.stack(split_data[j]))
                if style_name == "Transition":
                    # If transition, we need former frame as start
                    for j in range(N):
                        split_data[j] = [sequence_data[j][i - 1], sequence_data[j][i]]
                else:
                    # If not, we use this semantic frame as start
                    for j in range(N):
                        split_data[j] = [sequence_data[j][i]]
                static_pose_count = 1
        elif len(split_data[0]) >= max_len - 1:
            # We assume transition part cannot have more than max_len frames
            for j in range(N):
                split_data[j].append(sequence_data[j][i])
                split_data_total[j].append(np.stack(split_data[j]))
                split_data[j] = [sequence_data[j][i]]
        else:
            for j in range(N):
                split_data[j].append(sequence_data[j][i])
        style_previous = style_name
    if len(split_data) != 0:
        for j in range(N):
            split_data_total[j].append(np.stack(split_data[j]))
    lengths = [len(d) for d in split_data_total[0]]
    assert sum(lengths) - len(lengths) + 1 == len(sequence_data[0])
    return split_data_total, lengths


def extract_keypose(seq_datas, interaction_style, min_static_frames):
    """_summary_

    Args:
        seq_datas (dict(List[array])): [T, C]
        interaction_style (List(str)):
        min_static_frames (int):

    Returns:
        keypose_datas: (dict(List[array])): [C]
    """
    keypose_datas = {k: [] for k in seq_datas.keys()}
    prev_seq_id = None
    for i in range(len(seq_datas["style"])):
        seq_id = seq_datas["Sequences"][i][0]
        style = seq_datas["style"][i]
        if seq_id != prev_seq_id:
            for k, v in seq_datas.items():
                keypose_datas[k].append(v[i][0])
        if (
            style[0] in interaction_style
            and style[-1] in interaction_style
            and len(style) > min_static_frames + 1
        ):
            for k, v in seq_datas.items():
                keypose_datas[k].append(v[i][min_static_frames])

        for k, v in seq_datas.items():
            keypose_datas[k].append(v[i][-1])
        prev_seq_id = seq_id
    return keypose_datas


def merge_sequence2data(sequences_data, merge_keys, extra_value_func=None):
    """_summary_

    Args:
        sequences_data (List([dict, dict])): each dict contains tensors
            "xxx": (tensor)
        merge_keys (List[str]): the key that needs to process in dicts
        extra_value_func (lambda func, optional):
            we need some values should be obtained from data. Defaults to None.

    Raises:
        NotImplementedError: some keys are not in the dicts and do not have corresponding func

    Returns:
        data_total (dict[tensor]):
    """
    data_total = DictwithEmpty([])
    data_seq = DictwithEmpty([])

    prev_seq_id = None
    for pred, gt in sequences_data:
        seq_id = gt["sequences"][0][0]
        data_seq_0 = data_seq[merge_keys[0]]
        if seq_id != prev_seq_id and len(data_seq_0) > 0:
            for i in range(len(data_seq_0)):
                if i < len(data_seq_0) - 1:
                    # remove the duplicate frames
                    for k in merge_keys:
                        data_seq[k][i] = data_seq[k][i][:-1]
            for k in merge_keys:
                data_total[k].append(torch.cat(data_seq[k], dim=0))
                data_seq[k] = []
        # Add sequence data
        for k in merge_keys:
            if k in pred.keys():
                data_seq[k].append(pred[k][0])
            elif k in gt.keys():
                data_seq[k].append(gt[k][0])
            elif extra_value_func is not None:
                data_seq[k].append(extra_value_func(pred, gt))
            else:
                raise NotImplementedError

        prev_seq_id = seq_id

    data_seq_0 = data_seq[merge_keys[0]]
    if len(data_seq_0) > 0:
        for i in range(len(data_seq_0)):
            if i < len(data_seq_0) - 1:
                # remove the duplicate frames
                for k in merge_keys:
                    data_seq[k][i] = data_seq[k][i][:-1]
        for k in merge_keys:
            data_total[k].append(torch.cat(data_seq[k], dim=0))
    return data_total
