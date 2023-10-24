import numpy as np
import os.path as osp


def load_norm_data(data_dir, split="train"):
    # Always use norm of training data
    input_norm_data = np.float32(np.loadtxt(osp.join(data_dir, split, "InputNorm.txt")))
    input_mean = input_norm_data[0]
    input_std = input_norm_data[1]
    for i in range(input_std.size):
        if input_std[i] == 0:
            input_std[i] = 1

    if not osp.exists(osp.join(data_dir, split, "OutputNorm.txt")):
        return input_mean, input_std, None, None

    output_norm_data = np.float32(
        np.loadtxt(osp.join(data_dir, split, "OutputNorm.txt"))
    )
    output_mean = output_norm_data[0]
    output_std = output_norm_data[1]
    for i in range(output_std.size):
        if output_std[i] == 0:
            output_std[i] = 1
    return input_mean, input_std, output_mean, output_std


def load_norm_data_prefix(data_dir, split="train", prefix="Input"):
    # Always use norm of training data
    if not osp.exists(osp.join(data_dir, split, f"{prefix}Norm.txt")):
        print(osp.join(data_dir, split, f"{prefix}Norm.txt") + " does not exists!")
        return 0, 1

    input_norm_data = np.float32(
        np.loadtxt(osp.join(data_dir, split, f"{prefix}Norm.txt"))
    )
    input_mean = input_norm_data[0]
    input_std = input_norm_data[1]
    for i in range(input_std.size):
        if input_std[i] == 0:
            input_std[i] = 1

    return input_mean, input_std


def load_minmax_data_prefix(data_dir, split="train", prefix="Input"):
    # Always use norm of training data
    if not osp.exists(osp.join(data_dir, split, f"{prefix}MinMax.txt")):
        print(osp.join(data_dir, split, f"{prefix}MinMax.txt") + " does not exists!")
        return 0, 1

    input_minmax_data = np.float32(
        np.loadtxt(osp.join(data_dir, split, f"{prefix}MinMax.txt"))
    )
    input_max = input_minmax_data[0]
    input_min = input_minmax_data[1]

    return input_max, input_min


TRAJMilestone_PRIOR = {0: 2, 1: 4, 2: 5, 3: 6, 4: 7}


def get_priort(distance):
    if distance < 1:
        return TRAJMilestone_PRIOR[0]
    if distance < 2:
        return TRAJMilestone_PRIOR[1]
    if distance < 3:
        return TRAJMilestone_PRIOR[2]
    if distance < 4:
        return TRAJMilestone_PRIOR[3]
    return TRAJMilestone_PRIOR[4]


def get_pred_t(flag, pred_time_range, division):
    """_summary_

    Args:
        flag (torch.tensor): indicates whether interaction, mask
        pred_time_range (int): _description_
        division (int): _description_

    Returns:
        _type_: _description_
    """
    N = len(flag)
    if flag.sum() > 0:
        inds = flag.nonzero()[0]
        ind_first = inds[0]
        ind_last = inds[-1]
        length = flag.sum()
        if length != ind_last - ind_first + 1:
            last_i = 0
            t_inter = []
            mask = []
            for i in range(len(inds[:-1])):
                if inds[i] + 1 != inds[i + 1]:
                    t_inter_, mask_ = get_pred_t(
                        flag[last_i : (inds[i] + inds[i + 1]) // 2 + 1],
                        pred_time_range,
                        division,
                    )
                    last_i = (inds[i] + inds[i + 1]) // 2 + 1
                    t_inter += t_inter_
                    mask += mask_
            t_inter_, mask_ = get_pred_t(flag[last_i:], pred_time_range, division)
            t_inter += t_inter_
            mask += mask_
            ind_first = -1
            ind_last = N
            moving_length = 0
        else:
            moving_length = length // division
            if moving_length == 0:
                ind_last = ind_first + 1
            t_inter = [-1 for _ in range(length - moving_length - 2)]
            mask = [True for _ in range(len(flag))]
    else:
        ind_first = -1
        ind_last = N - pred_time_range
        if ind_last < 0:
            ind_last = -1
            moving_length = -1
            t_inter = []
        else:
            moving_length = 0
            # here t_inter is for all frames that are in the middle
            t_inter = [pred_time_range for _ in range(N - pred_time_range)]
        mask = [False for _ in range(N)]
    t_idle2inter = [i + 1 + moving_length for i in range(ind_first + 1)][::-1]
    t_inter2idle = [
        N - ind_last + moving_length - i for i in range(N - ind_last + moving_length)
    ]
    t = t_idle2inter + t_inter + t_inter2idle
    if len(t) != N:
        import ipdb

        ipdb.set_trace()
    assert len(t) == N, "pred t frames is not equal to data"
    return t, mask
