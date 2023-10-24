import numpy as np
import torch

from scipy.interpolate import splprep, splev
import scipy
import scipy.signal  # ubuntu needs this
import scipy.ndimage  # ubuntu needs this
from .utils import normalize, denormalize, to_cpu_numpy, to_tensor
from motion.utils import matrix
from motion.utils.quaternions import Quaternions


def split_features(
    data,
    start_trajectory=0,
    start_traj_inv=117,
    start_goal=169,
    start_interaction=312,
):
    traj = data[:, start_trajectory:start_traj_inv]

    traj_inv = data[:, start_traj_inv:start_goal]
    goal = data[:, start_goal:start_interaction]
    if data.shape[1] > start_interaction:
        interaction = data[:, start_interaction:]
    else:
        interaction = None
    return traj, traj_inv, goal, interaction


def normalize_vector(v):
    # norm_v = v.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm_v


def transform_mat(mat, root_mat_out_inv):
    # Convert to transforms wrt frame i+1
    mat_transformed = torch.matmul(root_mat_out_inv, mat)
    return mat_transformed


###############################################################################################################
#################### TRAJECTRORY
###############################################################################################################


def traj_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    traj = torch.zeros(
        (bs, window_size, 4 + num_actions), device=mat.device, dtype=mat.dtype
    )
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    traj[:, :, 4:] = style
    return traj.reshape(bs, window_size * (4 + num_actions))


def traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]
    style = data.reshape(bs, window_size, -1)[:, :, 4:]

    # Convert to 3D data
    pos = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, style


def transform_traj(traj_vec, root_mat_out_inv, num_actions=5):
    traj_mat, traj_style = traj_vec_2_mat(traj_vec)
    traj_transformed = transform_mat(traj_mat, root_mat_out_inv)
    traj_transformed = traj_mat_2_vec(
        traj_transformed, traj_style, num_actions=num_actions
    )
    return traj_transformed


###############################################################################################################
#################### GOAL Related
###############################################################################################################
def goal_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir, style = data.split([78, 65], dim=-1)
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :6]

    pos = pos_dir[:, :, :3]
    forward = pos_dir[:, :, 3:]
    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat, style


def goal_mat_2_vec(mat, style, window_size=13, num_actions=5):
    bs = mat.shape[0]
    goal = torch.zeros((bs, window_size, 6), dtype=torch.float32, device=mat.device)
    goal[:, :, :3] = mat[..., :3, 3]
    goal[:, :, 3:6] = mat[..., :3, 2]
    goal = goal.reshape(bs, window_size * 6)
    goal = torch.cat((goal, style), dim=-1)
    return goal


def transform_goal(goal_vec, root_mat_out_inv, num_actions=5):
    goal_mat, goal_style = goal_vec_2_mat(goal_vec)
    goal_transformed = transform_mat(goal_mat, root_mat_out_inv)
    goal_transformed = goal_mat_2_vec(
        goal_transformed, goal_style, num_actions=num_actions
    )
    return goal_transformed


###############################################################################################################
#################### Inv Traj
###############################################################################################################
def inv_traj_mat_2_vec(mat, window_size=13):
    bs = mat.shape[0]
    traj = torch.zeros((bs, window_size, 4), device=mat.device, dtype=mat.dtype)
    traj[:, :, 0] = mat[..., 0, 3]
    traj[:, :, 1] = mat[..., 2, 3]
    traj[:, :, 2] = mat[..., 0, 2]
    traj[:, :, 3] = mat[..., 2, 2]
    return traj.reshape(bs, window_size * 4)


def inv_traj_vec_2_mat(data, window_size=13):
    # Get position and direction data only, ignore style data
    bs = data.shape[0]
    lastrow = torch.tensor(
        np.tile(np.array([[[0, 0, 0, 1]]]), (bs, window_size, 1, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    pos_dir = data.reshape(bs, window_size, -1)[:, :, :4]

    # Convert to 3D data
    pos = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    pos[:, :, 0] = pos_dir[:, :, 0]
    pos[:, :, 2] = pos_dir[:, :, 1]

    forward = torch.zeros(
        (pos_dir.shape[0], pos_dir.shape[1], 3), device=data.device, dtype=torch.float32
    )
    forward[:, :, 0] = pos_dir[:, :, 2]
    forward[:, :, 2] = pos_dir[:, :, 3]

    forward = normalize_vector(forward)
    up = torch.tensor(
        np.tile(np.array([[0, 1, 0]]), (bs, window_size, 1)),
        device=data.device,
        dtype=torch.float32,
    )
    up = normalize_vector(up)
    right = torch.cross(forward, up)
    right = normalize_vector(right)

    # check normalization above
    mat = torch.stack([right, up, forward, pos], dim=-1)
    mat = torch.cat([mat, lastrow], dim=2)
    return mat


def get_pivot_goal_world(goal_vec):
    goal_mat, _ = goal_vec_2_mat(goal_vec)
    return goal_mat[:, 6, :, :]


def transform_traj_inv(traj_inv_out, goal_in, goal_out):
    FoR_in = get_pivot_goal_world(goal_in)
    FoR_out_inv = torch.inverse(get_pivot_goal_world(goal_out)[:, None])

    traj_inv_out_mat = inv_traj_vec_2_mat(traj_inv_out)

    traj_inv_out_mat = torch.matmul(FoR_in[:, None], traj_inv_out_mat)
    traj_inv_transformed = transform_mat(traj_inv_out_mat, FoR_out_inv)
    traj_inv_transformed = inv_traj_mat_2_vec(traj_inv_transformed)

    return traj_inv_transformed


###############################################################################################################
####################  Main
###############################################################################################################


def transform_data(
    inputs, outputs, prev_root_transform=None, state_dim=524, num_actions=5, **kwargs
):
    (
        traj_in,
        traj_inv_in,
        goal_in,
        interaction_in,
    ) = split_features(inputs)
    (
        traj_out,
        traj_inv_out,
        goal_out,
        interaction_out,
    ) = split_features(outputs)
    traj_mat, _ = traj_vec_2_mat(traj_out)
    root_mat_out = traj_mat[:, 6:7]
    root_mat_out_inv = torch.inverse(root_mat_out)
    # Transform egocentric features
    traj_out_trasformed = transform_traj(
        traj_out, root_mat_out_inv, num_actions=num_actions
    )
    goal_out_transformed = transform_goal(
        goal_out, root_mat_out_inv, num_actions=num_actions
    )
    # inv
    traj_inv_out_transformed = transform_traj_inv(traj_inv_out, goal_in, goal_out)

    outputs_transformed = torch.cat(
        [
            traj_out_trasformed,
            traj_inv_out_transformed,
            goal_out_transformed,
        ],
        dim=-1,
    )
    if inputs.shape[0] > 1:
        error = ((outputs_transformed[:-1] - inputs[1:, :state_dim]) ** 2).max()
    else:
        error = None

    return outputs_transformed, error


def transform_traj_output(
    p_prev,
    I,
    p_hat,
    input_mean,
    input_std,
    output_mean,
    output_std,
    state_dim=None,
    interaction_dim=None,
    num_actions=None,
    offset_dim=None,
    **kwargs
):
    if I is not None:
        inputs = denormalize(torch.cat((p_prev, I), dim=-1), input_mean, input_std)
    else:
        inputs = denormalize(p_prev, input_mean, input_std)
    outputs = denormalize(p_hat.reshape(-1, state_dim), output_mean, output_std)
    outputs_transformed, err = transform_data(
        inputs, outputs, state_dim=state_dim, num_actions=num_actions
    )
    return normalize(
        outputs_transformed, input_mean[..., :state_dim], input_std[..., :state_dim]
    )


###############################################################################################################
#################### TRAJECTRORY SEQUENCE
###############################################################################################################


def relative_trajvec2worldmat(traj_vec, origin_mat=None):
    """_summary_

    Args:
        traj_vec (tensor): [L, 4], relative postions and relative directions

    Returns:
        traj_world_mat (tensor): [L + 1, 4, 4]
    """
    if origin_mat is None:
        origin_mat = matrix.identity_mat(traj_vec, device=traj_vec.device)

    traj_pos = matrix.xz2xyz(traj_vec[..., :2])
    traj_dir = matrix.xz2xyz(traj_vec[..., 2:4])
    traj_dir = matrix.normalized(traj_dir)

    traj_world_mat = [origin_mat[..., None, :, :]]  # [B, 1, 4, 4]
    pos_world = matrix.get_position_from(traj_pos, origin_mat)
    dir_world = matrix.get_direction_from(traj_dir, origin_mat)
    rot_world = matrix.get_rot_mat_from_forward(dir_world)
    relative_last_mat = matrix.get_TRS(rot_world, pos_world)
    last_mat = origin_mat[..., None, :, :]  # [B, 1, 4, 4]
    for i in range(traj_vec.shape[-2]):
        last_mat = matrix.get_mat_BfromA(
            last_mat, relative_last_mat[..., i : i + 1, :, :]
        )
        traj_world_mat.append(last_mat)
    traj_world_mat = torch.cat(traj_world_mat, dim=-3)
    return traj_world_mat


def absolute_trajvec2worldmat(traj_vec, origin_mat=None):
    """_summary_

    Args:
        traj_vec (tensor): [L, 4], absolute postions and relative directions

    Returns:
        traj_world_mat (tensor): [L + 1, 4, 4]
    """
    if origin_mat is None:
        origin_mat = matrix.identity_mat(traj_vec, device=traj_vec.device)

    traj_pos = matrix.xz2xyz(traj_vec[..., :2])
    traj_dir = matrix.xz2xyz(traj_vec[..., 2:4])
    traj_dir = matrix.normalized(traj_dir)

    pos_world = matrix.get_position_from(traj_pos, origin_mat)
    dir_world = matrix.get_direction_from(traj_dir, origin_mat)
    rot_world = matrix.get_rot_mat_from_forward(dir_world)
    world_mat = matrix.get_TRS(rot_world, pos_world)
    traj_world_mat = torch.cat((origin_mat[..., None, :, :], world_mat), dim=-3)
    return traj_world_mat


def abstrajvec2points(traj_vec, root_mat=None):
    """_summary_

    Args:
        traj_vec (_type_): [1, T, C]

    Returns:
        _type_: _description_
    """
    traj_world_mat = absolute_trajvec2worldmat(to_tensor(traj_vec))
    traj_world_mat = torch.cat(
        (traj_world_mat[..., :1, :, :], traj_world_mat[..., 2:, :, :]), dim=-3
    )
    if root_mat is not None:
        traj_world_mat = matrix.get_mat_BfromA(root_mat, traj_world_mat)
    traj_world_mat = to_cpu_numpy(traj_world_mat)

    x = traj_world_mat[..., 0, 3].reshape(-1)
    y = traj_world_mat[..., 2, 3].reshape(-1)
    dirx = traj_world_mat[..., 0, 2].reshape(-1)
    diry = traj_world_mat[..., 2, 2].reshape(-1)
    return x, y, dirx, diry


def trajvec2points(traj_vec, root_mat=None):
    """_summary_

    Args:
        traj_vec (_type_): [1, T, C]

    Returns:
        _type_: _description_
    """
    traj_world_mat = relative_trajvec2worldmat(to_tensor(traj_vec))
    if root_mat is not None:
        traj_world_mat = matrix.get_mat_BfromA(root_mat, traj_world_mat)
    traj_world_mat = traj_world_mat[..., :-1, :, :]
    traj_world_mat = to_cpu_numpy(traj_world_mat)

    x = traj_world_mat[..., 0, 3].reshape(-1)
    y = traj_world_mat[..., 2, 3].reshape(-1)
    dirx = traj_world_mat[..., 0, 2].reshape(-1)
    diry = traj_world_mat[..., 2, 2].reshape(-1)
    return x, y, dirx, diry


def invtrajvec2points(traj_vec, goal_mat, root_mat, start_mat=None):
    """_summary_

    Args:
        traj_vec (_type_): [1, T, C]
        goal_mat (_type_): [4, 4]
        root_mat (_type_): [4, 4]
        start_mat (_type_): [4, 4] root in the coordinate of the whole start point

    Returns:
        _type_: _description_
    """
    invtraj_goal_mat = absolute_trajvec2worldmat(to_tensor(traj_vec))
    invtraj_goal_mat = torch.cat(
        (invtraj_goal_mat[..., 1:-1, :, :], invtraj_goal_mat[..., :1, :, :]), dim=-3
    )
    invtraj_world_mat = matrix.get_mat_BfromA(goal_mat, invtraj_goal_mat)
    invtraj_world_mat = matrix.get_mat_BtoA(root_mat, invtraj_world_mat)
    if start_mat is not None:
        invtraj_world_mat = matrix.get_mat_BfromA(start_mat, invtraj_world_mat)
    traj_world_mat = to_cpu_numpy(invtraj_world_mat)

    x = traj_world_mat[..., 0, 3].reshape(-1)
    y = traj_world_mat[..., 2, 3].reshape(-1)
    dirx = traj_world_mat[..., 0, 2].reshape(-1)
    diry = traj_world_mat[..., 2, 2].reshape(-1)
    return x, y, dirx, diry


def merge_abs_inv_traj(traj, invtraj, merge_weight=None):
    """_summary_

    Args:
        traj (_type_): [N]  abs traj
        invtraj (_type_): [N]

    Returns:
        _type_: _description_
    """
    traj_x, traj_y, traj_dirx, traj_diry = traj
    invtraj_x, invtraj_y, invtraj_dirx, invtraj_diry = invtraj
    T = traj_x.shape[0]
    if merge_weight is None:
        merge_weight = np.linspace(0, 1, T)
    merge_x = traj_x * (1 - merge_weight) + invtraj_x * merge_weight
    merge_y = traj_y * (1 - merge_weight) + invtraj_y * merge_weight

    traj_forward = np.stack(
        (traj_dirx, np.zeros_like(traj_dirx), traj_diry)
    ).transpose()  # [N, 3]
    traj_rot_mat = matrix.get_rot_mat_from_forward(to_tensor(traj_forward))
    traj_rot = Quaternions.from_transforms(to_cpu_numpy(traj_rot_mat))

    invtraj_forward = np.stack(
        (invtraj_dirx, np.zeros_like(invtraj_dirx), invtraj_diry)
    ).transpose()  # [N, 3]
    invtraj_rot_mat = matrix.get_rot_mat_from_forward(to_tensor(invtraj_forward))
    invtraj_rot = Quaternions.from_transforms(to_cpu_numpy(invtraj_rot_mat))

    merge_rot = Quaternions.slerp(traj_rot, invtraj_rot, merge_weight)
    merge_rot_mat = merge_rot.transforms()
    merge_dirx = merge_rot_mat[:, 0, 2]
    merge_diry = merge_rot_mat[:, 2, 2]

    # delta_dirx = traj_dirx - invtraj_dirx
    # delta_diry = traj_diry - invtraj_diry
    # merge_dirx = invtraj_dirx + delta_dirx * (1 - merge_weight)
    # merge_diry = invtraj_diry + delta_diry * (1 - merge_weight)
    # merge_dirx = merge_dirx / (merge_dirx ** 2 + merge_diry ** 2) ** 0.5
    # merge_diry = merge_diry / (merge_dirx ** 2 + merge_diry ** 2) ** 0.5
    return merge_x, merge_y, merge_dirx, merge_diry


def interpolate_trajpoint(p1, p2, w1=0.5, w2=0.5):
    # w1 control postions, w2 controls rotations. the value is for p2, the higher, the more p2
    xy_new = p1[..., :2] * (1 - w1) + p2[..., :2] * w1
    dir1 = np.stack(
        (p1[..., 2:3], np.zeros_like(p1[..., 2:3]), p1[..., 3:])
    ).transpose()
    dir2 = np.stack(
        (p2[..., 2:3], np.zeros_like(p2[..., 2:3]), p2[..., 3:])
    ).transpose()
    if len(p1.shape) > 1:
        dir1 = dir1.squeeze()
        dir2 = dir2.squeeze()
    rot1 = matrix.get_rot_mat_from_forward(dir1)
    rot2 = matrix.get_rot_mat_from_forward(dir2)
    rot1 = Quaternions.from_transforms(to_cpu_numpy(rot1))
    rot2 = Quaternions.from_transforms(to_cpu_numpy(rot2))
    rot_new = Quaternions.slerp(rot1, rot2, w2)
    rot_new = rot_new.transforms()
    dir_newx = rot_new[..., 0, 2]
    dir_newy = rot_new[..., 2, 2]
    if len(p1.shape) > 1:
        dir_newx = dir_newx[..., None]
        dir_newy = dir_newy[..., None]
    dir_new = np.concatenate((dir_newx, dir_newy), axis=-1)
    return np.concatenate((xy_new, dir_new), axis=-1)


def interpolate_trajpoints_L(p1, p2, L=61):
    p1e = np.tile(p1, (L, 1))
    p2e = np.tile(p2, (L, 1))
    w = np.linspace(0, 1, L)
    points = interpolate_trajpoint(p1e, p2e, w[..., None], w)
    # points = [p1]
    # w = 1.0 / (L - 2)
    # for i in range(1, L - 1):
    #     temp_p = interpolate_trajpoint(p1, p2, w1=w * i, w2=w * i)
    #     points.append(temp_p)
    # points.append(p2)
    return np.stack(points, axis=0)


def filter_traj(traj, merge_weight=None):
    """_summary_

    Args:
        traj (_type_): [N, 4]  abs traj

    Returns:
        _type_: _description_
    """
    traj_x, traj_y, traj_dirx, traj_diry = (
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        traj[:, 3],
    )

    # tck, u = splprep([traj_x, traj_y])
    # new_points = splev(u, tck)

    traj_x_ = scipy.signal.savgol_filter(traj_x, 15, 5)
    traj_y_ = scipy.signal.savgol_filter(traj_y, 15, 5)
    # import ipdb;ipdb.set_trace()
    traj_x = traj_x_
    traj_y = traj_y_

    # pooling_x = (traj_x[:-2] + traj_x[2:]) / 2.
    # traj_x[1 : -1] = (pooling_x + traj_x[1 : -1]) / 2.
    # # traj_x[1 : -1] = pooling_x
    # pooling_y = (traj_y[:-2] + traj_y[2:]) / 2.
    # traj_y[1 : -1] = (pooling_y + traj_y[ 1: -1]) / 2.

    # traj_y[1 : -1] = pooling_y

    # traj_forward = np.stack(
    #     (traj_dirx, np.zeros_like(traj_dirx), traj_diry)
    # ).transpose()  # [N, 3]
    # traj_rot_mat = matrix.get_rot_mat_from_forward(to_tensor(traj_forward))
    # traj_rot = Quaternions.from_transforms(to_cpu_numpy(traj_rot_mat))

    # merge_rot = Quaternions.slerp(traj_rot[:-2], traj_rot[2:], 0.5)
    # merge_rot = Quaternions.slerp(merge_rot, traj_rot[1:-1], 0.5)
    # merge_rot_mat = merge_rot.transforms()
    # pooling_dirx = merge_rot_mat[:, 0, 2]
    # pooling_diry = merge_rot_mat[:, 2, 2]
    # traj_dirx[1:-1] = pooling_dirx
    # traj_diry[1:-1] = pooling_diry
    pool_traj = np.stack((traj_x, traj_y, traj_dirx, traj_diry), axis=-1)

    return pool_traj


def gaussian_traj(traj, merge_weight=None):
    """_summary_

    Args:
        traj (_type_): [N, 4]  abs traj

    Returns:
        _type_: _description_
    """
    traj_x, traj_y, traj_dirx, traj_diry = (
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        traj[:, 3],
    )

    ksize = 15
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    # traj_x[1 : -1] = scipy.ndimage.gaussian_filter1d(traj_x, sigma, radius = ksize // 2)
    traj_x[1:-1] = scipy.ndimage.gaussian_filter1d(traj_x, sigma)[1:-1]

    pooling_x = (traj_x[:-2] + traj_x[2:]) / 2.0
    # traj_x[1 : -1] = (pooling_x + traj_x[1 : -1]) / 2.
    # traj_x[1 : -1] = pooling_x

    pooling_y = (traj_y[:-2] + traj_y[2:]) / 2.0
    # traj_y[1 : -1] = (pooling_y + traj_y[ 1: -1]) / 2.
    traj_y[1:-1] = scipy.ndimage.gaussian_filter1d(traj_y, sigma)[1:-1]
    # traj_y[1 : -1] = pooling_y
    traj_dirx[1:-1] = scipy.ndimage.gaussian_filter1d(traj_dirx, sigma)[1:-1]
    traj_diry[1:-1] = scipy.ndimage.gaussian_filter1d(traj_diry, sigma)[1:-1]

    pool_traj = np.stack((traj_x, traj_y, traj_dirx, traj_diry), axis=-1)

    return pool_traj


def gaussian_motion_samp(motion):
    """_summary_

    Args:
        traj (_type_): [N, 4]  abs traj

    Returns:
        _type_: _description_
    """
    ksize = 15
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    flag = False
    if isinstance(motion, torch.Tensor):
        motion = to_cpu_numpy(motion)
        flag = True
    motion[1:-1] = scipy.ndimage.gaussian_filter1d(motion, sigma, axis=0)[1:-1]
    if flag:
        motion = to_tensor(motion)
    return motion


def gaussian_motion(motion, ksize=15):
    """_summary_

    Args:
        traj (_type_): [N, 4]  abs traj

    Returns:
        _type_: _description_
    """
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    flag = False
    if isinstance(motion, torch.Tensor):
        motion = to_cpu_numpy(motion)
        flag = True
    motion = scipy.ndimage.gaussian_filter1d(motion, sigma, axis=0)
    if flag:
        motion = to_tensor(motion)
    return motion


def filter_motion(motion, ksize=15, merge_weight=None):
    """_summary_

    Args:
        traj (_type_): [N, 4]  abs traj

    Returns:
        _type_: _description_
    """
    flag = False
    if isinstance(motion, torch.Tensor):
        motion = to_cpu_numpy(motion)
        flag = True
    pool_motion = np.zeros_like(motion)
    polyorder = ksize // 3
    if polyorder < 2:
        polyorder = 2
    for i in range(motion.shape[1]):
        pool_motion[:, i] = scipy.signal.savgol_filter(motion[:, i], ksize, polyorder)
    if flag:
        pool_motion = to_tensor(pool_motion)
    return pool_motion
