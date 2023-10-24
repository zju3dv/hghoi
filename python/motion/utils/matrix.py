import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import numpy as np

from motion.utils.utils import to_tensor


def identity_mat(x=None, device="cpu", is_numpy=False):
    if x is not None:
        if isinstance(x, torch.Tensor):
            mat = torch.eye(4, device=device)
            mat = mat.repeat(x.shape[:-2] + (1, 1))
        elif isinstance(x, np.ndarray):
            mat = np.eye(4, dtype=np.float32)
            if x is not None:
                for _ in range(len(x.shape) - 2):
                    mat = mat[None]
            mat = np.tile(mat, x.shape[:-2] + (1, 1))
        else:
            raise ValueError
    else:
        if is_numpy:
            mat = np.eye(4, dtype=np.float32)
        else:
            mat = torch.eye(4, device=device)

    return mat


def vec2mat(vec):
    """_summary_

    Args:
        vec (tensor): [12], pos, forward, up and right

    Returns:
        mat_world(tensor): [4, 4]
    """
    # Assume bs = 1
    v = np.tile(np.array([[0, 0, 0, 1]]), (1, 1))
    if isinstance(vec, torch.Tensor):
        v = torch.tensor(
            v,
            device=vec.device,
            dtype=vec.dtype,
        )
    pos = vec[:3]
    forward = vec[3:6]
    up = vec[6:9]
    right = vec[9:12]

    if isinstance(vec, torch.Tensor):
        mat_world = torch.stack([right, up, forward, pos], dim=-1)
        mat_world = torch.cat([mat_world, v], dim=-2)
    elif isinstance(vec, np.ndarray):
        mat_world = np.stack([right, up, forward, pos], axis=-1)
        mat_world = np.concatenate([mat_world, v], axis=-2)
    else:
        raise ValueError
    mat_world = normalized_matrix(mat_world)
    return mat_world


def mat2vec(mat):
    """_summary_

    Args:
        mat(tensor): [4, 4]

    Returns:
        vec (tensor): [12], pos, forward, up and right
    """
    # Assume bs = 1
    pos = mat[:-1, 3]
    forward = normalized(mat[:-1, 2])
    up = normalized(mat[:-1, 1])
    right = normalized(mat[:-1, 0])
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((pos, forward, up, right))
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((pos, forward, up, right))
    else:
        raise ValueError

    return vec


def vec2mat_batch(vec):
    """_summary_

    Args:
        vec (tensor): [B, 12], pos, forward, up and right

    Returns:
        mat_world(tensor): [B, 4, 4]
    """
    # Assume bs = 1

    v = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (vec.shape[0], 1, 1))
    if isinstance(vec, torch.Tensor):
        v = torch.tensor(
            v,
            device=vec.device,
            dtype=vec.dtype,
        )
    pos = vec[..., :3]
    forward = vec[..., 3:6]
    up = vec[..., 6:9]
    right = vec[..., 9:12]
    if isinstance(vec, torch.Tensor):
        mat_world = torch.stack([right, up, forward, pos], dim=-1)
        mat_world = torch.cat([mat_world, v], dim=-2)
    elif isinstance(vec, np.ndarray):
        mat_world = np.stack([right, up, forward, pos], axis=-1)
        mat_world = np.concatenate([mat_world, v], axis=-2)
    else:
        raise ValueError

    mat_world = normalized_matrix(mat_world)
    return mat_world


def rotmat2tan_norm(mat):
    """_summary_

    Args:
        mat(tensor): [B, 3, 3]

    Returns:
        vec (tensor): [B, 6], tan norm
    """
    if isinstance(mat, np.ndarray):
        tan = np.zeros_like(mat[..., 2])
        norm = np.zeros_like(mat[..., 0])
    elif isinstance(mat, torch.Tensor):
        tan = torch.zeros_like(mat[..., 2])
        norm = torch.zeros_like(mat[..., 0])
    else:
        raise ValueError
    tan[...] = mat[..., 2, ::-1]
    tan[..., -1] *= -1
    norm[...] = mat[..., 0, ::-1]
    norm[..., -1] *= -1
    if isinstance(mat, np.ndarray):
        tan_norm = np.concatenate((tan, norm), axis=-1)
    elif isinstance(mat, torch.Tensor):
        tan_norm = torch.cat((tan, norm), dim=-1)
    else:
        raise ValueError
    return tan_norm


def mat2tan_norm(mat):
    """_summary_

    Args:
        mat(tensor): [B, 4, 4]

    Returns:
        vec (tensor): [B, 6], tan norm
    """
    rot_mat = mat[..., :-1, :-1]
    return rotmat2tan_norm(rot_mat)


def rotmat2tan_norm(mat):
    """_summary_

    Args:
        mat(tensor): [B, 3, 3]

    Returns:
        vec (tensor): [B, 6], tan norm
    """
    if isinstance(mat, np.ndarray):
        tan = np.zeros_like(mat[..., 2])
        norm = np.zeros_like(mat[..., 0])
        tan[...] = mat[..., 2, ::-1]
        norm[...] = mat[..., 0, ::-1]
    elif isinstance(mat, torch.Tensor):
        tan = torch.zeros_like(mat[..., 2])
        norm = torch.zeros_like(mat[..., 0])
        tan[...] = torch.flip(mat[..., 2], dims=[-1])
        norm[...] = torch.flip(mat[..., 0], dims=[-1])
    else:
        raise ValueError
    tan[..., -1] *= -1
    norm[..., -1] *= -1
    if isinstance(mat, np.ndarray):
        tan_norm = np.concatenate((tan, norm), axis=-1)
    elif isinstance(mat, torch.Tensor):
        tan_norm = torch.cat((tan, norm), dim=-1)
    else:
        raise ValueError
    return tan_norm


def tan_norm2rotmat(tan_norm):
    """_summary_

    Args:
        mat(tensor): [B, 6]

    Returns:
        vec (tensor): [B, 3]
    """
    tan = copy.deepcopy(tan_norm[..., :3])
    norm = copy.deepcopy(tan_norm[..., 3:])
    tan[..., -1] *= -1
    norm[..., -1] *= -1
    if isinstance(tan_norm, np.ndarray):
        rotmat = np.zeros(tan_norm.shape[:-1] + (3, 3))
        tan = tan[..., ::-1]
        norm = norm[..., ::-1]
        other = np.cross(tan, norm)
    elif isinstance(tan_norm, torch.Tensor):
        rotmat = torch.zeros(tan_norm.shape[:-1] + (3, 3), device=tan_norm.device)
        tan = torch.flip(tan, dims=[-1])
        norm = torch.flip(norm, dims=[-1])
        other = torch.cross(tan, norm)
    else:
        raise ValueError
    rotmat[..., 2, :] = tan
    rotmat[..., 0, :] = norm
    rotmat[..., 1, :] = other
    return rotmat


def rotmat332vec_batch(mat):
    """_summary_

    Args:
        mat(tensor): [B, 3, 3]

    Returns:
        vec (tensor): [B, 6], forward, up, right
    """
    # Assume bs = 1
    mat = normalized_matrix(mat)
    forward = mat[..., :, 2]
    up = mat[..., :, 1]
    right = mat[..., :, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def rotmat2vec_batch(mat):
    """_summary_

    Args:
        mat(tensor): [B, 4, 4]

    Returns:
        vec (tensor): [B, 9], forward, up, right
    """
    # Assume bs = 1
    mat = normalized_matrix(mat)
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    right = mat[..., :-1, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def mat2vec_batch(mat):
    """_summary_

    Args:
        mat(tensor): [B, 4, 4]

    Returns:
        vec (tensor): [B, 12], pos, forward, up and right
    """
    # Assume bs = 1
    mat = normalized_matrix(mat)
    pos = mat[..., :-1, 3]
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    right = mat[..., :-1, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((pos, forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((pos, forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def mat2pose_batch(mat, returnvel=True):
    """_summary_

    Args:
        mat(tensor): [B, 4, 4]

    Returns:
        vec (tensor): [B, 12], pos, forward, up, zeros
    """
    # Assume bs = 1
    mat = normalized_matrix(mat)
    pos = mat[..., :-1, 3]
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    if isinstance(mat, torch.Tensor):
        if returnvel:
            vel = torch.zeros_like(up)
            vec = torch.cat((pos, forward, up, vel), dim=-1)
        else:
            vec = torch.cat((pos, forward, up), dim=-1)
    elif isinstance(mat, np.ndarray):
        if returnvel:
            vel = np.zeros_like(up)
            vec = np.concatenate((pos, forward, up, vel), axis=-1)
        else:
            vec = np.concatenate((pos, forward, up), axis=-1)
    else:
        raise ValueError
    return vec


def get_mat_BtoA(matA, matB):
    """
        return matrix B in the coordinate of A

    Args:
        matA (tensor): [4, 4] world matrix
        matB (tensor): [4, 4] world matrix
    """
    if isinstance(matA, torch.Tensor):
        matA_inv = torch.inverse(matA)
    elif isinstance(matA, np.ndarray):
        matA_inv = np.linalg.inv(matA)
    else:
        raise ValueError
    matA_inv = normalized_matrix(matA_inv)
    if isinstance(matA, torch.Tensor):
        mat_BtoA = torch.matmul(matA_inv, matB)
    elif isinstance(matA, np.ndarray):
        mat_BtoA = np.matmul(matA_inv, matB)
    mat_BtoA = normalized_matrix(mat_BtoA)
    return mat_BtoA


def get_mat_BfromA(matA, matBtoA):
    """
        return world matrix B given matrix A and mat B realtive to A

    Args:
        matA (_type_): [4, 4] world matrix
        matBtoA (_type_): [4, 4] matrix B relative to A
    """
    if isinstance(matA, torch.Tensor):
        matB = torch.matmul(matA, matBtoA)
    if isinstance(matA, np.ndarray):
        matB = np.matmul(matA, matBtoA)
    matB = normalized_matrix(matB)
    return matB


def get_relative_position_to(pos, mat):
    """_summary_

    Args:
        pos (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        _type_: _description_
    """
    if isinstance(mat, torch.Tensor):
        mat_inv = torch.inverse(mat)
    elif isinstance(mat, np.ndarray):
        mat_inv = np.linalg.inv(mat)
    else:
        raise ValueError
    mat_inv = normalized_matrix(mat_inv)
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat_inv[..., :-1, :-1], pos.transpose(-1, -2)).transpose(
            -1, -2
        )
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat_inv[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(
            -1, -2
        )
    world_pos = rot_pos + mat_inv[..., None, :-1, 3]
    return world_pos

def get_rotation(mat):
    """_summary_

    Args:
        mat (_type_): [..., 4, 4]

    Returns:
        _type_: _description_
    """
    return mat[..., :-1, :-1]


def get_position(mat):
    """_summary_

    Args:
        mat (_type_): [..., 4, 4]

    Returns:
        _type_: _description_
    """
    return mat[..., :-1, 3]


def get_position_from(pos, mat):
    """_summary_

    Args:
        pos (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        _type_: _description_
    """
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat[..., :-1, :-1], pos.transpose(-1, -2)).transpose(
            -1, -2
        )
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    else:
        raise ValueError

    world_pos = rot_pos + mat[..., None, :-1, 3]
    return world_pos


def get_relative_direction_to(dir, mat):
    """_summary_

    Args:
        dir (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        _type_: _description_
    """
    if isinstance(mat, torch.Tensor):
        mat_inv = torch.inverse(mat)
    elif isinstance(mat, np.ndarray):
        mat_inv = np.linalg.inv(mat)
    else:
        raise ValueError
    mat_inv = normalized_matrix(mat_inv)
    rot_mat_inv = mat_inv[..., :3, :3]
    if isinstance(mat, torch.Tensor):
        rel_dir = torch.matmul(rot_mat_inv, dir.transpose(-1, -2))
        return rel_dir.transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rel_dir = np.matmul(rot_mat_inv, dir.swapaxes(-1, -2))
        return rel_dir.swapaxes(-1, -2)
    else:
        raise ValueError
    return


def get_direction_from(dir, mat):
    """_summary_

    Args:
        dir (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        tensor: [N, M, 3] or [N, 3]
    """
    rot_mat = mat[..., :3, :3]
    if isinstance(mat, torch.Tensor):
        world_dir = torch.matmul(rot_mat, dir.transpose(-1, -2))
        return world_dir.transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        world_dir = np.matmul(rot_mat, dir.swapaxes(-1, -2))
        return world_dir.swapaxes(-1, -2)
    else:
        raise ValueError
    return


def get_coord_vis(pos, rot_mat, scale=1.0):
    forward = rot_mat[..., :, 2]
    up = rot_mat[..., :, 1]
    right = rot_mat[..., :, 0]
    return pos + right * scale, pos + up * scale, pos + forward * scale


def project_vec(vec):
    """_summary_

    Args:
        vec (tensor): [*, 12], pos, forward, up and right

    Returns:
        proj_vec (tensor): [*, 4], posx, posz, forwardx, forwardz
    """
    posx = vec[..., 0:1]
    posz = vec[..., 2:3]
    forwardx = vec[..., 3:4]
    forwardz = vec[..., 5:6]
    if isinstance(vec, torch.Tensor):
        proj_vec = torch.cat((posx, posz, forwardx, forwardz), dim=-1)
    elif isinstance(vec, np.ndarray):
        proj_vec = np.concatenate((posx, posz, forwardx, forwardz), axis=-1)
    else:
        raise ValueError

    return proj_vec


def xz2xyz(vec):
    x = vec[..., 0:1]
    z = vec[..., 1:2]
    if isinstance(vec, torch.Tensor):
        y = torch.zeros(vec.shape[:-1] + (1,), device=vec.device)
        xyz_vec = torch.cat((x, y, z), dim=-1)
    elif isinstance(vec, np.ndarray):
        y = np.zeros(vec.shape[:-1] + (1,))
        xyz_vec = np.concatenate((x, y, z), axis=-1)
    else:
        raise ValueError

    return xyz_vec


def normalized(vec):
    if isinstance(vec, torch.Tensor):
        norm_vec = vec / (vec.norm(2, dim=-1, keepdim=True) + 1e-9)
    elif isinstance(vec, np.ndarray):
        norm_vec = vec / (np.linalg.norm(vec, ord=2, axis=-1, keepdims=True) + 1e-9)
    else:
        raise ValueError

    return norm_vec


def normalized_matrix(mat):
    if mat.shape[-1] == 4:
        rot_mat = mat[..., :-1, :-1]
    else:
        rot_mat = mat
    if isinstance(mat, torch.Tensor):
        rot_mat_norm = rot_mat / (rot_mat.norm(2, dim=-2, keepdim=True) + 1e-9)
        norm_mat = torch.zeros_like(mat)
    elif isinstance(mat, np.ndarray):
        rot_mat_norm = rot_mat / (
            np.linalg.norm(rot_mat, ord=2, axis=-2, keepdims=True) + 1e-9
        )
        norm_mat = np.zeros_like(mat)
    else:
        raise ValueError
    if mat.shape[-1] == 4:
        norm_mat[..., :-1, :-1] = rot_mat_norm
        norm_mat[..., :-1, -1] = mat[..., :-1, -1]
        norm_mat[..., -1, -1] = 1.0
    else:
        norm_mat = rot_mat_norm
    return norm_mat


def get_rot_mat_from_forward(forward):
    """_summary_

    Args:
        forward (tensor): [N, M, 3]

    Returns:
        mat (tensor): [N, M, 3, 3]
    """
    if isinstance(forward, torch.Tensor):
        mat = torch.eye(3, device=forward.device).repeat(forward.shape[:-1] + (1, 1))
        right = torch.zeros_like(forward)
    elif isinstance(forward, np.ndarray):
        mat = np.eye(3, dtype=np.float32)
        for _ in range(len(forward.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, forward.shape[:-1] + (1, 1))
        right = np.zeros_like(forward)
    else:
        raise ValueError

    right[..., 0] = forward[..., 2]
    right[..., 1] = 0.0
    right[..., 2] = -forward[..., 0]
    # right = torch.cross(mat[..., 1], forward)  # cannot backward

    mat[..., 2] = normalized(forward)
    right = normalized(right)
    mat[..., 0] = right
    return mat


def get_rot_mat_from_forward_up(forward, up):
    """_summary_

    Args:
        forward (tensor): [N, M, 3]
        up (tensor): [N, M, 3]

    Returns:
        mat (tensor): [N, M, 3, 3]
    """
    if isinstance(forward, torch.Tensor):
        mat = torch.eye(3, device=forward.device).repeat(forward.shape[:-1] + (1, 1))
        right = torch.cross(up, forward)
    elif isinstance(forward, np.ndarray):
        mat = np.eye(3, dtype=np.float32)
        for _ in range(len(forward.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, forward.shape[:-1] + (1, 1))
        right = np.cross(up, forward)
    else:
        raise ValueError

    right = normalized(right)
    mat[..., 2] = normalized(forward)
    mat[..., 1] = normalized(up)
    mat[..., 0] = right
    return mat


def get_rot_mat_from_pose_vec(vec):
    """_summary_

    Args:
        vec (tensor): [N, M, 6]

    Returns:
        mat (tensor): [N, M, 3, 3]
    """
    forward = vec[..., :3]
    up = vec[..., 3:6]
    return get_rot_mat_from_forward_up(forward, up)


def get_TRS(rot_mat, pos):
    """_summary_

    Args:
        rot_mat (tensor): [N, 3, 3]
        pos (tensor): [N, 3]

    Returns:
        mat (tensor): [N, 4, 4]
    """
    if isinstance(rot_mat, torch.Tensor):
        mat = torch.eye(4, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    elif isinstance(rot_mat, np.ndarray):
        mat = np.eye(4, dtype=np.float32)
        for _ in range(len(pos.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, pos.shape[:-1] + (1, 1))
    else:
        raise ValueError
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    mat = normalized_matrix(mat)
    return mat


def xzvec2mat(vec):
    """_summary_

    Args:
        vec (tensor): [N, 4]

    Returns:
        mat (tensor): [N, 4, 4]
    """
    vec_shape = vec.shape[:-1]
    if isinstance(vec, torch.Tensor):
        pos = torch.zeros(vec_shape + (3,))
        forward = torch.zeros(vec_shape + (3,))
    elif isinstance(vec, np.ndarray):
        pos = np.zeros(vec_shape + (3,))
        forward = np.zeros(vec_shape + (3,))
    else:
        raise ValueError

    pos[..., 0] = vec[..., 0]
    pos[..., 2] = vec[..., 1]
    forward[..., 0] = vec[..., 2]
    forward[..., 2] = vec[..., 3]
    rot_mat = get_rot_mat_from_forward(forward)
    mat = get_TRS(rot_mat, pos)
    return mat


def distance(vec1, vec2):
    return ((vec1 - vec2) ** 2).sum() ** 0.5


def get_relative_pose_from_vec(pose, root, N):
    root_p_mat = xzvec2mat(root)
    pose = pose.reshape(-1, N, 12)
    pose[..., :3] = get_position_from(pose[..., :3], root_p_mat)
    pose[..., 3:6] = get_direction_from(pose[..., 3:6], root_p_mat)
    pose[..., 6:9] = get_direction_from(pose[..., 6:9], root_p_mat)
    pose[..., 9:] = get_direction_from(pose[..., 9:], root_p_mat)
    pos = pose[..., 0, :3]
    rot = pose[..., 3:9].reshape(-1, N * 6)
    pose = np.concatenate((pos, rot), axis=-1)
    return pose
    return pose.reshape(-1, N * 12)


def get_window_pose(pose, root, N, mask=None):
    start_pose = np.tile(pose[:1], (30, 1))
    end_pose = np.tile(pose[-1:], (30, 1))
    expand_pose = np.concatenate((start_pose, pose, end_pose), axis=0)

    reltraj_num = 13
    pivot = 7
    step_size = 5
    relpose_total = []
    for i in range(reltraj_num):
        relind = [j + (i - pivot + 1) * step_size + 30 for j in range(root.shape[0])]
        relpose = get_relative_pose_from_vec(
            expand_pose[relind], root[..., i * 4 : (i + 1) * 4], N
        )
        relpose_total.append(relpose)
    relpose = np.concatenate(relpose_total, axis=-1)
    relroot = root
    if mask is not None:
        start_mask = np.tile(mask[:1], (30))
        end_mask = np.tile(mask[-1:], (30))
        mask = np.concatenate((start_mask, mask, end_mask), axis=0)
        calculate_mask = []
        for i in range(reltraj_num):
            relind = [
                j + (i - pivot + 1) * step_size + 30 for j in range(root.shape[0])
            ]
            calculate_mask.append(mask[relind])
        calculate_mask = np.stack(calculate_mask, axis=-1)
        calculate_mask = calculate_mask.sum(axis=-1) == calculate_mask.shape[-1]
        relpose = relpose[calculate_mask]
        relroot = relroot[calculate_mask]

    return relpose, relroot, mask


def get_window_traj(mat, reltraj_num=13, pivot=7, step_size=5):
    start_mat = np.tile(mat[:1], (30, 1, 1))
    last_mat = np.tile(mat[-1:], (30, 1, 1))
    expand_mat = np.concatenate((start_mat, mat, last_mat), axis=0)
    relxz_total = []
    for i in range(reltraj_num):
        relind = [j + (i - pivot + 1) * step_size + 30 for j in range(mat.shape[0])]
        relmat = get_mat_BtoA(mat, expand_mat[relind])
        relxz = project_vec(mat2vec_batch(relmat))
        relxz_total.append(relxz)
    relxz_total = np.concatenate((relxz_total), axis=-1)  # T, 13 * 4
    return relxz_total


def get_window_style(style, reltraj_num=13, pivot=7, step_size=5):
    start_style = np.tile(style[:1], (30, 1))
    last_style = np.tile(style[-1:], (30, 1))
    expand_style = np.concatenate((start_style, style, last_style), axis=0)
    relstyle_total = []
    for i in range(reltraj_num):
        relind = [j + (i - pivot + 1) * step_size + 30 for j in range(style.shape[0])]
        relstyle = expand_style[relind]
        relstyle_total.append(relstyle)
    relstyle_total = np.concatenate((relstyle_total), axis=-1)
    return relstyle_total


def ase_quat2our_quat(quat):
    new_quat = np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1)
    return new_quat


def our_quat2ase_quat(quat):
    new_quat = np.concatenate([quat[..., 1:4], quat[..., :1]], axis=-1)
    return new_quat
