import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import open3d as o3d
import smplx

from .utils import get_random_color, to_cpu_numpy
from .func_wrapper import data_filter, recursive_do_wrap, recursive_everydo_wrap


def plot_code_histogram(output_codes):
    """
    Args:
        output_codes (list): (tensor, int) [1, P]
    """
    codes = torch.cat(output_codes, dim=0)
    if codes.shape[1] < 7:
        show_histogram_subplot([codes[:, i] for i in range(codes.shape[1])])
    else:
        show_histogram_individual([codes[:, i] for i in range(codes.shape[1])])


@data_filter
def show_histogram_subplot(data, name=""):
    @recursive_do_wrap
    def _show_histogram(x, label="", subplot_array=()):
        if isinstance(x, torch.Tensor):
            x = x.squeeze().cpu().detach().numpy()
        x = x.reshape(-1)
        if len(subplot_array) > 0:
            plt.subplot(*subplot_array)
        plt.hist(x, bins=100, alpha=0.5, label=label)
        if len(subplot_array) > 0:
            plt.legend()

    if not isinstance(data, (dict, list)):
        _show_histogram(data, name)
    else:
        _show_histogram(data, subplot_array=True)

    plt.legend()
    plt.show()


@data_filter
def show_histogram_individual(data, name=""):
    @recursive_do_wrap
    def _show_histogram(x, label=""):
        if isinstance(x, torch.Tensor):
            x = x.squeeze().cpu().detach().numpy()
        x = x.reshape(-1)
        plt.hist(x, bins=100, alpha=0.5, label=label)
        plt.legend()
        plt.show()

    if not isinstance(data, (dict, list)):
        _show_histogram(data, name)
    else:
        _show_histogram(data)


@data_filter
def show_histogram(data, name=""):
    @recursive_do_wrap
    def _show_histogram(x, label=""):
        if isinstance(x, torch.Tensor):
            x = x.squeeze().cpu().detach().numpy()
        x = x.reshape(-1)
        plt.hist(x, bins=100, alpha=0.5, label=label)

    if not isinstance(data, (dict, list)):
        _show_histogram(data, name)
    else:
        _show_histogram(data)

    plt.legend()
    plt.show()


@data_filter
def plot_3d_points(data, name=""):
    fig = plt.figure()
    ax = Axes3D(fig)

    @recursive_do_wrap
    def _show_3d_points(x, label=""):
        if isinstance(x, torch.Tensor):
            x = x.squeeze().cpu().detach().numpy()
        if len(x.shape) > 2:
            for i in range(len(x)):
                _show_3d_points(x[i], label + "_" + str(i))
        else:
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], alpha=0.3, label=label)

    if not isinstance(data, (dict, list)):
        _show_3d_points(data, name)
    else:
        _show_3d_points(data)

    plt.legend()
    plt.show()


def plot_skeleton(offsets, edges, name=""):
    fig = plt.figure()
    ax = Axes3D(fig)

    @recursive_everydo_wrap
    def _show_skeleton(x, edge, label=""):
        if isinstance(x, torch.Tensor):
            x = x.squeeze().cpu().detach().numpy()
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], alpha=0.3, label=label)
        for i in range(len(x)):
            r = random.uniform(1.0, 1.2)
            pos = x[i] * r
            ax.text(*pos, f"{i}")
        color = get_random_color()
        for e in edge:
            start, end, _ = e
            curve = []
            for p1, p2 in zip(x[start], x[end]):
                curve.append([p1, p2])
            ax.plot(*curve, c=color)

    if len(edges) == 1:
        edges = [edges[0] for _ in range(len(offsets))]

    _show_skeleton(offsets, edges, label=name)

    plt.legend()
    plt.show()


def vis_code_pca(xs, output_codes, quants):
    """
    Args:
        output_codes (list): (tensor, int) [1, P]
    """
    xs = xs.flatten(1)
    xs = to_cpu_numpy(xs)
    output_codes = to_cpu_numpy(output_codes)
    quants = to_cpu_numpy(quants)

    # pca
    pca_manifold = PCA(n_components=2, whiten=True).fit_transform(xs)

    plt.figure(figsize=(19.2, 10.8))
    for i in range(quants.shape[0]):
        plt.scatter(
            pca_manifold[output_codes == i, 0],
            pca_manifold[output_codes == i, 1],
            alpha=0.5,
        )
    plt.title(f"PCA Visualization of {xs.shape[-1]}-d latent segment")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()

    # tsne
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(xs)

    plt.figure(figsize=(19.2, 10.8))
    for i in range(quants.shape[0]):
        plt.scatter(
            transformed_data[output_codes == i, 0],
            transformed_data[output_codes == i, 1],
            alpha=0.5,
        )
    plt.title(f"t-SNE Visualization of {xs.shape[-1]}-d latent segment")
    plt.show()

    # code hist
    plt.figure(figsize=(19.2, 10.8))
    plt.hist(output_codes, bins="auto", density=True, alpha=0.5)
    plt.title(f"code hist")

    plt.show()


def plot_grab_prediction():
    pass


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def vis_smpl_forward_animation(
    gender,
    n_comps,
    vtemp,
    smplx_pose_param,
    obj_mesh,
    mat_obj,
):
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
    ]
    colors = [COLORS[0], COLORS[2], COLORS[2], COLORS[1], COLORS[2]]
    ground_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_line_set.colors = o3d.utility.Vector3dVector(colors)
    smpl_model = smplx.create(
        "./datasets/models",
        model_type="smplx",
        gender=gender,
        num_pca_comps=n_comps,
        v_template=vtemp,
        batch_size=1,
        use_pca=False,
    )
    T = smplx_pose_param["body_pose"].shape[0]
    first_frame = {k: v[:1] for k, v in smplx_pose_param.items()}
    output = smpl_model(**first_frame, return_verts=True)
    verts = output.vertices.detach().cpu().numpy()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
    smpl_mesh.compute_vertex_normals()
    smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(smpl_mesh)
    vis.add_geometry(ground_line_set)
    vis.add_geometry(obj_mesh)
    for i in range(T):
        frame_pose = {k: v[i : i + 1] for k, v in smplx_pose_param.items()}
        output = smpl_model(**frame_pose, return_verts=True)
        verts = output.vertices.detach().cpu().numpy()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
        smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])
        vis.update_geometry(smpl_mesh)

        obj_mesh.transform(mat_obj[i])
        vis.update_geometry(obj_mesh)

        vis.poll_events()
        vis.update_renderer()
        obj_mesh.transform(np.linalg.inv(mat_obj[i]))
        print(i)
    vis.destroy_window()
