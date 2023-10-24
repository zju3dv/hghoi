import collections
import os
import shutil
import torch
import pickle
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from motion.utils.func_wrapper import (
    recursive_do_wrap,
    recursive_wrap,
    recursive_dict_wrap,
)


def get_socket_func(TRAINERS, cfg, epoch, external_model=None):
    trainer = TRAINERS.get(cfg.TYPE)
    cfg = cfg.cfg
    model, device = trainer.build_model(cfg)
    model = load_ckpt(model, cfg.output_dir, epoch)
    pipeline = trainer.build_pipeline(cfg, model, optimizer=None)
    socket_func = trainer.socket_loop(
        cfg, pipeline, device, external_model=external_model
    )
    return socket_func


def get_model(TRAINERS, cfg, epoch):
    trainer = TRAINERS.get(cfg.TYPE)
    cfg = cfg.cfg
    model, device = trainer.build_model(cfg)
    model = load_ckpt(model, cfg.output_dir, epoch)
    return model


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_random_slice(x, max_step):
    lower_bound = 0
    upper_bound = x.shape[0] - max_step
    if lower_bound >= upper_bound:
        st_idx = 0
    else:
        st_idx = np.random.randint(low=lower_bound, high=upper_bound)
    ed_idx = st_idx + max_step
    rnd_slice = slice(st_idx, ed_idx)

    return rnd_slice


def repeat_last_frame(array, max_step):
    length = array.shape[0]
    if length < max_step:
        last_array = array[length - 1 : length]
        repeat = np.ones(len(array.shape), dtype=int)
        repeat[0] = max_step - length
        last_array = np.tile(last_array, repeat)
        return np.concatenate((array, last_array))
    else:
        return array


def load_from_pickle(path, data_type=None):
    with open(os.path.join(path), "rb") as f:
        data = pickle.load(f)
        if data_type is not None:
            convert_np_type(data, data_type)
        f.close()
    return data


def instance_norm(data):
    """
    data (tensor): B, T, C
    """
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    return (data - mean) / (std + 1e-9), mean, std


def denorm(data, pid, means, stds):
    mean = means[pid, ...]
    std = stds[pid, ...]
    return data * std + mean


def pure_denorm(data, means, std):
    return data * std + means


def normalize(x, mean, std):
    if isinstance(x, torch.Tensor) and not isinstance(mean, torch.Tensor):
        mean = to_tensor(mean)
        std = to_tensor(std)
    return (x - mean) / std


def denormalize(x, mean, std):
    if isinstance(x, torch.Tensor) and not isinstance(mean, torch.Tensor):
        mean = to_tensor(mean)
        std = to_tensor(std)
    return x * std + mean


def normalize_to_neg_one_to_one(x, x_min=0.0, x_max=1.0):
    l = x_max - x_min
    if isinstance(l, (torch.Tensor, np.ndarray)):
        l[l < 1e-6] = 1.0
    x = (x - x_min) / l
    return 2 * x - 1


def unnormalize_to_zero_to_one(x, x_min=0.0, x_max=1.0):
    x = (x + 1) * 0.5
    l = x_max - x_min
    x = x * l + x_min
    return x


def shutil_copy(src, dst):
    try:
        try:
            shutil.copy(src, dst)
        except IsADirectoryError:
            shutil.copytree(src, dst)
    except FileExistsError:
        print(f'"{dst}" already exists!')
        pass


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def symlink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def pd_load(path):
    data = pd.read_csv(
        path,
        header=None,
        na_filter=False,
        delim_whitespace=True,
    )
    return data


def pd_load2numpy(path):
    return pd_load(path).to_numpy()


def save_array2txt(path, x):
    with open(path, "w") as f:
        if len(x.shape) == 2:
            str_x = [" ".join([str(x_) for x_ in line]) + "\n" for line in x]
        elif len(x.shape) == 1:
            str_x = [str(x_) + "\n" for x_ in x]
        else:
            raise NotImplementedError
        f.writelines(str_x)


def get_random_color():
    color = "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    return color


def np_softmax(x, *args, **kwargs):
    softness = kwargs.pop("softness", 1.0)
    maxi, mini = np.max(x, **kwargs), np.min(x, **kwargs)
    return maxi + np.log(softness + np.exp(mini - maxi))


def np_softmin(x, *args, **kwargs):
    return -np_softmax(-x, **kwargs)


def torch_softmax(x, *args, **kwargs):
    softness = kwargs.pop("softness", 1.0)
    maxi, mini = torch.max(x, **kwargs)[0], torch.min(x, **kwargs)[0]
    return maxi + torch.log(softness + torch.exp(mini - maxi))


def torch_softmin(x, *args, **kwargs):
    return -torch_softmax(-x, *args, **kwargs)


def calculate_grad_norm(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def count_parameters(model, requires_grad=False):
    # count parameteres of a torch model
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


@recursive_wrap
def detach(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach()
    else:
        pass
    return obj


@recursive_dict_wrap
def multiply(x, b):
    return x * b


@recursive_dict_wrap
def np_mean(obj):
    try:
        obj = np.mean(np.array(obj))
    except ValueError:
        # obj shape are not same
        obj = np.concatenate(obj, axis=0)
        obj = np.mean(obj)
    return obj


@recursive_wrap
def np_transpose(obj, target_shape):
    return obj.transpose(target_shape)


@recursive_wrap
def convert_np_type(obj, target_type):
    obj = obj.astype(target_type)
    return obj


def to_tensor(obj, dtype=torch.float32):
    return torch.tensor(obj, dtype=dtype)


@recursive_wrap
def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        pass
    return obj


@recursive_wrap
def to_cpu_numpy(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    else:
        pass
    return obj


@recursive_wrap
def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu()
    else:
        pass
    return obj


@recursive_wrap
def to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        if torch.cuda.is_available():
            obj = obj.cuda(non_blocking=True)
        elif True:
            pass
        elif not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            mps_device = torch.device("mps")
            obj = obj.to(mps_device)
    else:
        pass
    return obj


@recursive_wrap
def mode_train(obj):
    obj.train()
    return obj


@recursive_wrap
def mode_eval(obj):
    obj.eval()
    return obj


@recursive_wrap
def load_ckpt(load_obj, resume_dir, epoch=None):
    if epoch is None:
        epoch = os.listdir(os.path.join(resume_dir, "ckpts"))
        assert len(epoch) > 0, f"No models in {resume_dir}/ckpts"
        epoch = max([int(e) for e in epoch])
    resume_dir = os.path.join(resume_dir, "ckpts", str(epoch))
    path = os.path.join(resume_dir, "model.pth")
    # assert os.path.exists(path), f"The model does not exists at {path}"
    ckpt = torch.load(path, map_location="cpu")
    ckpt = to_cuda(ckpt)
    try:
        if "state" in ckpt.keys():
            ckpt = ckpt["state"]
        load_obj.load_state_dict(ckpt, strict=True)
        print(f"Succesfully load ckpt from {path}")
    except Exception as e:
        print(e)
        try:
            load_obj.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print(e)
    return load_obj


def print_dict(infos, prefix=""):
    if isinstance(infos, dict):
        for k, v in infos.items():
            print_dict(v, prefix=prefix + k + "_")
    else:
        print(f"    {prefix[:-1]:30s}: {infos:.6g}")


def flatten_dict(dict_, parent_key="", sep="."):
    """Flattens a nested dictionary. Namedtuples within
    the dictionary are converted to dicts.
    Args:
        dict_: The dictionary to flatten.
        parent_key: A prefix to prepend to each key.
        sep: Separator between parent and child keys, a string. For example
            { "a": { "b": 3 } } will become { "a.b": 3 } if the separator
            is ".".
    Returns:
        A new flattened dictionary.
    """
    items = []
    for key, value in dict_.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, tuple) and hasattr(value, "_asdict"):
            dict_items = collections.OrderedDict(zip(value._fields, value))
            items.extend(flatten_dict(dict_items, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


class MeanstdNormalizer:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def calculate(self, x, chunk_size=30000):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-3] = 1.0
        return mean, std

    def set_parameter(self, mean=None, std=None):
        if mean is not None:
            self.mean = mean
        if std is not None:
            std[std < 1e-3] = 1.0
            self.std = std

    def normalize(self, x, mean=None, std=None):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        return normalize(x, mean, std)

    def denormalize(self, x, mean=None, std=None):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        return denormalize(x, mean, std)


class MinmaxNormalizer:
    def __init__(self, x_min=0.0, x_max=1.0):
        self.x_min = x_min
        self.x_max = x_max

    def calculate(self, x, chunk_size=30000):
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        return x_min, x_max

    def set_parameter(self, x_min=None, x_max=None):
        if x_min is not None:
            self.x_min = x_min
        if x_max is not None:
            self.x_max = x_max

    def normalize(self, x, x_min=None, x_max=None):
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        return normalize_to_neg_one_to_one(x, x_min, x_max)

    def denormalize(self, x, x_min=None, x_max=None):
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        return unnormalize_to_zero_to_one(x, x_min, x_max)


def welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return (float("nan"), float("nan"))
    else:
        variance_n = M2 / count
        variance = M2 / (count - 1)
        variance[variance < 1e-4] = 1.0
        return (mean, variance**0.5)


def calculate_meanstd(x, chunk_size=100000):
    # calculate mean and std on very large array
    # x: [np.array(T, D)]

    # initialize
    aggregate = (0, np.zeros((x[0].shape[1],)), np.zeros((x[0].shape[1],)))

    for array in x:
        mean_array = np.mean(array, axis=0)
        aggregate = welford_update(aggregate, mean_array)

    mean, std = finalize(aggregate)
    return mean, std


def minmax_update(x, min_values, max_values):
    min_values = np.minimum(min_values, x.min(axis=0))
    max_values = np.maximum(max_values, x.max(axis=0))
    return min_values, max_values


def calculate_minmax(x, chunk_size=100000):
    N = len(x)
    D = x[0].shape[1]
    min_values = np.full(D, float("inf"))
    max_values = np.full(D, -float("inf"))

    n_chunks = N // chunk_size
    for i in range(n_chunks):
        chunk = x[i * chunk_size : (i + 1) * chunk_size]
        chunk = np.concatenate(chunk, axis=0)
        min_values = np.minimum(min_values, chunk.min(axis=0))
        max_values = np.maximum(max_values, chunk.max(axis=0))

    # last chunk
    if N % chunk_size != 0:
        chunk = x[n_chunks * chunk_size :]
        chunk = np.concatenate(chunk, axis=0)
        min_values = np.minimum(min_values, chunk.min(axis=0))
        max_values = np.maximum(max_values, chunk.max(axis=0))

    return min_values, max_values
