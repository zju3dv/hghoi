import copy
import os
import numpy as np
from tqdm.contrib import tzip

import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.dataset.samp.utils import load_norm_data
from motion.utils.utils import to_cpu_numpy, makedirs, shutil_copy, denormalize


def save_motioncode(
    xs,
    codes,
    quants,
    statistics,
    dataset_save_dir,
    split,
    original_dataset_dir,
    *args,
    **kwargs,
):
    """
    Args:
        xs (list(tensor)): List([1, C]) tensor of input
        codes (Dict(list(int64 tensor))): List([1, 1]) tensor of predicted code
        statistics (dict): mean and std of input and output
        dataset_save_dir (str): the path to save the predicted code
        split (str): train or test
        original_dataset_dir (str): the path to localpose
    """
    dataset_save_dir = os.path.join(dataset_save_dir, split)
    makedirs(dataset_save_dir)
    for k in codes.keys():
        code = torch.cat(codes[k], dim=0)
        quant = torch.cat(quants[k], dim=0)
        code = to_cpu_numpy(code)
        quant = to_cpu_numpy(quant)

        print(f"Start to save code {k}...")
        np.savetxt(os.path.join(dataset_save_dir, f"Code_{k}.txt"), code)
        print(f"Start to save quant {k}...")
        np.savetxt(os.path.join(dataset_save_dir, f"Quant_{k}.txt"), quant)
    print("Finish saving code!")
