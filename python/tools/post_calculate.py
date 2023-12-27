import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
import sys

sys.path.extend("../")
from motion.config import get_config, save_config
from motion.dataset import build_train_dataloader, build_valid_dataloader
from motion.utils.utils import to_cpu_numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Milestone or Traj",
    )
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()
    cfg = get_config(args.config, args.opts)
    cfg.cfg.DATALOADER.batch_size = 256
    dataloader = build_train_dataloader(cfg.cfg)

    NAME = args.name

    trajs = []
    for data in tqdm(dataloader):
        traj = data["traj"]
        mask = data["mask"]
        trajs.append(traj[mask])
    trajs = torch.cat(trajs, dim=0)
    max_v = to_cpu_numpy(trajs.max(dim=0)[0])
    min_v = to_cpu_numpy(trajs.min(dim=0)[0])
    save_path = os.path.join(cfg.cfg.DATASET.cfg.data_dir, f"train/{NAME}MinMax.txt")
    np.savetxt(save_path, [max_v, min_v])
    mean_v = to_cpu_numpy(trajs.mean(dim=0))
    std_v = to_cpu_numpy(trajs.std(dim=0))
    save_path = os.path.join(cfg.cfg.DATASET.cfg.data_dir, f"train/{NAME}Norm.txt")
    np.savetxt(save_path, [mean_v, std_v])

    pose = dataloader.dataset.motion_data
    max_v = to_cpu_numpy(pose.max(axis=0))
    min_v = to_cpu_numpy(pose.min(axis=0))
    save_path = os.path.join(cfg.cfg.DATASET.cfg.data_dir, "train/InputMinMax.txt")

    np.savetxt(save_path, [max_v, min_v])
    print(f"Save at {save_path}")


if __name__ == "__main__":
    main()
