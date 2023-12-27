import sys

sys.path.extend("../")
import torch

torch.autograd.set_detect_anomaly(True)
import argparse

from motion.config import get_config
from motion.engine.builder import TRAINERS
from motion.utils.utils import load_ckpt


if __name__ == "__main__":
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
        "--epoch",
        required=True,
        type=int,
        help="Choose test model with epoch",
    )
    parser.add_argument(
        "--func",
        required=True,
        default="extract_code",
        type=str,
        help="Extract function you want to call",
    )
    parser.add_argument(
        "--forward-func",
        required=True,
        default="forward_step",
        type=str,
        help="Forward function you want to call",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize code distribution",
    )
    parser.add_argument(
        "--onlytrain",
        action="store_true",
        help="Only process train data",
    )
    args = parser.parse_args()
    cfg = get_config(args.config, args.opts)

    trainer = TRAINERS.get(cfg.TYPE)

    cfg = cfg.cfg
    model, device = trainer.build_model(cfg)
    model = load_ckpt(model, cfg.output_dir, args.epoch)
    pipeline = trainer.build_pipeline(cfg, model, optimizer=None)

    if not args.onlytrain:
        trainer.postprocess_func(
            cfg,
            pipeline,
            device,
            "test",
            args.vis,
            args.forward_func,
            args.func,
            cfg.FUNC,
        )

    trainer.postprocess_func(
        cfg,
        pipeline,
        device,
        "train",
        args.vis,
        args.forward_func,
        args.func,
        cfg.FUNC,
    )
