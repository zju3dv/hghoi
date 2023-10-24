# import isaacgym
import torch
import argparse
from motion.config import get_config, save_config
from motion.engine import build_trainer

# For debug
# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)


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
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()
    cfg = get_config(args.config, args.opts)
    trainer = build_trainer(cfg)
    save_config(trainer, cfg)
    trainer.train_loop()


if __name__ == "__main__":
    main()
