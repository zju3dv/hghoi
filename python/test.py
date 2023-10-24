import argparse
from motion.config import get_config
from motion.engine.builder import TRAINERS
from motion.utils.utils import load_ckpt, print_dict


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
        "--test-epoch",
        default=0,
        type=str,
        help="Choose test model with epoch",
    )
    parser.add_argument(
        "--split",
        default="valid",
        type=str,
        help="Choose test on valid dataset or train dataset",
    )

    args = parser.parse_args()
    cfg = get_config(args.config, args.opts)

    trainer = TRAINERS.get(cfg.TYPE)
    cfg = cfg.cfg
    model, device = trainer.build_model(cfg)
    model = load_ckpt(model, cfg.output_dir, args.test_epoch)
    pipeline = trainer.build_pipeline(cfg, model, optimizer=None)
    results = trainer.test(cfg, pipeline, device, args.test_epoch, args.split)
    print("Results:")
    print_dict(results)


if __name__ == "__main__":
    main()
