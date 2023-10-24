import os
from motion.config.default import _C


CONFIG_FILE_SEPARATOR = ","


def get_config(config_paths=None, opts=None):
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    return config


def save_config(engine, cfg):
    logger = engine.logger
    logger.info(f"Running with full config:\n{cfg.dump()}")
    cfg_path = os.path.join(cfg.cfg.output_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(cfg.dump())
    logger.info(f"Full config saved to {cfg_path}")
