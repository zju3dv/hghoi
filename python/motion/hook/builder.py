from motion.utils.registry import Registry


HOOKS = Registry("hook")


def build_hook(cfg, trainer_cfg):
    new_cfg = cfg.clone()
    new_cfg["cfg"] = trainer_cfg
    return HOOKS.build(new_cfg)
