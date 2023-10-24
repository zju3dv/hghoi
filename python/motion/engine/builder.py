from motion.utils.registry import Registry


TRAINERS = Registry("trainer")


def build_trainer(cfg):
    return TRAINERS.build(cfg)
