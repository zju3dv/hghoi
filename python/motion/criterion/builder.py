from motion.utils.registry import Registry


CRITERIONS = Registry("criterion")


def build_criterion(cfg):
    return CRITERIONS.build(cfg)
