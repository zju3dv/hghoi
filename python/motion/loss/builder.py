from motion.utils.registry import Registry


LOSSES = Registry("loss")


def build_loss(cfg):
    return LOSSES.build(cfg)
