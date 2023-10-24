import torch

from motion.utils.registry import Registry


MODULES = Registry("modules")


def build_module(cfg):
    module = MODULES.build(cfg)
    return module
