from .trainer import DefaultTrainer
from motion.optim.optimizer import build_optimizer
from motion.utils.func_wrapper import call_func_monitor

from .builder import TRAINERS


@TRAINERS.register_module()
class EmptyTrainer(DefaultTrainer):
    """
    This trainer is for model like copyrotation which does not need training
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    @call_func_monitor
    def build_optimizer(cls, cfg, model):
        return {}


@TRAINERS.register_module()
class GANTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    @call_func_monitor
    def build_optimizer(cls, cfg, model):
        D_parameters = []
        G_parameters = []
        if isinstance(model, list):
            for m in model:
                D_parameters += m.D_parameters()
                G_parameters += m.G_parameters()
        else:
            D_parameters = model.D_parameters()
            G_parameters = model.G_parameters()
        optimizer = {
            "D": build_optimizer(cfg, D_parameters),
            "G": build_optimizer(cfg, G_parameters),
        }
        return optimizer
