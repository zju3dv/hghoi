import math
import torch
import motion.optim.cyclic_scheduler as cyclic_scheduler


class lambda_func_lib:
    @staticmethod
    def linear_epoch_func(cfg):
        return lambda epoch: (cfg.max_epochs - epoch) / float(cfg.max_epochs)

    @staticmethod
    def cosine_epoch_func(cfg):
        return lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / cfg.max_epochs))

    @staticmethod
    def cosine_minlr_epoch_func(cfg):
        def lr_func(epoch):
            lr = cfg.lr
            min_lr = cfg.min_lr
            new_lr = min_lr + 0.5 * (lr - min_lr) * (
                1 + math.cos(math.pi * epoch / cfg.max_epochs)
            )
            factor = new_lr / lr
            return factor

        return lr_func

    @staticmethod
    def constant_epoch_func(cfg):
        return lambda epoch: 1.0


def build_optimizer(cfg, parameters):
    optim_cfg = cfg.OPTIMIZER
    args = optim_cfg.copy()
    optim_type = args.pop("TYPE")
    optimizer = getattr(torch.optim, optim_type)(parameters, **args)
    return optimizer


def build_scheduler(cfg, optimizer):
    scheduler_cfg = cfg.SCHEDULE.get("scheduler", None)
    if scheduler_cfg is None:
        return None
    args = scheduler_cfg.copy()
    scheduler_type = args.pop("TYPE")
    if hasattr(torch.optim.lr_scheduler, scheduler_type):
        for k in args.keys():
            if args[k][-4:] == "func":
                args[k] = getattr(lambda_func_lib, args[k])(cfg.SCHEDULE)
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **args)
    else:
        scheduler = getattr(cyclic_scheduler, scheduler_type)(optimizer, **args)
    return scheduler
