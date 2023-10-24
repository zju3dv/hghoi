import types
import torch

from motion.loss import build_loss
from motion.criterion import build_criterion
from motion.hook import build_hook
from motion.utils.utils import mode_eval, mode_train
import motion.utils.func_wrapper as func_wrapper_lib
from motion.utils.func_wrapper import (
    call_func_monitor,
    recursive_wrap,
    gpu_memory_clear,
)
from motion.pipeline.builder import PIPELINES


class BasePipeline:
    def __init__(self, model, hook=None):
        self.model = model
        self._hooks = []
        if hook is not None:
            self._hooks = hook

    @call_func_monitor
    def build_loss(self, cfg):
        loss_cfg = cfg.LOSS
        for k, v in loss_cfg.items():
            setattr(self, "criterion_" + k.lower(), build_loss(v))

    @call_func_monitor
    def build_criterion(self, cfg):
        criterion_cfg = cfg.CRITERION
        criterion_cfg.output_dir = cfg.output_dir
        return build_criterion(criterion_cfg)

    def compute_results(self, results, cfg, epoch, split="test", tb_writter=None):
        criterion = self.build_criterion(cfg)
        return criterion(results, epoch, split, tb_writter)

    def before_forward(self, data):
        for h in self._hooks:
            data = h.before_forward(self, data)
        return data

    def after_forward(self, data, output):
        for h in self._hooks:
            output = h.after_forward(self, data, output)
        return output

    def before_backward(self, loss_dict):
        for h in self._hooks:
            data = h.before_backward(self, loss_dict)
        return data

    def after_backward(self, loss_dict):
        for h in self._hooks:
            output = h.after_backward(self, loss_dict)
        return output

    def run_step(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def forward_step(self, data):
        raise NotImplementedError

    def mode_eval(self, model=None):
        if model is None:
            model = self.model
        mode_eval(model)

    def mode_train(self, model=None):
        if model is None:
            model = self.model
        mode_train(model)


@PIPELINES.register_module()
class SLPipeline(BasePipeline):
    def __init__(self, cfg, model, optimizer, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.optimizer = optimizer
        self.build_loss(cfg)
        self.wrap_func(cfg)
        self.load_cfg(cfg)
        self.cfg = cfg

    def load_cfg(self, cfg):
        pass

    def wrap_func(self, cfg):
        func_wrap_pairs = cfg.func_wrap_pairs
        for func_name, wrapper in func_wrap_pairs:
            wrapper = getattr(func_wrapper_lib, wrapper)
            wrapped_func = wrapper(getattr(self, func_name))
            wrapped_func = types.MethodType(wrapped_func, self)
            setattr(self, func_name, wrapped_func)

    def forward_dummpy(self, data):
        return self.model(data)

    def model_step(self, data):
        data = self.before_forward(data)
        pred = self.forward_dummpy(data)
        pred = self.after_forward(data, pred)

        loss_dict = self.output_loss(data, pred)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum([v for k, v in loss_dict.items() if "loss" in k])
            loss_dict["total_loss"] = losses
        loss_dict["info"] = pred.get("info", None)
        loss_dict = self.before_backward(loss_dict)
        losses.backward()
        loss_dict = self.after_backward(loss_dict)
        return loss_dict

    def run_step(self, data):
        self.optimizer.zero_grad()
        loss_dict = self.model_step(data)
        self.optimizer.step()
        return loss_dict

    def _forward_step(self, data):
        output = self.forward_dummpy(data)
        return output

    @gpu_memory_clear
    @torch.no_grad()
    def forward_step(self, data):
        self.mode_eval(self.model)
        output = self._forward_step(data)
        self.mode_train(self.model)
        return output, data

    def _output_loss(self, data, pred):
        return {"loss": self.loss(pred, data)}

    def output_loss(self, data, pred):
        loss_pairs = self._output_loss(data, pred)
        return self.calculate_loss(loss_pairs)

    def calculate_loss(self, loss_pairs, postfix=""):
        if len(postfix) > 0:
            postfix = "_" + postfix
        loss_dict = {}
        for k, v in loss_pairs.items():
            l, w = self._calculate_loss(k.split("_")[0], *v)
            loss_dict[k + "_loss" + postfix] = l
            loss_dict[k + postfix + "_weight"] = w
        return loss_dict

    def _calculate_loss(self, criterion, pred, gt=None, *args, **kwargs):
        w = 1.0
        criterion = getattr(self, "criterion_" + criterion)
        if "weighted" in type(criterion).__name__:
            w *= args[0].mean().item()
        return criterion(pred, gt, *args, **kwargs), criterion.weight * w

    # @gpu_memory_clear
    # @torch.no_grad()
    def socket_forward(self, data, *args, **kwargs):
        self.mode_eval(self.model)
        return self.model.socket_forward(data, *args, **kwargs)


@PIPELINES.register_module()
class OptimPipeline(SLPipeline):
    def socket_forward(self, data, *args, **kwargs):
        return data
