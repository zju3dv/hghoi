from tqdm import tqdm

from motion.modeling import build_model
from motion.dataset import build_forward_dataloader, build_dataset
from motion.dataset import build_train_dataloader, build_valid_dataloader
from motion.optim import build_optimizer, build_scheduler
from motion.pipeline import build_pipeline
from motion.hook import build_hook
from motion.utils.utils import to_device, to_cpu, count_parameters
from motion.utils.log import build_logger, build_tb_writter
from motion.utils.func_wrapper import (
    gpu_memory_clear,
    call_func_monitor,
    recursive_wrap,
)
from .builder import TRAINERS


class BaseTrainer:
    def __init__(self):
        self._hooks = []
        self.start_iters = 0
        self.iters = 0
        self.epochs = 0
        self.max_epochs = 0
        self.step_infos = None
        self.dataloader = None
        self.device = None

    @call_func_monitor
    def register_hooks(self, cfg):
        pass

    def train_loop(self):
        self.before_train()
        while self.epochs < self.max_epochs:
            self.before_epoch()
            self.run_epoch()
            self.epochs += 1
            self.after_epoch()
        self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train(self)

    def after_train(self):
        for h in self._hooks:
            h.after_train(self)

    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch(self)

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch(self)

    def before_step(self):
        for h in self._hooks:
            h.before_step(self)

    def after_step(self):
        for h in self._hooks:
            h.after_step(self)

    def run_epoch(self):
        for data in self.train_dataloader:
            self.before_step()
            data = to_device(data, self.device)
            self.step_infos = self.run_step(data)
            self.after_step()
            self.iters += 1

    def run_step(self, *args, **kwargs):
        raise NotImplementedError


@TRAINERS.register_module()
class DefaultTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.logger = build_logger(cfg.output_dir)
        self.tb_writter = build_tb_writter(cfg.output_dir)
        self.register_hooks(cfg)
        self.model, self.device = self.build_model(cfg, self.logger)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_scheduler(cfg, self.optimizer)
        self.pipeline = self.build_pipeline(
            cfg, self.model, self.optimizer, self._hooks
        )
        self.train_dataloader = self.build_train_dataloader(cfg)
        self.iters_per_epoch = len(self.train_dataloader)
        self.max_epochs = cfg.SCHEDULE.max_epochs

    @call_func_monitor
    def register_hooks(self, cfg):
        hook_cfg = cfg.HOOK
        for v in hook_cfg.values():
            self._register_hook(build_hook(v, cfg))

    def _register_hook(self, hook):
        inserted = False
        priority = hook.priority
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
        self.logger.info(f"{type(hook).__name__} has been registered!")

    @classmethod
    @call_func_monitor
    def build_model(cls, cfg, logger=None):
        model, device = build_model(cfg)
        if logger is not None:
            logger.info(model)
            logger.info(f"Model has {count_parameters(model)} trainable parameters")
            logger.info(
                f"Model has {count_parameters(model, requires_grad=False)} parameters in total"
            )
        return model, device

    @classmethod
    @call_func_monitor
    def build_train_dataloader(cls, cfg):
        dataloader = build_train_dataloader(cfg)
        return dataloader

    @classmethod
    @call_func_monitor
    def build_valid_dataloader(cls, cfg, split):
        dataloader = build_valid_dataloader(cfg, split)
        return dataloader

    @classmethod
    @call_func_monitor
    def build_optimizer(cls, cfg, model):
        optimizer = build_optimizer(cfg, model.parameters())
        return optimizer

    @classmethod
    @call_func_monitor
    def build_scheduler(cls, cfg, optimizer):
        scheduler = build_scheduler(cfg, optimizer)
        return scheduler

    @classmethod
    @call_func_monitor
    def build_pipeline(cls, cfg, model, optimizer, hook=None):
        return build_pipeline(cfg, model, optimizer, hook)

    @classmethod
    @call_func_monitor
    @gpu_memory_clear
    def test(cls, cfg, pipeline, device, epoch, split="valid", tb_writter=None):
        results = []
        dataloader = cls.build_valid_dataloader(cfg, split)
        for data in tqdm(dataloader):
            data = to_device(data, device)
            result = pipeline.forward_step(data)
            results.append(to_cpu(result))
        return pipeline.compute_results(
            results, cfg, epoch, split, tb_writter=tb_writter
        )

    @classmethod
    @call_func_monitor
    def build_forward_dataloader(cls, cfg, split):
        dataloader = build_forward_dataloader(cfg, split)
        return dataloader

    @classmethod
    @call_func_monitor
    @gpu_memory_clear
    def forward(cls, cfg, pipeline, device, split, epoch):
        results = []
        dataloader = cls.build_forward_dataloader(cfg, split)
        for data in tqdm(dataloader):
            data = to_device(data, device)
            result = pipeline.forward_step(data)
            results.append(to_cpu(result))
        return pipeline.compute_results(results, cfg, epoch, split)

    @classmethod
    @call_func_monitor
    @gpu_memory_clear
    def socket_loop(cls, cfg, pipeline, device, *args, **kwargs):
        dataset = build_dataset(cfg, is_valid=True)

        def socket_func(data):
            data = dataset.preprocess_data_socket(data)
            data = to_device(data, device)
            result = pipeline.socket_forward(data, *args, **kwargs)
            result = to_cpu(result)
            result = dataset.postprocess_data_socket(result, to_cpu(data))
            return result

        return socket_func

    def run_epoch(self):
        self.pipeline.mode_train()
        super().run_epoch()
        if self.scheduler is not None:
            self.scheduler.step()

    def run_step(self, *args, **kwargs):
        output = self.pipeline.run_step(*args, **kwargs)
        if self.scheduler is not None:
            if hasattr(self.scheduler, "batch_step"):
                self.scheduler.batch_step()
        return output

    @call_func_monitor
    def do_eval(self, split="valid", tb_writter=None):
        self.logger.info(f"Start to evaluate on {split} dataset...")
        return self.test(
            self.cfg,
            self.pipeline,
            self.device,
            self.epochs,
            split,
            tb_writter=tb_writter,
        )
