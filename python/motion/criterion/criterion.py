import os
from tqdm import tqdm

from motion.config import get_config
from motion.engine.builder import TRAINERS
from motion.utils.utils import load_ckpt, print_dict

from motion.utils.func_wrapper import call_func_monitor, time_monitor
from motion.utils.utils import multiply, np_mean
from motion.utils.basics import DictwithEmpty


class BaseCriterion:
    def __init__(self, weight):
        self.weight = weight

    @call_func_monitor
    @time_monitor
    def compute_result(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        output = self.compute_result(*args, **kwargs)
        return multiply(output, self.weight)


class SimpleCriterion(BaseCriterion):
    def __init__(
        self, cfg, weight, output_dir, save_results=False, external_model_config=None
    ):
        super().__init__(weight)
        self.cfg = cfg
        self.error_types = []
        self.output_dir = os.path.join(output_dir, "results")
        self.save_results = save_results
        self.external_model_cfg = external_model_config
        if external_model_config is not None:
            self._build_external_model()
        self.load_cfg(cfg)

    def load_cfg(self, cfg):
        pass

    @call_func_monitor
    @time_monitor
    def _build_external_model(self):
        cfg = get_config(self.external_model_cfg)
        trainer = TRAINERS.get(cfg.TYPE)
        cfg = cfg.cfg
        self.external_model_cfg = cfg
        model, self.model_device = trainer.build_model(cfg)
        self.model = load_ckpt(model, cfg.output_dir)

    @call_func_monitor
    @time_monitor
    def compute_result(self, outputs, epoch=0, split="test", tb_writter=None):
        errors = DictwithEmpty([])
        process_outputs = []
        for i, output in enumerate(tqdm(outputs)):
            pred, gt = output
            es, process_output = self._compute_result(pred, gt)
            if self.save_results:
                self._save_result(i, output, epoch, split)
            for k in es.keys():
                errors[k].append(es[k])
            process_outputs.append(process_output)
        process_errors, process_outputs = self._compute_process_result(
            process_outputs, split
        )
        if self.save_results:
            self._save_process_result(process_outputs, epoch, split, tb_writter)
        errors.update(process_errors)
        errors = np_mean(errors)
        return errors

    def _compute_result(self, pred, gt):
        raise NotImplementedError

    def _compute_process_result(self, process_output, split):
        return {}, process_output

    def _save_result(self, i, output, epoch, split):
        pass

    def _save_process_result(self, processoutput, epoch, split, tb_writter=None):
        pass

    def _external_network_process(self, data_total, **kwargs):
        pass
