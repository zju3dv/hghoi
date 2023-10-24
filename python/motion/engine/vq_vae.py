import copy
from tqdm import tqdm

from motion.engine.trainer import DefaultTrainer
from motion.dataset import build_forward_dataloader
from motion.utils.func_wrapper import gpu_memory_clear, call_func_monitor
from motion.utils.utils import to_device, to_cpu
from .builder import TRAINERS


@TRAINERS.register_module()
class VQVAETrainer(DefaultTrainer):
    @classmethod
    @call_func_monitor
    @gpu_memory_clear
    def postprocess_func(
        cls,
        cfg,
        pipeline,
        device,
        split,
        vis,
        forward_func_name,
        func_name,
        func_kwargs,
    ):
        results = []
        dataloader = cls.build_forward_dataloader(cfg, split)
        for data in tqdm(dataloader):
            data = to_device(data, device)
            forward_func = getattr(pipeline, forward_func_name)
            result = forward_func(data)
            results.append(to_cpu(result))
        getattr(pipeline, func_name)(
            results, cfg, split, vis, dataloader, **func_kwargs
        )
