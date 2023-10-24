from motion.utils.registry import Registry


PIPELINES = Registry("pipeline")


def build_pipeline(cfg, model, optimizer, hook):
    pipeline_cfg = cfg.PIPELINE.clone()
    pipeline_cfg["model"] = model
    pipeline_cfg["optimizer"] = optimizer
    pipeline_cfg["hook"] = hook
    return PIPELINES.build(pipeline_cfg)
