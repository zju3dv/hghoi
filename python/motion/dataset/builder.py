from torch.utils.data.dataloader import DataLoader
from . import collate

from motion.utils.registry import Registry


DATASETS = Registry("dataset")


def get_dataset(cfg):
    dataset_cfg = cfg.DATASET
    return DATASETS.get(dataset_cfg.TYPE)


def build_dataset(cfg, is_valid=False, split="train"):
    dataset_cfg = cfg.DATASET.clone()
    dataset_cfg["is_valid"] = is_valid
    dataset = DATASETS.build(dataset_cfg)
    return dataset


def build_train_dataloader(cfg):
    train_dataset = build_dataset(cfg)
    collate_fn = cfg.DATASET.cfg.get("collate_fn", "default_collate")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        collate_fn=getattr(collate, collate_fn),
        **cfg.DATALOADER,
    )
    return train_dataloader


def build_valid_dataloader(cfg, split):
    val_dataset = build_dataset(cfg, is_valid=(split != "train"), split=split)
    val_collate_fn = cfg.DATASET.cfg.get("valid_collate_fn", "default_collate")
    valid_batch_size = cfg.DATASET.cfg.get("valid_batch_size", 1)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=getattr(collate, val_collate_fn),
    )
    return val_dataloader


def build_forward_dataloader(cfg, split):
    # For vq-vae to extract code of training data
    forward_collate_fn = cfg.DATASET.cfg.get("forward_collate_fn", "default_collate")
    forward_batch_size = cfg.DATASET.cfg.get("forward_batch_size", 1)
    forward_dataset = build_dataset(cfg, split != "train")
    forward_dataloader = DataLoader(
        forward_dataset,
        batch_size=forward_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        collate_fn=getattr(collate, forward_collate_fn),
    )
    return forward_dataloader
