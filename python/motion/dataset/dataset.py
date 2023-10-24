import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_valid=False):
        super().__init__()
        self.cfg = cfg
        split = "train"
        # split = "test"
        if is_valid:
            split = "test"
        self.is_valid = is_valid
        self.split = split
        self._load_cfg(cfg)
        self._load_data()
        self._calculate_statistics()
        self._preprocess_dataset()

    def _load_cfg(self, cfg):
        pass

    def _load_data(self):
        raise NotImplementedError

    def _calculate_statistics(self):
        pass

    def _preprocess_dataset(self):
        pass

    def _load_statistics(self):
        return None

    def preprocess_data_socket(self, data):
        return data

    def postprocess_data_socket(self, data):
        return data
