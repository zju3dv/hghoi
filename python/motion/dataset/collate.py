from typing import Dict
import torch

from torch.utils.data.dataloader import default_collate
from motion.utils.tensor import lengths_to_mask
from motion.utils.utils import to_tensor
from motion.utils.basics import DictwithEmpty
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
)


def mixamo_test_collate(batch):
    assert (
        len(batch) == 1
    ), f"we assume batch_size = 1 here, but batch_size = {len(batch)}"
    batch = batch[0]
    for i in range(len(batch)):
        batch[i]["offset"] = batch[i]["offset"].unsqueeze(0)
        batch[i]["offset_idx"] = torch.tensor(batch[i]["offset_idx"], dtype=torch.long)
        batch[i]["ind"] = torch.tensor(batch[i]["ind"], dtype=torch.long)
        batch[i]["characters"] = batch[i]["characters"]
    return batch


def remove_batch_collate(batch):
    obj_names = batch[0].pop("obj_names")
    batch = default_collate(batch)

    def _remove_batch_func(batch):
        if isinstance(batch, dict):
            return {k: _remove_batch_func(batch[k]) for k in batch.keys()}
        if isinstance(batch, list):
            return [_remove_batch_func(b) for b in batch]
        if isinstance(batch, torch.Tensor):
            assert (
                batch.shape[0] == 1
            ), f"The batchsize should be 1, but here is {batch.shape[0]}"
            return batch[0]

    batch = _remove_batch_func(batch)
    batch["obj_names"] = obj_names
    return batch


def variable_length_collate(batch):
    NO_PADDING_KEYS = ["lengths", "static_extra_ind"]
    no_padding_vars = DictwithEmpty([])
    batch_data = DictwithEmpty([])
    for b in batch:
        for k in b.keys():
            if k in NO_PADDING_KEYS:
                no_padding_vars[k].append(torch.tensor(b[k]))
            elif isinstance(b[k], (str, int, float)):
                batch_data["meta_" + k].append(b[k])
            else:
                batch_data[k].append(to_tensor(b[k]))

    for k in no_padding_vars.keys():
        no_padding_vars[k] = torch.stack(no_padding_vars[k])
    mask = lengths_to_mask(
        no_padding_vars["lengths"], no_padding_vars["lengths"].device
    )
    STATISTICS = ["mean", "std"]
    for k in batch_data.keys():
        flag = False
        for S in STATISTICS:
            if S in k:
                flag = True
        if "meta" in k:
            pass
        elif not flag:
            batch_data[k] = pad_sequence(
                batch_data[k], batch_first=True, padding_value=0.0
            )
        else:
            batch_data[k] = default_collate(batch_data[k])
    for k in no_padding_vars.keys():
        batch_data[k] = no_padding_vars[k]
    batch_data["mask"] = mask
    return batch_data
