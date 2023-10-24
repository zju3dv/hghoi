import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.modeling.operators.builder import OPERATORS

from motion.modeling.modules.builder import MODULES


@MODULES.register_module()
class KeyQuantizeLayer(nn.Module):
    def __init__(self, quant_dim_enc, quant_n, quant_net, quant_key_list, **kwargs):
        super().__init__()
        self.key_list = quant_key_list
        for k in self.key_list:
            setattr(
                self, f"quant_{k}", OPERATORS.get(quant_net)(quant_dim_enc, quant_n)
            )

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x (Dict(tensor)): Dict([B, C])

        Returns:
            dict:
                quant (tensor): [B, P * C]
                diff (tensor): [B, P * C]
                ind (int tensor): [B, P]
        """
        quants = {}
        diffs = []
        ind_xs = {}
        quantize_infos = {}
        for k in self.key_list:
            quant_layer = getattr(self, f"quant_{k}")
            if k not in x.keys():
                continue
            quant_x, diff_x, ind_x, quantize_info = quant_layer(x[k])
            quants[k] = quant_x
            diffs.append(diff_x)
            ind_xs[k] = ind_x
            quantize_infos[k] = quantize_info
        return {
            "quant": quants,
            "diff": torch.cat(diffs, dim=-1),
            "ind": ind_xs,
            "info": quantize_infos,
        }

    def decode_forward(self, x):
        """
        Args:
            x (int64 tensor): [*, P], P is part number

        Returns:
            dict:
                quant (tensor): [B, P * C]
        """
        quants = {}
        for k in self.key_list:
            quant_layer = getattr(self, f"quant_{k}")
            if k not in x.keys():
                continue
            quant = quant_layer.embed_code(x[k])
            quants[k] = quant
        return {"quant": quants}
