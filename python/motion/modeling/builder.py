import torch

from motion.utils.registry import Registry


MODELS = Registry("models")


def build_model(cfg):
    model_cfg = cfg.MODEL
    model = MODELS.build(model_cfg)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif True:
        device = torch.device("cpu")
    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    model = model.to(device)
    return model, device
