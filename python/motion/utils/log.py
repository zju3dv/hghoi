import sys
import os
import logging
import torch
import tensorboardX
from termcolor import colored

from motion.utils.utils import makedirs


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.val = 0.0
        self.n = 0.0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()
        self.val = val
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, *args, **kwargs):
        super().__init__(fmt, datefmt, *args, **kwargs)
        self._debug, self._info, self._warn, self._error = [
            logging.Formatter(colored(fmt, c) + "%(message)s", datefmt=datefmt)
            for c in ["blue", "green", "yellow", "red"]
        ]

    def format(self, record):
        if record.name == "root":
            record.name = ""
        if len(record.name):
            record.name = " " + record.name

        if logging.INFO > record.levelno >= logging.DEBUG:
            return self._debug.format(record)
        elif logging.WARNING > record.levelno >= logging.INFO:
            return self._info.format(record)
        elif logging.ERROR > record.levelno >= logging.WARNING:
            return self._warn.format(record)
        else:
            return self._error.format(record)


def build_tb_writter(path):
    tb_writer_dir = os.path.join(path, "tensorboard")
    makedirs(tb_writer_dir)
    return tensorboardX.SummaryWriter(tb_writer_dir)


def build_logger(path, name="log"):
    exp_name = path.split("/")[-1]
    makedirs(path)
    logger = logging.getLogger(name)
    path = os.path.join(path, name + ".txt")
    basic_format = f"[EXP: {exp_name}][%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] "
    date_format = "%Y-%m-%d %H:%M:%S"
    fh = logging.FileHandler(path)
    fh_formatter = logging.Formatter(basic_format + "%(message)s", date_format)
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    sh_formatter = _ColorFormatter(basic_format, date_format)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(sh_formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger
