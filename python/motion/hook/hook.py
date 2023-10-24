import os
import time
import datetime
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from motion.utils.log import AverageMeter
from motion.utils.utils import makedirs, flatten_dict, calculate_grad_norm
from .builder import HOOKS


"""
Priority:
Lowest number is Highest
"""


class HookBase:
    def before_train(self, engine):
        pass

    def after_train(self, engine):
        pass

    def before_epoch(self, engine):
        pass

    def after_epoch(self, engine):
        pass

    def before_step(self, engine):
        pass

    def after_step(self, engine):
        pass

    def before_forward(self, pipeline, data):
        return data

    def after_forward(self, pipeline, data, output):
        return output

    def before_backward(self, pipeline, loss_dict):
        return loss_dict

    def after_backward(self, pipeline, loss_dict):
        return loss_dict


@HOOKS.register_module()
class TensorBoardHook(HookBase):
    """
    Last Hook
    """

    def __init__(self, priority, cfg, interval=1):
        self.priority = priority
        self.interval = interval
        self.tb_writter = None
        self.variables = {}

    def write(self, info, iters_num, name=None, prefix="", suffix="val"):
        if isinstance(info, dict):
            for k, v in info.items():
                self.write(v, iters_num, name=k, prefix=prefix, suffix=suffix)
        else:
            if name not in self.variables:
                self.variables[name] = AverageMeter(name)
            if suffix == "val":
                self.variables[name].update(info)
            self.tb_writter.add_scalar(
                prefix + "/" + name, getattr(self.variables[name], suffix), iters_num
            )

    def _multiply_weight(self, info):
        for k in info.keys():
            if "loss" in k:
                k_w = k.replace("_loss", "_weight")
                if k_w in info.keys():
                    val = info[k] / (info[k_w] + 1e-9)
                    info[k] = val

    def before_train(self, engine):
        self.tb_writter = engine.tb_writter

    def after_step(self, engine):
        info = engine.step_infos
        iters_num = engine.iters + 1
        if iters_num % self.interval == 0:
            self._multiply_weight(info)
            self.write(info, iters_num, prefix="iter")

    def before_epoch(self, engine):
        self.variables = {}

    def after_epoch(self, engine, info=None):
        if info is None:
            info = engine.step_infos
        epoch_num = engine.epochs
        self._multiply_weight(info)
        self.write(info, epoch_num, prefix="epoch", suffix="avg")


@HOOKS.register_module()
class LogShowHook(HookBase):
    """
    LogShowHook should be last but earlier than TensorboardHook
    """

    def __init__(self, priority, cfg, interval=1):
        self.priority = priority
        self.interval = interval
        self.logger = None
        self.iter_time = None
        self.start_time = None

    def write(self, info, engine):
        optimizer = engine.optimizer
        iters_num = engine.iters + 1
        epoch_num = engine.epochs + 1
        total_epoch_num = engine.max_epochs
        iters_per_epoch = engine.iters_per_epoch
        iters_this_epoch = (
            iters_num % iters_per_epoch
            if iters_num % iters_per_epoch
            else iters_per_epoch
        )
        print_str = (
            f"epoch: [{epoch_num}/{total_epoch_num}], "
            f"iter: [{iters_this_epoch}/{iters_per_epoch}], "
        )
        if isinstance(optimizer, dict):
            for k, v in optimizer.items():
                print_str += f"{k}_lr: {v.param_groups[0]['lr']:.6g}, "
                print_str += f"{k}_wd: {v.param_groups[0]['weight_decay']:.6g}, "
        else:
            print_str += f"lr: {optimizer.param_groups[0]['lr']:.6g}, "
            print_str += f"wd: {optimizer.param_groups[0]['weight_decay']:.6g}, "
        used_time = time.time() - self.start_time
        eta_time = (
            used_time
            / (1.0 * (iters_num - engine.start_iters))
            * (iters_per_epoch * total_epoch_num - iters_num)
        )

        used_time = str(datetime.timedelta(seconds=int(used_time)))
        eta_time = str(datetime.timedelta(seconds=int(eta_time)))
        print_str += f"used_time: {used_time}, eta: {eta_time}, "

        used_iter_time = time.time() - self.last_finish_iter_time
        used_iter_time = str(datetime.timedelta(seconds=int(used_iter_time)))
        print_str += f"iter_time: {used_iter_time}, "

        per_data_time = self.start_step_time - self.last_finish_iter_time
        per_data_time = str(datetime.timedelta(seconds=int(per_data_time)))
        print_str += f"data_time: {per_data_time}, "

        network_time = self.end_step_time - self.start_step_time
        network_time = str(datetime.timedelta(seconds=int(network_time)))
        print_str += f"network_time: {network_time}, "

        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            print_str += f"mem: {max_mem:.0f};\n"

        loss_str = ""
        weights_str = ""
        info_str = ""
        for k, v in info.items():
            try:
                val = v.item()  # sometimes it is a tensor
            except AttributeError:
                val = v
            if "loss" in k:
                k_w = k.replace("_loss", "_weight")
                if k_w in info.keys():
                    if abs(info[k_w]) > 0:
                        val = val / info[k_w]
                loss_str += f"{k}: {val:.6g}, "
            elif "weight" in k:
                weights_str += f"{k}: {val:.6g}, "
            else:
                info_str += f"{k}: {val:.6g}, "
        if len(info_str) > 2:
            info_str = info_str[:-2]
            info_str += ";\n"
        if len(weights_str) > 2:
            weights_str = weights_str[:-2]
            weights_str += ";\n"
        print_str += info_str
        print_str += weights_str
        print_str += loss_str
        self.logger.info(print_str[:-2])

    def process_step_infos(self, infos):
        try:
            extra_info = infos.pop("info")
        except KeyError:
            return infos
        if isinstance(extra_info, dict):
            infos.update(flatten_dict({"info": extra_info}))
        elif extra_info is not None:
            infos["info"] = extra_info
        return infos

    def before_train(self, engine):
        self.logger = engine.logger
        self.start_time = time.time()
        self.last_finish_iter_time = time.time()

    def before_step(self, engine):
        self.start_step_time = time.time()

    def after_step(self, engine):
        self.end_step_time = time.time()
        engine.step_infos = self.process_step_infos(engine.step_infos)
        info = engine.step_infos
        iters_num = engine.iters + 1
        if iters_num % self.interval == 0:
            self.write(info, engine)
        # For tensorboard logging lr
        engine.step_infos["lr"] = engine.optimizer.param_groups[0]["lr"]
        engine.step_infos["wd"] = engine.optimizer.param_groups[0]["weight_decay"]
        self.last_finish_iter_time = time.time()


@HOOKS.register_module()
class SaveCkptHook(HookBase):
    def __init__(self, priority, cfg, interval=1):
        self.priority = priority
        self.interval = interval
        self.output_dir = os.path.join(cfg.output_dir, "ckpts")
        self.logger = None

    def _save_ckpt(self, save_obj, epoch, save_type, suffix=""):
        if isinstance(save_obj, list):
            for i in range(len(save_obj)):
                self._save_ckpt(
                    save_obj[i], epoch, save_type, suffix=suffix + "_" + str(i)
                )
        elif isinstance(save_obj, dict):
            for k, v in save_obj.items():
                self._save_ckpt(v, epoch, save_type, suffix=suffix + "_" + k)
        else:
            output_dir = os.path.join(self.output_dir, str(epoch))
            makedirs(output_dir)
            path = os.path.join(output_dir, save_type + suffix + ".pth")
            torch.save(save_obj.state_dict(), path)

    def save_ckpt(self, engine):
        model = engine.model
        optimizer = engine.optimizer
        scheduler = engine.scheduler
        epoch = engine.epochs
        self.logger.info(f"Save model at {self.output_dir}/{epoch}!")
        self._save_ckpt(model, epoch, "model")
        self._save_ckpt(optimizer, epoch, "optimizer")
        if scheduler is not None:
            self._save_ckpt(scheduler, epoch, "scheduler")

    def before_train(self, engine):
        self.logger = engine.logger
        self.save_ckpt(engine)

    def after_epoch(self, engine):
        epoch = engine.epochs
        if epoch % self.interval == 0:
            self.save_ckpt(engine)

    def after_train(self, engine):
        epoch = engine.epochs
        if epoch % self.interval != 0:
            self.save_ckpt(engine)


@HOOKS.register_module()
class ResumeHook(HookBase):
    def __init__(self, priority, cfg, resume_dir=None):
        self.priority = priority
        if resume_dir is None:
            self.resume_dir = os.path.join(cfg.output_dir, "ckpts")
        else:
            self.resume_dir = resume_dir
        self.logger = None

    def _load_ckpt(self, load_obj, epoch, load_type, suffix=""):
        if isinstance(load_obj, list):
            for i in range(len(load_obj)):
                self._load_ckpt(
                    load_obj[i], epoch, load_type, suffix=suffix + "_" + str(i)
                )
        elif isinstance(load_obj, dict):
            for k, v in load_obj.items():
                self._load_ckpt(v, epoch, load_type, suffix=suffix + "_" + k)
        else:
            resume_dir = os.path.join(self.resume_dir, str(epoch))
            path = os.path.join(resume_dir, load_type + suffix + ".pth")
            if not os.path.exists(path):
                self.logger.error(f"{path} does not exist!")
                return
            ckpt = torch.load(path)
            try:
                load_obj.load_state_dict(ckpt)
            except Exception as e:
                self.logger.warning(e)
                try:
                    load_obj.load_state_dict(ckpt, strict=False)
                except Exception as e:
                    self.logger.error(e)

    def resume(self, engine):
        model = engine.model
        optimizer = engine.optimizer
        scheduler = engine.scheduler
        epoch = self.check_file()
        if epoch is None:
            self.logger.warning("Do not find the ckpts to resume.")
            return
        self._load_ckpt(model, epoch, "model")
        self._load_ckpt(optimizer, epoch, "optimizer")
        if scheduler is not None:
            self._load_ckpt(scheduler, epoch, "scheduler")
        engine.epochs = epoch
        engine.start_iters = epoch * engine.iters_per_epoch
        engine.iters = epoch * engine.iters_per_epoch
        self.logger.info(
            f"Load the model from {os.path.join(self.resume_dir, str(epoch))}"
        )

    def check_file(self):
        if not os.path.exists(self.resume_dir):
            return None
        epochs = os.listdir(self.resume_dir)
        if len(epochs) == 0:
            return None
        epochs = [int(e) for e in epochs]
        max_epoch = max(epochs)
        return max_epoch

    def before_train(self, engine):
        self.logger = engine.logger
        self.resume(engine)


@HOOKS.register_module()
class EvalHook(HookBase):
    def __init__(
        self, priority, cfg, interval=1, test_before_train=False, split="valid"
    ):
        self.priority = priority
        self.interval = interval
        self.test_before_train = test_before_train
        self.split = split
        self.start_time = None
        self.logger = None
        self.tb_writter = None

    def write(self, info, engine):
        epoch = engine.epochs
        total_epoch_num = engine.max_epochs
        print_str = f"\n[{type(self).__name__}] "
        print_str += f"epoch: [{epoch}/{total_epoch_num}], "
        used_time = time.time() - self.start_time
        used_time = str(datetime.timedelta(seconds=int(used_time)))
        print_str += f"used_time: {used_time}\n    Results:\n"
        flatten_info = {}
        self.convert_errors_dict(flatten_info, info)
        for k, v in flatten_info.items():
            self.tb_writter.add_scalar("val/" + self.split + "_" + k, v, epoch)
            print_str += f"        {k:30s}:    {v:.6g}\n"
        self.logger.info(print_str)

    def convert_errors_dict(self, flatten_info, info, prefix=""):
        if isinstance(info, dict):
            for k, v in info.items():
                flatten_v = self.convert_errors_dict(
                    flatten_info, v, prefix=prefix + k + "_"
                )
                try:
                    flatten_info.update(flatten_v)
                except TypeError:
                    # only need leaf value
                    pass
        else:
            return {prefix[:-1]: info}

    def test(self, engine):
        self.start_time = time.time()
        info = engine.do_eval(self.split, tb_writter=self.tb_writter)
        self.write(info, engine)

    def before_train(self, engine):
        self.logger = engine.logger
        self.tb_writter = engine.tb_writter
        if self.test_before_train:
            self.test(engine)

    def after_epoch(self, engine):
        epoch = engine.epochs
        if epoch % self.interval == 0:
            self.test(engine)

    def after_train(self, engine):
        epoch = engine.epochs
        if epoch % self.interval != 0:
            self.test(engine)


@HOOKS.register_module()
class ScheduledSamplingHook(HookBase):
    def __init__(self, priority, cfg, milestones):
        self.priority = priority
        self.cfg = cfg
        self.logger = None
        self.milestones = milestones
        assert len(milestones) == 2, "Only support scheduling in one interval"
        self.interval = milestones[1] - milestones[0]
        self.P = 1
        self.Bernoulli = None
        self.prev_output = None

    def before_train(self, engine):
        self.logger = engine.logger

    def before_epoch(self, engine):
        epoch = engine.epochs
        if epoch < self.milestones[0]:
            self.P = 1
        elif self.milestones[0] <= epoch < self.milestones[1]:
            self.P = 1 - (epoch + 1 - self.milestones[0]) / self.interval
        else:
            self.P = 0
        self.Bernoulli = torch.distributions.bernoulli.Bernoulli(
            torch.tensor(1 - self.P, dtype=torch.float)
        )
        self.logger.debug(
            f"[{type(self).__name__}] epoch: {epoch + 1}, p_value: {self.P}"
        )

    def before_step(self, engine):
        self.prev_output = None

    def after_step(self, engine):
        engine.step_infos["p_value"] = self.P

    def before_forward(self, pipeline, data):
        if (
            self.prev_output is not None
            and self.P < 1
            and self.Bernoulli.sample().int() == 1
        ):
            data = pipeline.scheduled_sampling_processing(self.prev_output, data)
        return data

    def after_forward(self, pipeline, data, output):
        if self.P < 1:
            self.prev_output = pipeline.select_saved_output(
                data, output, self.prev_output
            )
        return output


@HOOKS.register_module()
class AnnealingHook(HookBase):
    def __init__(self, priority, cfg, annealing_pairs):
        self.priority = priority
        self.cfg = cfg
        self.logger = None
        self.loss_names = []
        self.annealing = []
        self.annealing_weights = []
        self.milestones = []
        self.intervals = []
        for loss, milestones in annealing_pairs:
            assert (
                len(milestones) == 2
            ), f"{loss} has more than 2 values for milestones, only support annealing in one interval"
            self.loss_names.append(loss)
            self.milestones.append(milestones)
            self.intervals.append(milestones[1] - milestones[0])

    def before_train(self, engine):
        for l_name in self.loss_names:
            if hasattr(engine.pipeline, "criterion_" + l_name.lower()):
                loss = getattr(engine.pipeline, "criterion_" + l_name.lower())
                self.annealing.append(loss)
                self.annealing_weights.append(loss.weight)
            else:
                engine.logger.info(
                    f"[{type(self).__name__}] Warning! {l_name} does not exist!"
                )
                self.annealing.append(None)
                self.annealing_weights.append(0.0)

    def before_epoch(self, engine):
        epoch = engine.epochs
        for name, loss, ori_w, milestone, interval in zip(
            self.loss_names,
            self.annealing,
            self.annealing_weights,
            self.milestones,
            self.intervals,
        ):
            self._anneal_loss(loss, ori_w, milestone, interval, epoch)
            engine.logger.info(
                f"[{type(self).__name__}] {name} loss weight = {loss.weight}"
            )

    def _anneal_loss(self, loss, ori_weight, milestone, interval, epoch):
        if loss is not None:
            if epoch < milestone[0]:
                weight = 0.0
            elif milestone[0] <= epoch < milestone[1]:
                weight = ori_weight * (epoch + 1 - milestone[0]) / interval
                if weight > ori_weight:
                    weight = ori_weight
            else:
                weight = ori_weight
            loss.weight = weight


@HOOKS.register_module()
class NSMSchedulerHook(HookBase):
    def __init__(self, priority, cfg, Te=10, Tmult=2):
        self.priority = priority
        self.cfg = cfg
        self.logger = None
        self.Te = Te
        self.Tmult = Tmult
        self.EpochNext = self.Te + 1
        self.T_cur = 0.0
        self.H_cur = 0.9

    def before_train(self, engine):
        self.lr = engine.optimizer.param_groups[0]["lr"]
        self.weight_decay = engine.optimizer.param_groups[0]["weight_decay"]
        self.nBatches = engine.iters_per_epoch
        self.wd = self.weightDecayNormalized()

    def weightDecayNormalized(self):
        return self.weight_decay / (np.power(self.nBatches * self.Te, 0.5))

    def adjust_optimizer(self, optimizer, epoch):
        self.T_cur = self.T_cur + 1.0 / (self.Te * self.nBatches)
        if self.T_cur >= self.H_cur:
            self.T_cur = self.H_cur
        if (self.T_cur >= self.H_cur) and (epoch == self.EpochNext):
            self.T_cur = 0
            self.Te = self.Te * self.Tmult
            self.EpochNext = self.EpochNext + self.Te
        yita = 0.5 * (1 + np.cos(np.pi * self.T_cur))
        lr = self.lr * yita
        wd = yita * self.wd
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[0]["weight_decay"] = wd

    def before_step(self, engine):
        self.adjust_optimizer(engine.optimizer, engine.epochs)


@HOOKS.register_module()
class ClipGradHook(HookBase):
    def __init__(self, priority, cfg, max_grad=None, norm_type=2.0):
        self.priority = priority
        self.cfg = cfg
        self.max_grad = max_grad
        self.norm_type = norm_type

    def after_backward(self, pipeline, loss_dict):
        if self.max_grad is not None:
            total_norm = clip_grad_norm_(
                pipeline.model.parameters(), self.max_grad, self.norm_type
            )
        else:
            total_norm = calculate_grad_norm(
                pipeline.model.parameters(), self.norm_type
            )
        loss_dict["gradnorm"] = total_norm
        return loss_dict
