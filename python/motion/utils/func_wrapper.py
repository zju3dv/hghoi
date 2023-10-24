from functools import wraps
import copy
import torch
import time
import datetime

from motion.utils.basics import DictwithEmpty


# ========================================================================================
# ===========================Wrap functions and chagne data ==============================
# ================================Dangerous behaviors=====================================
# ========================================================================================


def sequence_process(func):
    # For B, L, * data, process data in L one by one
    @wraps(func)
    def wrapped_func(self, data):
        outputs = DictwithEmpty([])
        for k in data.keys():
            if "mean" in k or "std" in k or "meta" in k:
                continue
            L = data[k].shape[1]
            break
        for i in range(L):
            data_per = {}
            for k, v in data.items():
                if "mean" in k or "std" in k or "meta" in k:
                    # Do not modify statistics
                    data_per[k] = data[k]
                else:
                    data_per[k] = v[:, i]
            output_per = func(data_per)
            for k, v in output_per.items():
                outputs[k].append(v)
        for k in outputs.keys():
            if outputs[k][0] is not None:
                outputs[k] = sum(outputs[k]) / len(outputs[k])
            else:
                outputs[k] = None
        return outputs

    return wrapped_func


# ========================================================================================
# ====================================== Decorators ======================================
# ========================================================================================


def gpu_memory_clear(func):
    is_cuda = torch.cuda.is_available()

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if is_cuda:
            torch.cuda.empty_cache()
        output = func(*args, **kwargs)
        if is_cuda:
            torch.cuda.empty_cache()
        return output

    return wrapped_func


def call_func_monitor(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        print(f"Call func: '{func.__name__}'...")
        output = func(*args, **kwargs)
        print(f"Finish func: '{func.__name__}'")
        return output

    return wrapped_func


def time_monitor(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time = time.time()
        output = func(*args, **kwargs)
        used_time = time.time() - start_time
        used_time = str(datetime.timedelta(seconds=int(used_time)))
        print(f"func: '{func.__name__}' costs {used_time}")
        return output

    return wrapped_func


def recursive_dict_wrap(func):
    """
    Recursively do func for inputs
        Directly operate on list or value, recursively on dict
    Return the func(inputs)
    """

    @wraps(func)
    def wrapped_func(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = wrapped_func(v, *args, **kwargs)
        else:
            inputs = func(inputs, *args, **kwargs)
        return inputs

    return wrapped_func


def recursive_wrap(func):
    """
    Recursively do func for inputs
    Return the func(inputs)
    """

    @wraps(func)
    def wrapped_func(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = wrapped_func(v, *args, **kwargs)
        elif isinstance(inputs, list):
            for i in range(len(inputs)):
                inputs[i] = wrapped_func(inputs[i], *args, **kwargs)
        elif isinstance(inputs, tuple):
            outputs = []
            for i in range(len(inputs)):
                outputs.append(wrapped_func(inputs[i], *args, **kwargs))
            inputs = outputs
        else:
            inputs = func(inputs, *args, **kwargs)
        return inputs

    return wrapped_func


def recursive_do_wrap(func):
    """
    Recursively do func for inputs
    Do not return inputs
    """

    @wraps(func)
    def wrapped_func(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            nrows = len(inputs.keys())
            ind = 1  # For subplot array
            for k, v in inputs.items():
                # For visualization, we give different label as key
                label = kwargs.get("label")
                new_kwargs = copy.deepcopy(kwargs)
                if label is not None:
                    new_kwargs["label"] += "_" + k
                else:
                    new_kwargs["label"] = k

                # For visualization, we set subplot array
                subplot_array = kwargs.get("subplot_array")
                if subplot_array is not None:
                    subplot_array = (nrows, 1, ind)
                    new_kwargs["subplot_array"] = subplot_array
                wrapped_func(v, *args, **new_kwargs)
                ind += 1
        elif isinstance(inputs, list):
            for i in range(len(inputs)):
                # For visualization, we give different label as ind
                label = kwargs.get("label")
                new_kwargs = copy.deepcopy(kwargs)
                if label is not None:
                    new_kwargs["label"] += "_" + str(i)
                else:
                    new_kwargs["label"] = str(i)

                # For visualization, we set subplot array
                subplot_array = kwargs.get("subplot_array")
                if subplot_array is not None:
                    subplot_array = (len(inputs), 1, i + 1)
                    new_kwargs["subplot_array"] = subplot_array
                wrapped_func(inputs[i], *args, **new_kwargs)
        else:
            func(inputs, *args, **kwargs)

    return wrapped_func


def recursive_everydo_wrap(func):
    """
    Recursively do func for inputs
        The args and kwargs are also arranged as inputs one by one
    Do not return func(inputs)
    """

    @wraps(func)
    def wrapped_func(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                label = kwargs.get("label")
                new_kwargs = copy.deepcopy(kwargs)
                if label is not None:
                    new_kwargs["label"] += "_" + k
                else:
                    new_kwargs["label"] = k
                new_args = [arg[k] for arg in args] if len(args) > 0 else []
                wrapped_func(v, *new_args, **new_kwargs)
        elif isinstance(inputs, list):
            for i in range(len(inputs)):
                label = kwargs.get("label")
                new_kwargs = copy.deepcopy(kwargs)
                if label is not None and len(label) > 0:
                    new_kwargs["label"] += "_" + str(i)
                else:
                    new_kwargs["label"] = str(i)
                new_args = [arg[i] for arg in args] if len(args) > 0 else []
                wrapped_func(inputs[i], *new_args, **new_kwargs)
        else:
            func(inputs, *args, **kwargs)

    return wrapped_func


def vis_wrap(vis_func, **vis_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            output = func(*args, **kwargs)
            vis_func(output, **vis_kwargs)
            return output

        return wrapped_func

    return decorator


def data_filter(func):
    @wraps(func)
    def wrapped_func(data, *args, **kwargs):
        contain_key = kwargs.get("contain_key")
        if contain_key is not None:
            new_data = {}
            kwargs.pop("contain_key")
            for k, v in data.items():
                for ck in contain_key:
                    if ck in k:
                        for i in range(len(v)):
                            new_data[f"{k}_{i}"] = v[i]
            data = new_data
        output = func(data, *args, **kwargs)
        return output

    return wrapped_func
