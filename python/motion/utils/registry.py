import inspect
from collections import abc


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "TYPE" not in cfg:
        if default_args is None or "TYPE" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "TYPE", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args must be a dict or None, " f"but got {type(default_args)}"
        )

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("TYPE")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        self.build_func = build_from_cfg

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        if key not in self._module_dict.keys():
            raise KeyError(f"{key} has not been registered!")
        return self._module_dict[key]

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _register_module(self, module_class, module_name=None):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module_class

    def register_module(self, name=None, module=None):
        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence"
                f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name)
            return cls

        return _register
