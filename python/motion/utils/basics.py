import copy


class CallHelper:
    """
    Enable class with the ability to call with 'attr' + '_' + k
    """

    def __getattr__(self, key: str):
        attr_k = key.split("_")
        if len(attr_k) != 2:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )
        attr, k = attr_k
        if hasattr(self, attr):
            attr = getattr(self, attr)
            if isinstance(attr, dict):
                if k in attr.keys():
                    return attr[k]
            elif isinstance(attr, list):
                if k.isdigit():
                    return attr[k]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


class DictwithEmpty(dict):
    """
    Enable dict with the ability to visit a key not contained and init a empty template
    """

    def __init__(self, empty_template, *args, **kwargs):
        self.empty_template = empty_template
        super().__init__(*args, **kwargs)

    def __getitem__(self, __k):
        if __k not in self.keys():
            self.__setitem__(__k, copy.deepcopy(self.empty_template))
        return super().__getitem__(__k)
