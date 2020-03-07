from typing import Any, Dict


class Config:
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    def __len__(self):
        return len(self._config)

    def __getitem__(self, key: str):
        return self._config[key]

    def __getattr__(self, key: str):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config

    def __iter__(self):
        return iter(self._config)

    def __repr__(self):
        cls_name = self.__class__.__name__
        params = []
        for key in self:
            param = f"{key}=<{self[key].__class__.__qualname__}>"
            params.append(param)
        params = ", ".join(params)
        return f"{cls_name}({params})"

    def __str__(self):
        params = []
        for key in self:
            o = self[key]
            module = o.__class__.__module__
            if module is None or module == str.__class__.__module__:
                cls_name = o.__class__.__name__
            else:
                cls_name = module + "." + o.__class__.__name__
            param = f"  {key}: {cls_name}"
            params.append(param)
        params = "\n".join(params)
        cls_name = self.__class__.__name__
        return f"{cls_name}\n{params}"
