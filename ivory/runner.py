from typing import Any, Dict

import ivory


class Runner:
    @classmethod
    def create(cls, config, default: Dict[str, Any] = None):
        cfg = ivory.utils.parse(config, default)
        runner = cfg["runner"] if "runner" in cfg else cls()
        runner.cfg = cfg
        return runner
