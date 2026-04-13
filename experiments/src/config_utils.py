import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _expand_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(value) for value in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def load_yaml_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)
