from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def prepare_runtime(
    *,
    config_path: Path | str | None = None,
    module_names: list[str] | tuple[str, ...] = (),
) -> tuple[object, dict[str, object]]:
    if config_path is not None:
        os.environ["GELAI_CONFIG"] = str(Path(config_path).expanduser().resolve())

    settings_module = importlib.import_module("config.settings")
    importlib.reload(settings_module)
    config_module = importlib.import_module("config")
    importlib.reload(config_module)

    reloaded: dict[str, object] = {}
    for module_name in module_names:
        if module_name in sys.modules:
            reloaded[module_name] = importlib.reload(sys.modules[module_name])
        else:
            reloaded[module_name] = importlib.import_module(module_name)
    return config_module, reloaded


def apply_config_attrs(target_globals: dict, config_module: object, attr_map: dict[str, str]) -> None:
    for target_name, source_name in attr_map.items():
        target_globals[target_name] = getattr(config_module, source_name)
