"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import random
import sys
from typing import TYPE_CHECKING, Type

import numpy as np
import toml
from ml_collections import ConfigDict

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def env_param(config: ConfigDict, key: str) -> Any:
    """Read a shared env parameter (``freq``/``control_mode``/``sensor_range``) from either schema.

    Multi-drone eval configs (the ``multi_sim.py`` schema) place these under a per-drone
    ``[[env.kwargs]]`` list; single-drone and RL-training configs place them flat under ``[env]``.
    The RL pipeline shares one policy across all drones and steps the whole env at a single rate, so
    when the list form is present it reads the shared value from slot 0. This lets one
    ``multi_levelN.toml`` feed both ``multi_sim.py`` (per-drone kwargs) and training (flat read).

    Args:
        config: The loaded race config.
        key: The parameter name, e.g. ``"freq"``, ``"control_mode"`` or ``"sensor_range"``.
    """
    kwargs = config.env.get("kwargs")
    if kwargs is not None:
        return kwargs[0][key]
    return config.env[key]


def strip_env_randomization(config: ConfigDict) -> ConfigDict:
    """Remove per-episode randomizations & disturbances in place, for deterministic runs.

    Backs the ``--no_randomization`` flag on ``train.py``/``render.py``. The scripted-PID opponent
    flies a fixed waypoint spline with little gate clearance, so it crashes into randomized gates;
    clean renders / opponent demos need the nominal track. Deletes the ``[env.randomizations]`` and
    ``[env.disturbances]`` blocks if present, so the env builders fall back to their ``None`` default.

    Args:
        config: The loaded race config to mutate.
    """
    for key in ("randomizations", "disturbances"):
        if key in config.env:
            del config.env[key]
    return config


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
