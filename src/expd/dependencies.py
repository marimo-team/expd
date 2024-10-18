from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal


def numpy() -> bool:
    return bool(importlib.util.find_spec("numpy"))


def torch() -> bool:
    return bool(importlib.util.find_spec("torch"))


def scipy() -> bool:
    return bool(importlib.util.find_spec("scipy"))


def pil() -> bool:
    return bool(importlib.util.find_spec("PIL"))


def in_conda_env() -> bool:
    return os.getenv("CONDA_PREFIX") is not None


def in_virtual_env() -> bool:
    """Returns True if a venv/virtualenv is activated"""
    # https://stackoverflow.com/questions/1871549/how-to-determine-if-python-is-running-inside-a-virtualenv/40099080#40099080  # noqa: E501
    base_prefix = (
        getattr(sys, "base_prefix", None)
        or getattr(sys, "real_prefix", None)
        or sys.prefix
    )
    return sys.prefix != base_prefix


@dataclass(frozen=True)
class PackageRequirements:
    manager: Literal["conda", "uv", "pip"]
    requirements_text: str


def get_package_requirements() -> PackageRequirements:
    """Returns current package requirements."""
    if in_conda_env():
        p = subprocess.run(["conda", "list", "--export"], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError("conda freeze failed: %s", p.stderr)
        return PackageRequirements(
            manager="conda", requirements_text=p.stdout.decode()
        )
    elif in_virtual_env():
        p = subprocess.run(["pip", "freeze"], capture_output=True)
        if p.returncode == 0:
            return PackageRequirements(
                manager="pip", requirements_text=p.stdout.decode()
            )
        else:
            # last ditch attempt: try uv ...
            p = subprocess.run(["uv", "pip", "freeze"], capture_output=True)
            if p.returncode == 0:
                return PackageRequirements(
                    manager="uv", requirements_text=p.stdout.decode()
                )
    raise RuntimeError(
        "Failed to freeze requirements; only conda, pip, "
        "and uv are supported."
    )
