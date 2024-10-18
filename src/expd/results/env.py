from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Union

import psutil
import pynvml  # type: ignore

from expd.dependencies import PackageRequirements, get_package_requirements
from expd.strategies.utils import get_global_seed


@dataclass(frozen=True)
class MachineStats:
    cpu_count: int
    cpu_speed_mhz: float
    memory_bytes: int
    gpus: Union[list[str], None]

    @staticmethod
    def gpu_info() -> list[str] | None:
        try:
            pynvml.nvmlInit()
        except Exception:
            return None

        deviceCount = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            devices.append(pynvml.nvmlDeviceGetName(handle))
        return devices

    @staticmethod
    def create() -> MachineStats:
        return MachineStats(
            cpu_count=psutil.cpu_count(),
            cpu_speed_mhz=psutil.cpu_freq().max,
            memory_bytes=psutil.virtual_memory().total,
            gpus=MachineStats.gpu_info(),
        )


@dataclass(frozen=True)
class Env:
    python_version: str
    package_requirements: PackageRequirements
    os: str
    machine_stats: MachineStats
    global_seed: int

    @staticmethod
    def create() -> Env:
        return Env(
            python_version=platform.python_version(),
            os=platform.platform(),
            package_requirements=get_package_requirements(),
            machine_stats=MachineStats.create(),
            global_seed=get_global_seed(),
        )
