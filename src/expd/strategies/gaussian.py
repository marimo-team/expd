from __future__ import annotations

import pathlib
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator

from expd.artifacts.protocols import NumpyProtocol
from expd.strategies.base import Gaussian

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ScalarGaussian(Gaussian[float]):
    name: str = "scalar-gaussian"

    def iterator(self) -> Iterator[float]:
        while True:
            yield min(max(random.gauss(self.mu, self.sigma), self.lb), self.ub)


@dataclass
class ArrayGaussian(Gaussian["NDArray[Any]"]):
    shape: tuple[int, ...]
    name: str = "array-gaussian"

    def iterator(self) -> Iterator["NDArray[Any]"]:
        import numpy as np

        while True:
            ret = np.random.randn(*self.shape) * self.sigma + self.mu
            if self.lb is not None:
                ret = np.maximum(ret, self.lb)
            if self.ub is not None:
                ret = np.minimum(ret, self.ub)
            yield ret

    @classmethod
    def load(cls, **kwargs: Any) -> ArrayGaussian:
        low_path = pathlib.Path(kwargs["lb"])
        high_path = pathlib.Path(kwargs["ub"])

        low = NumpyProtocol.load(low_path)
        high = NumpyProtocol.load(high_path)
        kwargs["lb"] = low
        kwargs["ub"] = high
        return ArrayGaussian(**kwargs)

    def serialize(self, directory: pathlib.Path) -> dict[str, Any]:
        dictified = asdict(self)

        lb_path = directory / "lb.npy"
        ub_path = directory / "ub.npy"
        NumpyProtocol.save(lb_path, self.lb)
        NumpyProtocol.save(ub_path, self.ub)
        dictified["lb"] = str(lb_path)
        dictified["ub"] = str(ub_path)
        return dictified
