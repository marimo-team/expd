from __future__ import annotations

import pathlib
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator, Union, cast

from expd.artifacts.protocols import NumpyProtocol
from expd.strategies.base import Uniform

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ScalarUniform(Uniform[Union[int, float]]):
    name: str = "scalar-uniform"

    def iterator(self) -> Iterator[int | float]:
        is_integer = isinstance(self.low, int) and isinstance(self.high, int)
        while True:
            yield (
                random.uniform(self.low, self.high)
                if not is_integer
                else random.randint(cast(int, self.low), cast(int, self.high))
            )


@dataclass
class ArrayUniform(Uniform["np.ndarray[Any, Any]"]):
    shape: tuple[int, ...]
    name: str = "array-uniform"

    def iterator(self) -> Iterator[np.ndarray[Any, Any]]:
        import numpy as np

        if isinstance(self.low, np.ndarray):
            is_integer = issubclass(
                self.low.dtype.type, np.integer
            ) and isinstance(self.high.dtype.type, np.integer)
        else:
            is_integer = isinstance(self.low, int) and isinstance(
                self.high, int
            )

        while True:
            yield (
                np.random.uniform(self.low, self.high, size=self.shape)
                if not is_integer
                else np.random.randint(self.low, self.high, size=self.shape)
            )

    @classmethod
    def load(cls, **kwargs: Any) -> ArrayUniform:
        low_path = pathlib.Path(kwargs["low"])
        high_path = pathlib.Path(kwargs["high"])

        low = NumpyProtocol.load(low_path)
        high = NumpyProtocol.load(high_path)
        kwargs["low"] = low
        kwargs["high"] = high
        return ArrayUniform(**kwargs)

    def serialize(self, directory: pathlib.Path) -> dict[str, Any]:
        dictified = asdict(self)
        lb_path = directory / "low.npy"
        ub_path = directory / "high.npy"
        NumpyProtocol.save(lb_path, self.low)
        NumpyProtocol.save(ub_path, self.high)
        dictified["low"] = str(lb_path)
        dictified["high"] = str(ub_path)
        return dictified


@dataclass
class CategoricalUniform(Uniform[Any]):
    objects: list[object]
    length: int
    with_replacement: bool

    name: str = "categorical-uniform"

    def iterator(self) -> Iterator[Any]:
        if self.with_replacement:
            while True:
                elem = random.choices(self.objects, k=self.length)
                yield elem if self.length > 1 else elem[0]
        else:
            while True:
                elem = random.choices(self.objects, k=self.length)
                yield elem if self.length > 1 else elem[0]
