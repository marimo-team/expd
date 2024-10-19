from __future__ import annotations

import itertools
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator, TypeVar, Union

from expd.artifacts.protocols import NumpyProtocol
from expd.strategies.base import Grid

if TYPE_CHECKING:
    import scipy.sparse as sp  # type: ignore
    import torch
    from numpy.typing import NDArray

    ArrayT = TypeVar("ArrayT", NDArray[Any], torch.Tensor, sp.spmatrix)


@dataclass
class ScalarGrid(Grid[Union[int, float]]):
    # inclusive start
    low: int | float
    # inclusive end
    high: int | float
    # number of trials
    n_trials: int
    # whether yielded values must be integers; ignored if logspace
    integral: bool = False
    # number of steps in the grid
    # whether to step through the space in logspace (base 10)
    logspace: bool = False

    name: str = "scalar-grid"

    def iterator(self) -> Iterator[int | float]:
        if self.logspace:
            high = math.log(self.high, 10)
            low = math.log(self.low, 10)
        else:
            high = self.high
            low = self.low

        # low + (trials - 1)*step = high
        step = (high - low) / (self.n_trials - 1) if self.n_trials > 1 else 0
        step = int(step) if self.integral and not self.logspace else step
        value = low
        for _ in range(self.n):
            if value > self.high:
                # value may end up exceeding self.high by a very small amount,
                # due to floating point error.
                yield self.high
            else:
                yield value if not self.logspace else 10**value
            value += step

    @property
    def n(self) -> int:
        return self.n_trials


@dataclass
class ArrayGrid(Grid["NDArray[Any]"]):
    shape: tuple[int, ...]
    low: NDArray[Any]
    high: NDArray[Any]
    n_trials: int
    logspace: bool

    name: str = "array-grid"

    # NB doesn't support integer grid
    def iterator(self) -> Iterator[NDArray[Any]]:
        import numpy as np

        gridded = (
            np.linspace(self.low, self.high, num=self.n)
            if not self.logspace
            else np.logspace(self.low, self.high, num=self.n, base=10)
        )
        for i in range(gridded.shape[0]):
            yield gridded[i]

    @property
    def n(self) -> int:
        return self.n_trials

    @classmethod
    def load(cls, **kwargs: Any) -> ArrayGrid:
        low_path = pathlib.Path(kwargs["low"])
        high_path = pathlib.Path(kwargs["high"])

        low = NumpyProtocol.load(low_path)
        high = NumpyProtocol.load(high_path)
        kwargs["low"] = low
        kwargs["high"] = high
        return ArrayGrid(**kwargs)

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
class CategoricalGrid(Grid[Any]):
    objects: list[object]
    length: int
    # if True, returns cartesian product of object with itself `length` times
    # if False, returns set of permtuations of length `length`
    with_replacement: bool
    name: str = "categorical-grid"

    def iterator(self) -> Iterator[Any]:
        if self.with_replacement:
            for elem in itertools.product(
                *[self.objects for _ in range(self.length)]
            ):
                yield elem if self.length > 1 else elem[0]
        else:
            for elem in itertools.permutations(self.objects, self.length):
                yield elem if self.length > 1 else elem[0]

    @property
    def n(self) -> int:
        # TODO probably off-by-1, check
        if self.with_replacement:
            return len(self.objects) ** self.length  # type: ignore
        else:
            return math.perm(len(self.objects), self.length)
