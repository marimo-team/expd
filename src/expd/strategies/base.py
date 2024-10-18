from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    import pathlib


@dataclass
class Strategy(Generic[T]):

    @property
    @abc.abstractmethod
    def name(cls) -> str:
        pass

    @abc.abstractmethod
    def iterator(self) -> Iterator[T]:
        pass

    @classmethod
    def load(cls, **kwargs: Any) -> Strategy[Any]:
        return cls(**kwargs)

    def serialize(self, directory: pathlib.Path) -> dict[str, Any]:
        del directory
        return asdict(self)


@dataclass
class Grid(Strategy[T]):
    @property
    @abc.abstractmethod
    def n(self) -> int:
        """The number of trials in the strategy"""
        pass


@dataclass
class Gaussian(Strategy[T]):
    mu: float
    sigma: float
    lb: Any
    ub: Any


@dataclass
class Uniform(Strategy[T]):
    low: T
    high: T
