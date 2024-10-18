from __future__ import annotations

from typing import Any, Type

from expd.strategies.base import Strategy
from expd.strategies.gaussian import ArrayGaussian, ScalarGaussian
from expd.strategies.grid import ArrayGrid, CategoricalGrid, ScalarGrid
from expd.strategies.uniform import (
    ArrayUniform,
    CategoricalUniform,
    ScalarUniform,
)

_strategies: list[Type[Strategy[Any]]] = [
    ArrayGrid,
    ScalarGrid,
    CategoricalGrid,
    ArrayUniform,
    ScalarUniform,
    CategoricalUniform,
    ArrayGaussian,
    ScalarGaussian,
]


def strategy_from_kwargs(kwargs: Any) -> Strategy[Any]:
    name = kwargs.pop("name")
    for st in _strategies:
        if name == st.name:
            return st.load(**kwargs)
    raise RuntimeError(f"No strategy found matching {name}.")
