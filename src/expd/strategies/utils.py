from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from typing import Any, Sequence, cast

import expd.dependencies as deps
from expd.strategies.base import Grid, Strategy

SEED = 243


def get_global_seed() -> int:
    return SEED


def set_seeds(seed: int) -> None:
    global SEED
    SEED = seed

    random.seed(seed)
    try:
        if deps.numpy():
            import numpy as np

            np.random.seed(seed)
        if deps.torch():
            import torch

            torch.manual_seed(seed)
    except Exception:
        # in case of version mismatch, API changes ...
        pass


def has_stochastic_strategies(strategies: Sequence[Strategy[Any]]) -> bool:
    return any(not isinstance(s, Grid) for s in strategies)


def compute_n_trials(
    strategies: Sequence[Strategy[Any]], n_samples_per_stochastic_strategy: int
) -> int:
    n_trials = 0 if not strategies else 1
    for s in strategies:
        if isinstance(s, Grid):
            if n_trials is None:
                n_trials = s.n
            else:
                n_trials *= s.n
    return (
        n_trials
        if not has_stochastic_strategies(strategies)
        else n_trials * n_samples_per_stochastic_strategy
    )


def strategy_iterator(
    strategies: Sequence[Strategy[Any]],
    n_samples_per_stochastic_strategy: int = 1,
) -> Iterator[tuple[Any, ...]]:
    _grids = [
        (i, s.iterator())
        for i, s in enumerate(strategies)
        if isinstance(s, Grid)
    ]
    _rest = [
        (i, s.iterator())
        for i, s in enumerate(strategies)
        if not isinstance(s, Grid)
    ]
    rest_order, rest = tuple(zip(*_rest)) if _rest else ([], [])
    grid_order, grids = tuple(zip(*_grids)) if _grids else ([], [])

    rest_order = cast(tuple[int], rest_order)
    grid_order = cast(tuple[int], grid_order)
    rest = cast(tuple[Iterator[Any]], rest)
    grids = cast(tuple[Iterator[Any]], grids)

    if grids:
        for grid_values in itertools.product(*grids):
            for _ in range(n_samples_per_stochastic_strategy):
                stochastic_values = tuple(next(s) for s in rest)
                # reorder values according to original order
                values = [None] * len(strategies)
                for i, v in zip(grid_order, grid_values):
                    values[i] = v
                for i, v in zip(rest_order, stochastic_values):
                    values[i] = v

                yield tuple(values)
    else:
        while True:
            yield tuple(next(s) for s in rest)
