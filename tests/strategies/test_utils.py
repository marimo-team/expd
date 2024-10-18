import pytest
from hypothesis import given, strategies as st

from expd.strategies.grid import CategoricalGrid, ScalarGrid
from expd.strategies.uniform import ScalarUniform
from expd.strategies.utils import compute_n_trials, strategy_iterator


@given(
    st.floats(min_value=0, max_value=2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=2, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=10),
)
def test_n_trials_single_grid(low: float, high: float, n_trials: int) -> None:
    low, high = sorted([low, high])

    s = ScalarGrid(
        low=low, high=high, n_trials=n_trials, logspace=False, integral=False
    )
    it = strategy_iterator([s])
    for _ in range(n_trials):
        next(it)

    with pytest.raises(StopIteration):
        next(it)


def test_n_trials_multiple_scalar_grids() -> None:
    grids = [
        ScalarGrid(low=0, high=8, n_trials=9),
        ScalarGrid(low=0, high=8, n_trials=5),
        ScalarGrid(low=0, high=8, n_trials=3),
    ]
    n = 9 * 5 * 3
    assert n == compute_n_trials(grids, 1)
    it = strategy_iterator(grids)
    for _ in range(n):
        next(it)

    with pytest.raises(StopIteration):
        next(it)


def test_n_trials_scalar_categorical_grid() -> None:
    grids = [
        ScalarGrid(low=0, high=8, n_trials=9),
        CategoricalGrid(
            objects=[True, False],
            with_replacement=True,
            length=2,
        ),
    ]
    n = 9 * 4
    assert n == compute_n_trials(grids, 1)

    it = strategy_iterator(grids)
    for _ in range(n):
        next(it)

    with pytest.raises(StopIteration):
        next(it)


def test_n_trials_scalar_categorical_grid_with_uniform() -> None:
    strats = [
        ScalarGrid(low=0, high=8, n_trials=9),
        CategoricalGrid(
            objects=[True, False],
            with_replacement=True,
            length=2,
        ),
        ScalarUniform(low=0, high=1),
    ]
    n = 9 * 4
    assert n == compute_n_trials(strats, 1)

    it = strategy_iterator(strats)
    for _ in range(n):
        next(it)

    with pytest.raises(StopIteration):
        next(it)


def test_n_trials_stochastic() -> None:
    strats = [
        ScalarUniform(low=0, high=1),
    ]
    assert compute_n_trials(strats, 1) == 1
    assert compute_n_trials(strats, 10) == 10


def test_strategy_iterator_scalar_uniform_is_unbounded() -> None:
    strats = [
        ScalarUniform(low=0, high=1),
    ]
    it = strategy_iterator(strats)
    # smoke test: do lots of iterations
    for _ in range(1000):
        v = next(it)
        assert len(v) == 1
        assert v[0] >= 0
        assert v[0] <= 1


def test_strategy_iterator_single_grid() -> None:
    low = -1
    high = 1
    s = ScalarGrid(low=low, high=high, n_trials=3)
    it = strategy_iterator([s])
    assert s.n == 3

    assert next(it) == (-1,)
    assert next(it) == (0,)
    assert next(it) == (1,)
    with pytest.raises(StopIteration):
        next(it)


def test_strategy_iterator_multiple_grids() -> None:
    s1 = ScalarGrid(low=0, high=1, n_trials=2)
    s2 = ScalarGrid(low=2, high=4, n_trials=3)
    it = strategy_iterator([s1, s2])
    assert s1.n == 2
    assert s2.n == 3

    values = set([v for v in it])
    assert values == set(
        [
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
        ]
    )
