import math

from hypothesis import assume, given, strategies as st

from expd.strategies.grid import CategoricalGrid, ScalarGrid


@given(
    st.floats(
        min_value=-1, max_value=2, allow_nan=False, allow_infinity=False
    ),
    st.floats(
        min_value=-1, max_value=2, allow_nan=False, allow_infinity=False
    ),
    st.floats(min_value=0, max_value=2, allow_nan=False, allow_infinity=False),
)
def test_scalar_grid(low: float, high: float, step: float) -> None:
    low, step, high = sorted([low, step, high])
    if step < 0.1:
        step = 0.1

    n_trials = int((high - low) / step) + 1
    s = ScalarGrid(
        low=low, high=high, n_trials=n_trials, integral=False, logspace=False
    )
    assert s.n == n_trials

    it = s.iterator()
    i = -1
    v = -1
    try:
        for i in range(s.n + 1):  # noqa: B007
            v = next(it)
    except StopIteration:
        assert i == s.n, f"iteration ended early: {i=} {step=} {s.n=} {v=}"
        assert (
            s.n == 0 or v >= low
        ), f"iteration yielded out of bounds value: {i=} {step=} {s.n=} {v=}"
        assert (
            v <= high
        ), f"iteration yielded out of bounds value: {i=} {step=} {s.n=} {v=}"


def test_logspace_scalar_grid():
    s = ScalarGrid(
        low=1, high=100, n_trials=3, logspace=True
    )
    assert s.n == 3

    it = s.iterator()
    assert next(it) == 1
    assert next(it) == 10
    assert next(it) == 100

@given(st.data())
def test_categorical_grid_n_no_replacement(data) -> None:
    objects = data.draw(st.lists(st.integers(), min_size=0, unique=True))
    length = data.draw(st.integers(min_value=0, max_value=len(objects)))
    assume(length < len(objects))
    s = CategoricalGrid(
        objects=objects,
        length=length,
        with_replacement=False,
    )

    assert s.n == math.perm(len(objects), length)


@given(st.data())
def test_categorical_grid_n_with_replacement(data) -> None:
    objects = data.draw(st.lists(st.integers(), min_size=0, unique=True))
    length = data.draw(st.integers(min_value=0, max_value=len(objects)))
    s = CategoricalGrid(
        objects=objects,
        length=length,
        with_replacement=True,
    )

    assert s.n == len(objects) ** length


def test_scalar_grid_values() -> None:
    low = -1
    high = 1
    s = ScalarGrid(
        low=low,
        high=high,
        n_trials=3,
        integral=True,
        logspace=False,
    )
    it = s.iterator()
    values = [v for v in it]
    assert values == [-1, 0, 1]

    low = -1
    high = 1
    s = ScalarGrid(
        low=low, high=high, integral=True, logspace=False, n_trials=2
    )
    it = s.iterator()
    values = [v for v in it]
    assert values == [-1, 1]

    low = -1
    high = 1
    s = ScalarGrid(
        low=low, high=high, integral=True, logspace=False, n_trials=1
    )
    it = s.iterator()
    values = [v for v in it]
    assert values == [-1]


def test_categorical_grid_values_without_replacement() -> None:
    s = CategoricalGrid(
        objects=[0, 1, 2],
        with_replacement=False,
        length=3,
    )
    it = s.iterator()
    values = set([v for v in it])
    assert values == set(
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    )
    assert s.n == len(values)


def test_categorical_grid_values_with_replacement() -> None:
    s = CategoricalGrid(
        objects=[True, False],
        with_replacement=True,
        length=2,
    )
    it = s.iterator()
    values = set([v for v in it])
    assert values == set(
        [(True, False), (True, True), (False, True), (False, False)]
    )
