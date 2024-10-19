from __future__ import annotations

import abc
import importlib.util
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Type,
    TypeVar,
    cast,
)

import marimo as mo

import expd.strategies as st
from expd.strategies import Gaussian, Grid, Uniform
from expd.strategies.base import Strategy
from expd.strategies.gaussian import ArrayGaussian
from expd.strategies.grid import ArrayGrid
from expd.strategies.uniform import ArrayUniform

X = TypeVar("X")

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Supported primitive types
#
#   int
#   float
#   bool
#   str
#
# in the UI, we'll map int/float primitives to bounded types, bool/str to
# categorical


T = Annotated


class SupportsGrid(abc.ABC):
    @abc.abstractmethod
    def grid(self, low: Any, high: Any, **kwargs: Any) -> Grid[Any]:
        pass


class SupportsUniform(abc.ABC):
    @abc.abstractmethod
    def uniform(self, low: Any, high: Any, **kwargs: Any) -> Uniform[Any]:
        pass


class SupportsGaussian(abc.ABC):
    @abc.abstractmethod
    def gaussian(
        self, mu: Any, sigma: Any, lb: Any, ub: Any, **kwargs: Any
    ) -> Gaussian[Any]:
        pass


@dataclass
class StrategyUI:
    # ui element for configuring the parameters of a strategy
    element: mo.Html
    # takes the value of UI element value and returns a strategy
    create_strategy: Callable[[Any], Strategy[Any]]


@dataclass(frozen=True)
class DType(abc.ABC):
    @abc.abstractmethod
    def as_strategy_ui(
        self,
        strategy: Literal["grid"] | Literal["uniform"] | Literal["gaussian"],
        initial_params: dict[str, Any],
    ) -> StrategyUI:
        pass


@dataclass(frozen=True)
class Bounded(DType, SupportsGrid, SupportsUniform, SupportsGaussian):
    low: int | float
    high: int | float
    _domain: Literal["int"] | Literal["float"] | None = None

    def grid(
        self, low: int | float, high: int | float, **kwargs: Any
    ) -> Grid[int | float]:
        return st.ScalarGrid(
            low=low,
            high=high,
            integral=kwargs["integral"],
            n_trials=kwargs["n_trials"],
            logspace=kwargs["logspace"],
        )

    def uniform(
        self, low: int | float, high: int | float, **kwargs: Any
    ) -> Uniform[float]:
        del kwargs
        return st.ScalarUniform(low=low, high=high)

    def gaussian(
        self,
        mu: float,
        sigma: float,
        lb: float,
        ub: float,
        **kwargs: Any,
    ) -> Gaussian[float]:
        del kwargs
        return st.ScalarGaussian(mu=mu, sigma=sigma, lb=lb, ub=ub)

    def as_strategy_ui(
        self,
        strategy: Literal["grid"] | Literal["uniform"] | Literal["gaussian"],
        initial_params: dict[str, Any],
    ) -> StrategyUI:
        params: dict[str, Any] = {}
        if strategy == "grid":
            need_lb = self.low == float("-inf")
            need_ub = self.high == float("inf")

            params["trials"] = mo.ui.number(
                start=1,
                stop=int(1e8),
                step=1,
                label="$n$ trials",
                value=initial_params.get("trials", None),
            )
            if self.low != 0:
                params["space"] = mo.ui.dropdown(
                    ["linspace", "logspace"],
                    value=initial_params.get("space", "linspace"),
                )
            if need_lb and need_ub:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 0), start=-1e9, stop=1e9
                )
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1), start=-1e9, stop=1e9
                )
            elif need_ub:
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", self.low + 1),
                    start=-1e9,
                    stop=1e9,
                    step=1,
                )
            elif need_lb:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", self.high - 1),
                    start=-1e9,
                    stop=1e9,
                    step=1,
                )

            def create_grid(v: dict[str, Any]) -> Grid[Any]:
                low = v.get("low", self.low)
                high = v.get("high", self.high)
                if "space" in v:
                    logspace = v["space"] == "logspace"
                else:
                    logspace = False

                if self._domain == "int" and v["trials"] > (high - low) + 1:
                    raise ValueError(f"Invalid number of trials {v['trials']}")

                if self._domain == "int":
                    low = int(low)
                    high = int(high)
                return self.grid(
                    low=low,
                    high=high,
                    integral=self._domain == "int",
                    n_trials=v["trials"],
                    logspace=logspace,
                )

            return StrategyUI(
                element=mo.ui.dictionary(params, label="grid"),
                create_strategy=create_grid,
            )
        elif strategy == "uniform":
            # uniform also needs bounds
            need_lb = self.low == float("-inf")
            need_ub = self.high == float("inf")

            if need_lb and need_ub:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 0), start=-1e9, stop=1e9
                )
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1), start=-1e9, stop=1e9
                )
            elif need_ub:
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", self.low + 1),
                    start=-1e9,
                    stop=1e9,
                    step=1,
                )
            elif need_lb:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", self.high - 1),
                    start=-1e9,
                    stop=1e9,
                    step=1,
                )

            def create_uniform(v: dict[str, Any]) -> Uniform[Any]:
                low = v.get("low", self.low)
                high = v.get("high", self.high)

                if self._domain == "float":
                    low = float(low)
                    high = float(high)
                else:
                    low = int(low)
                    high = int(high)
                return self.uniform(low=low, high=high)

            return StrategyUI(
                element=mo.ui.dictionary(params, label="uniform"),
                create_strategy=create_uniform,
            )
        elif strategy == "gaussian":
            return StrategyUI(
                element=mo.ui.dictionary(
                    {
                        "mu": mo.ui.number(
                            value=initial_params.get("mu", 0),
                            start=-1e7,
                            stop=1e7,
                        ),
                        "sigma": mo.ui.number(
                            value=initial_params.get("sigma", 1),
                            start=-1e7,
                            stop=1e7,
                        ),
                    },
                    label="params",
                ),
                create_strategy=lambda v: self.gaussian(
                    **v, lb=self.low, ub=self.high
                ),
            )
        raise NotImplementedError("Unknown strategy %s" % strategy)


NonNeg = T[X, Bounded(0, float("inf"))]
NonPos = T[X, Bounded(float("-inf"), 0)]


@dataclass(frozen=True)
class Array(DType, SupportsGrid, SupportsUniform, SupportsGaussian):
    """A dense array with a fixed shape.

    Uniform and Gaussian strategies sample entries elementwise.

    Usage:

    T[np.array, Array(domain=float, shape=(3, 4))]

    # elementwise-bounds
    T[np.array, Array(domain=float, shape=(3, 4), Bounded=(-1, 1)]

    # array bounds
    T[np.array, Array(domain=float, shape=(3, 4), Bounded=(np.zeros((3, 4)), np.ones((3, 4))]

    # integer entries
    T[np.array, Array(domain=int, shape=(3, 4))]
    """  # noqa: E501

    shape: tuple[int, ...] | None
    logspace: bool | None = None
    bounds: Bounded = field(
        default_factory=lambda: Bounded(float("-inf"), float("inf"))
    )
    domain: Type[int] | Type[float] = float

    def grid(
        self, low: NDArray[Any], high: NDArray[Any], **kwargs: Any
    ) -> ArrayGrid:
        return ArrayGrid(
            shape=kwargs["shape"],
            low=low,
            high=high,
            n_trials=kwargs["n_trials"],
            logspace=kwargs["logspace"],
        )

    def uniform(
        self, low: NDArray[Any], high: NDArray[Any], **kwargs: Any
    ) -> ArrayUniform:
        return ArrayUniform(shape=kwargs["shape"], low=low, high=high)

    def gaussian(
        self,
        mu: float,
        sigma: float,
        lb: NDArray[Any],
        ub: NDArray[Any],
        **kwargs: Any,
    ) -> ArrayGaussian:
        return ArrayGaussian(
            shape=kwargs["shape"], mu=mu, sigma=sigma, lb=lb, ub=ub
        )

    def as_strategy_ui(
        self,
        strategy: Literal["grid"] | Literal["uniform"] | Literal["gaussian"],
        initial_params: dict[str, Any],
    ) -> StrategyUI:
        import numpy as np

        params: dict[str, Any] = {}
        if strategy == "grid":
            # need n trials
            # if lb and ub are missing, ask for scalar bounds
            need_lb = self.bounds.low == float("-inf")
            need_ub = self.bounds.high == float("inf")
            need_space = self.logspace is None
            need_shape = self.shape is None

            params["trials"] = mo.ui.number(
                start=initial_params.get("trials", 1),
                stop=int(1e8),
                step=1,
                label="$n$ trials",
            )
            if need_space:
                params["space"] = mo.ui.dropdown(
                    ["linspace", "logspace"],
                    value=initial_params.get("space", "linspace"),
                )
            if need_lb and need_ub:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 0), start=-1e9, stop=1e9
                )
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1), start=-1e9, stop=1e9
                )
            elif need_ub:
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1), start=-1e9, stop=1e9
                )
            elif need_lb:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 1), start=-1e9, stop=1e9
                )

            if need_shape:
                params["shape"] = mo.ui.text(
                    value=initial_params.get("shape", ""),
                    placeholder="3,4",
                    label="comma-separated shape",
                )

            def create_grid(v: dict[str, Any]) -> Grid[Any]:
                low = v.get("low", self.bounds.low)
                high = v.get("high", self.bounds.high)
                if "space" in v:
                    logspace = v["space"] == "logspace"
                else:
                    logspace = self.logspace
                if "shape" in v:
                    shape_string = v["shape"]
                    try:
                        shape = tuple(
                            int(dim) for dim in shape_string.split(",")
                        )
                    except Exception as err:
                        raise ValueError(
                            "Invalid shape string: shape should be a "
                            "comma-separated value, such as: 3,4 "
                        ) from err
                else:
                    shape = cast(tuple[int, ...], self.shape)

                if self.domain is int:
                    low = int(low) if isinstance(low, float) else low
                    high = int(high) if isinstance(high, float) else high

                if not isinstance(low, np.ndarray):
                    low = np.ones(shape) * low
                if not isinstance(high, np.ndarray):
                    high = np.ones(shape) * high
                return self.grid(
                    low=low,
                    high=high,
                    n_trials=v["trials"],
                    shape=shape,
                    logspace=logspace,
                )

            return StrategyUI(
                element=mo.ui.dictionary(params, label="grid"),
                create_strategy=create_grid,
            )
        elif strategy == "uniform":
            # if lb and ub are missing, ask for scalar bounds
            need_lb = self.bounds.low == float("-inf")
            need_ub = self.bounds.high == float("inf")
            need_shape = self.shape is None
            if need_shape:
                params["shape"] = mo.ui.text(
                    value=initial_params.get("shape", ""),
                    placeholder="3,4",
                    label="comma-separated shape",
                )

            if need_lb and need_ub:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 0), start=-1e9, stop=1e9
                )
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1), start=-1e9, stop=1e9
                )
            elif need_ub:
                params["high"] = mo.ui.number(
                    value=initial_params.get("high", 1),
                    start=self.bounds.low,
                    stop=1e9,
                )
            elif need_lb:
                params["low"] = mo.ui.number(
                    value=initial_params.get("low", 1),
                    start=-1e9,
                    stop=self.bounds.high,
                )

            def create_uniform(v: dict[str, Any]) -> Uniform[Any]:
                low = v.get("low", self.bounds.low)
                high = v.get("high", self.bounds.high)
                if "shape" in v:
                    shape_string = v["shape"]
                    try:
                        shape = tuple(
                            int(dim.strip()) for dim in shape_string.split(",")
                        )
                    except Exception as err:
                        raise ValueError(
                            "Invalid shape string: shape should be "
                            "a comma-separated value, such as: 3,4 "
                        ) from err
                else:
                    shape = cast(tuple[int, ...], self.shape)

                if not isinstance(low, np.ndarray):
                    low = np.ones(shape) * low
                if not isinstance(high, np.ndarray):
                    high = np.ones(shape) * high

                if self.domain is int:
                    low = low.astype(np.int64)
                    high = low.astype(np.int64)

                return self.uniform(low=low, high=high, shape=shape)

            return StrategyUI(
                element=mo.ui.dictionary(params, label="uniform"),
                create_strategy=create_uniform,
            )
        elif strategy == "gaussian":
            low = self.bounds.low
            high = self.bounds.high
            if low is None:
                low = -np.inf
            if high is None:
                high = np.inf
            lb = np.asarray(low)
            ub = np.asarray(high)

            need_shape = self.shape is None

            params = {
                "mu": mo.ui.number(
                    value=initial_params.get("mu", 0), start=-1e7, stop=1e7
                ),
                "sigma": mo.ui.number(
                    value=initial_params.get("sigma", 1), start=-1e7, stop=1e7
                ),
            }
            if need_shape:
                params["shape"] = mo.ui.text(
                    value=initial_params.get("shape", ""),
                    placeholder="3,4",
                    label="comma-separated shape",
                )

            def create_gaussian(v: dict[str, Any]) -> Gaussian[Any]:
                if "shape" in v:
                    shape_string = v["shape"]
                    try:
                        shape = tuple(
                            int(dim) for dim in shape_string.split(",")
                        )
                    except Exception as err:
                        raise ValueError(
                            "Invalid shape string: shape should be a "
                            "comma-separated value, such as: 3,4 "
                        ) from err
                else:
                    shape = cast(tuple[int, ...], self.shape)
                return self.gaussian(
                    mu=v["mu"], sigma=v["sigma"], lb=lb, ub=ub, shape=shape
                )

            return StrategyUI(
                element=mo.ui.dictionary(params, label="params"),
                create_strategy=create_gaussian,
            )


@dataclass(frozen=True)
class Categorical(DType, SupportsGrid, SupportsUniform):
    """
    Usage:

    T[str, Categorical(options="ab", length=2)]
        -> "aa" "ab" "ba" "bb"

    T[list[bool], Categorical(options=[True, False], length=2)]
        -> [T, T], [T, F], [F, F], [F, T]
    """

    # Can be used to lists of objects, such as bool vectors.
    #
    # For example:
    #   options = [True, False]
    #   length = 2
    #   with_replacement = True
    #
    # If gridded, yields [T, T], [T, F], [F, F], [F, T].
    options: list[object]
    length: int = 1
    with_replacement: bool = True

    def grid(self, low: Any, high: Any, **kwargs: Any) -> Grid[Any]:
        del low, high, kwargs
        return st.CategoricalGrid(
            objects=self.options,
            length=self.length,
            with_replacement=self.with_replacement,
        )

    def uniform(self, low: int, high: int, **kwargs: Any) -> Uniform[Any]:
        del kwargs
        return st.CategoricalUniform(
            objects=self.options,
            length=self.length,
            low=low,
            high=high,
            with_replacement=self.with_replacement,
        )

    def as_strategy_ui(
        self,
        strategy: Literal["grid"] | Literal["uniform"] | Literal["gaussian"],
        initial_params: dict[str, Any],
    ) -> StrategyUI:
        del initial_params
        if strategy == "grid":
            return StrategyUI(
                element=mo.ui.dictionary({}),
                create_strategy=lambda _: self.grid(
                    low=0, high=len(self.options) - 1
                ),
            )
        elif strategy == "uniform":
            return StrategyUI(
                element=mo.ui.dictionary({}),
                create_strategy=lambda _: self.uniform(
                    low=0, high=len(self.options)
                ),
            )
        elif strategy == "gaussian":
            raise ValueError("Categorical does not support Gaussian.")
        raise NotImplementedError


def bare_type_to_dtype(t: Type[Any]) -> DType:
    if t is int:
        return Bounded(low=float("-inf"), high=float("inf"), _domain="int")
    elif t is float:
        return Bounded(low=float("-inf"), high=float("inf"), _domain="float")
    elif t is bool:
        return Categorical(
            options=[True, False], length=1, with_replacement=False
        )
    elif t is str:
        raise NotImplementedError
    elif importlib.util.find_spec("numpy"):
        import numpy as np

        if t is np.ndarray:
            return Array(domain=float, shape=None, logspace=None)
    raise ValueError("Unsupported type: ", t)
