from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Callable, Literal, Type

import marimo as mo

import expd.dtypes.dtypes as dt
from expd.dtypes.dtyping import get_object_annotated_types
from expd.dtypes.models import Model
from expd.strategies.base import Strategy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from expd.dtypes.dtyping import AnnotatedType


def _type_to_element(
    annotated_type: AnnotatedType,
) -> tuple[mo.Html, Callable[[Any], Any]]:
    """Explorer utility that constructs a marimo element given a type hint"""
    bare_type = annotated_type.bare_type
    dtype = annotated_type.metadata

    if isinstance(dtype, dt.Bounded):
        start, stop = dtype.low, dtype.high
        value = 0 if start == float("-inf") and stop == float("inf") else start
        # ui.number() doesn't handle infinities well
        start = max(start, int(-1e9))
        stop = min(stop, int(1e9))
        element = mo.ui.number(
            start=start,
            stop=stop,
            value=value,
            step=1 if bare_type is int else None,
        )
        return element, lambda value: value
    if isinstance(dtype, dt.Array):
        if importlib.util.find_spec("numpy"):
            import numpy as np

            if bare_type is np.ndarray:
                params_dict: dict[str, Any] = {}
                if dtype.shape is None:
                    params_dict["shape"] = mo.ui.text(
                        placeholder="3,4", label="comma-separated shape"
                    )
                params_dict["values"] = mo.ui.dropdown(
                    {
                        "ones": lambda shape: np.ones(shape),
                        "zeros": lambda shape: np.zeros(shape),
                        "rand": lambda shape: np.random.randn(*shape),
                    }
                )
                params = mo.ui.dictionary(params_dict, label="params")

                def constructor(values: dict[str, Any]) -> NDArray[Any]:
                    shape = (
                        tuple(int(d) for d in values["shape"].split(","))
                        if "shape" in values
                        else dtype.shape
                    )
                    return values["values"](shape)  # type: ignore

                return params, constructor

        if importlib.util.find_spec("torch"):
            raise NotImplementedError
    if isinstance(dtype, dt.Categorical):
        options = {str(o): o for o in dtype.options}
        keys = list(options.keys())
        if dtype.length == 1:
            return (
                mo.ui.dropdown(
                    options, value=keys[0], allow_select_none=False
                ),
                lambda value: value,
            )
        else:
            return (
                mo.ui.array(
                    [
                        mo.ui.dropdown(
                            options, value=keys[0], allow_select_none=False
                        )
                        for _ in range(dtype.length)
                    ]
                ),
                lambda value: value,
            )

    raise NotImplementedError


def get_explorer_elements(
    inputs: object,
) -> tuple[dict[str, mo.Html], dict[str, Callable[[Any], Any]]]:
    """Explorer utility returns UI elements given inputs dataclass

    Inputs is a dataclass.
    """
    types = get_object_annotated_types(inputs)
    elements: dict[str, mo.Html] = {}
    constructors: dict[str, Callable[[Any], Any]] = {}
    for name, t in types.items():
        element, constructor = _type_to_element(t)
        elements[name] = element
        constructors[name] = constructor

    return elements, constructors


def dtype_to_strategy_dropdown(t: dt.DType) -> mo.ui.dropdown:
    options = []
    if isinstance(t, dt.SupportsGrid):
        options.append("grid")
    if isinstance(t, dt.SupportsGaussian):
        options.append("gaussian")
    if isinstance(t, dt.SupportsUniform):
        options.append("uniform")
    return mo.ui.dropdown(options)


def get_strategy_dropdowns(inputs: object) -> mo.ui.dictionary:
    types = get_object_annotated_types(inputs)
    return mo.ui.dictionary(
        {
            name: dtype_to_strategy_dropdown(annotated_type.metadata)
            for name, annotated_type in types.items()
        },
        label="strategies",
    )


def get_strategy_params(
    inputs: Type[Model],
    selected_strategies: dict[str, Literal["grid", "uniform", "gaussian"]],
    initial_params: dict[str, dict[str, Any]],
) -> tuple[
    mo.ui.dictionary, Callable[[dict[str, Any]], dict[str, Strategy[Any]]]
]:
    field_types = get_object_annotated_types(inputs)
    strategy_params: dict[str, Any] = {}
    strategy_uis: dict[str, dt.StrategyUI] = {}
    for name in field_types:
        dtype = field_types[name].metadata
        strategy = selected_strategies[name]
        params = initial_params.get(name, {}).get(strategy, {})
        if strategy is not None:
            stui = dtype.as_strategy_ui(
                strategy=strategy, initial_params=params
            )
            strategy_params[name] = stui.element
            strategy_uis[name] = stui
    param_dict = mo.ui.dictionary(strategy_params, label="strategy parameters")

    def strategy_instantiator(
        values: dict[str, Any]
    ) -> dict[str, Strategy[Any]]:
        return {
            name: strategy_ui.create_strategy(values[name])
            for name, strategy_ui in strategy_uis.items()
        }

    return param_dict, strategy_instantiator
