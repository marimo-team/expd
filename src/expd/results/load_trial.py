import importlib.util
from typing import Callable, Type, cast

from expd.dtypes.dtyping import set_local_ns
from expd.dtypes.models import Model


def load_models(filename: str) -> tuple[Type[Model], Type[Model]]:
    spec = importlib.util.spec_from_file_location("marimo_app", filename)
    if spec is None:
        raise RuntimeError("Failed to load module spec")

    trial_notebook = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load module spec's loader")
    spec.loader.exec_module(trial_notebook)

    _, typedefs = trial_notebook.types.run()
    # populates the local namespace with names defined by the
    # types cell, so that we can get type hints of the Inputs and Outputs
    # models
    set_local_ns(typedefs)
    return typedefs["Inputs"], typedefs["Outputs"]


def load_trial_function(filename: str) -> Callable[[Model], Model]:
    spec = importlib.util.spec_from_file_location("marimo_app", filename)
    if spec is None:
        raise RuntimeError("Failed to load module spec")

    trial_notebook = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load module spec's loader")
    spec.loader.exec_module(trial_notebook)
    _, defs = trial_notebook.trial.run()
    return cast(Callable[[Model], Model], defs["trial"])
