from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from expd.dtypes.models import Model
from expd.strategies.base import Strategy


@dataclass(frozen=True)
class ModelSchema:
    # column name to str representation of type
    columns: dict[str, str]

    @staticmethod
    def create(model: Model) -> ModelSchema:
        return ModelSchema({f.name: str(f.type) for f in fields(model)})

    def extract_values(self, model: Model) -> dict[str, Any]:
        return {name: getattr(model, name) for name in self.columns}


@dataclass
class Schema:
    inputs: ModelSchema
    outputs: ModelSchema
    # Strategies used to generate inputs
    strategies: list[Strategy[Any]]
    n_samples_per_stochastic_strategy: int
    # total number of trials: n_samples_per_stochastic_strategy x grid size
    n_trials: int

    @staticmethod
    def create(
        inputs: Model,
        outputs: Model,
        strategies: list[Strategy[Any]],
        n_samples_per_stochastic_strategy: int,
        n_trials: int,
    ) -> Schema:
        return Schema(
            inputs=ModelSchema.create(inputs),
            outputs=ModelSchema.create(outputs),
            strategies=strategies,
            n_samples_per_stochastic_strategy=n_samples_per_stochastic_strategy,
            n_trials=n_trials,
        )

    def __post_init__(self) -> None:
        if len(self.inputs.columns) != len(self.strategies):
            raise ValueError(
                "The number of input columns must match the number of "
                "strategies. Columns: %s Strategies :%s"
                % (self.inputs.columns, self.strategies)
            )
