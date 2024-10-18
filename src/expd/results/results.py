from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Type

import marimo as mo

from expd.dtypes.models import Model
from expd.results.env import Env
from expd.results.load_trial import load_models, load_trial_function
from expd.results.schemas import Schema
from expd.strategies.base import Strategy
from expd.strategies.container import strategy_from_kwargs
from expd.utils.parse_dataclass import parse_raw


@dataclass
class ResultsIdentifier:
    # Goal: try to save/load the entire inputs/outputs objects
    #
    # Requires reading the trial notebook from disk, which requires having same
    # package environment as was used to run the experiment.
    #
    # {path}/key/trials/trial-xxx/
    # {path}/key/trials/trial-xxx/inputs/*.ckpt
    # {path}/key/trials/trial-xxx/outputs/*.ckpt
    key: str
    path: str

    def __post_init__(self) -> None:
        self.path = os.path.realpath(self.path)
        self.base_dir = Path(self.path) / self.key

        self.results_dir = self.base_dir / "trials"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.strategies_dir = self.base_dir / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        self.src_dir = self.base_dir / "src"
        self.src_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.base_dir / "experiment_meta.json"
        self.requirements_path = self.base_dir / "requirements.txt"

    def _trial_str(self, trial: int, n_trials: int) -> str:
        n_digits = len(str(n_trials))
        return str(trial).zfill(n_digits)

    def trial_dir(self, trial: int, n_trials: int) -> Path:
        trial_str = self._trial_str(trial, n_trials)
        _trial_dir = self.results_dir / f"trial-{trial_str}"
        _trial_dir.mkdir(exist_ok=True)
        return _trial_dir

    def trial_metadata_path(self, trial: int, n_trials: int) -> Path:
        _trial_dir = self.trial_dir(trial, n_trials)
        return _trial_dir / "trial-meta.json"

    def trial_inputs_dir(self, trial: int, n_trials: int) -> Path:
        _inputs_dir = self.trial_dir(trial, n_trials) / "inputs"
        _inputs_dir.mkdir(exist_ok=True)
        return _inputs_dir

    def trial_outputs_dir(self, trial: int, n_trials: int) -> Path:
        _outputs_dir = self.trial_dir(trial, n_trials) / "outputs"
        _outputs_dir.mkdir(exist_ok=True)
        return _outputs_dir

    def trial_dir_to_trial(self, trial_dir: Path) -> int:
        # trial-###, extract ###
        return int(trial_dir.stem.split("-")[-1])


@dataclass
class TrialResult:
    inputs: Model
    outputs: Model
    elapsed_time_s: float


@dataclass
class TrialsTable:
    rows: list[dict[str, Any]]
    elapsed_time_s: float


@dataclass
class Results:
    # Identifier for the table on storage medium
    identifier: ResultsIdentifier
    # Runtime environment used to produce the results
    env: Env
    # Results table schema
    schema: Schema
    # If frozen, rows can't be added
    frozen: bool

    def __post_init__(self) -> None:
        # trial -> input/output model
        self._trial_results: dict[int, TrialResult] = {}

        serialized_results = sorted(self.identifier.results_dir.glob("*"))
        if serialized_results:
            # For resumption, we resume at the highest trial number for which
            # all previous trials have completed, for simplicity.
            trials = sorted(
                [
                    self.identifier.trial_dir_to_trial(r)
                    for r in serialized_results
                ]
            )
            self.trial = trials[-1] + 1
            for i in range(len(trials) - 1):
                if trials[i + 1] != trials[i] + 1:
                    self.trial = trials[i]
                    break
        else:
            self.trial = 0

        if self.trial != 0 and not self.frozen:
            logging.info("Resuming experiment at trial number %d", self.trial)

        # outputs/<experiment_dir>/
        #   <metadata_path.json>
        #   <requirements.txt>
        #   src/
        metadata_path = self.identifier.metadata_path
        if not metadata_path.exists():
            with metadata_path.open("w") as f:
                json.dump(self._construct_experiment_metadata(), f)
            shutil.copytree(
                "src/", str(self.identifier.src_dir), dirs_exist_ok=True
            )
            with self.identifier.requirements_path.open("w") as f:
                f.write(self.env.package_requirements.requirements_text)

        trial_filepath = str(self.identifier.src_dir / "trial.py")
        self._inputs_model, self._outputs_model = load_models(trial_filepath)
        self._trial_function = load_trial_function(trial_filepath)

    def _check_metadata(self) -> None:
        metadata_path = self.identifier.metadata_path
        if not self.frozen:
            if metadata_path.exists():
                with metadata_path.open("r") as f:
                    constructed_meta = self._construct_experiment_metadata()
                    loaded = json.load(f)
                    assert constructed_meta == loaded, (
                        "Non-frozen results, but metadata does not match:"
                        f"constructed: {constructed_meta}"
                        f"loaded: {loaded}"
                    )
            else:
                with metadata_path.open("w") as f:
                    json.dump(self._construct_experiment_metadata(), f)
                shutil.copytree("src/", str(self.identifier.base_dir / "src"))

    @staticmethod
    def create(
        inputs: Any,
        outputs: Any,
        strategies: list[Strategy[Any]],
        n_samples_per_stochastic_strategy: int,
        n_trials: int,
    ) -> Results:
        env = Env.create()
        schema = Schema.create(
            inputs=inputs,
            outputs=outputs,
            strategies=strategies,
            n_samples_per_stochastic_strategy=n_samples_per_stochastic_strategy,
            n_trials=n_trials,
        )
        results_identifier = ResultsIdentifier(
            key=datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S"),
            path="outputs/",
        )
        return Results(
            identifier=results_identifier, env=env, schema=schema, frozen=False
        )

    @staticmethod
    def load(directory: str, frozen: bool = True) -> Results:
        """Load a results folder from disk."""
        directory_as_path = Path(directory)
        key = directory_as_path.name
        identifier = ResultsIdentifier(
            key=key, path=str(directory_as_path.parent)
        )
        with identifier.metadata_path.open("r") as f:
            args = json.load(f)

        # Fixup strategies, since we don't have specific type annotations
        # for them (they are only annotated as a base class Strategy, but
        # user picks subclasses of Strategy).
        strategies = [
            strategy_from_kwargs(kwargs)
            for kwargs in args["schema"]["strategies"]
        ]

        results = Results(
            identifier=identifier,
            env=parse_raw(args["env"], Env),
            schema=parse_raw(args["schema"], Schema),
            frozen=frozen,
        )
        results.schema.strategies = strategies
        results._check_metadata()
        return results

    def _construct_experiment_metadata(self) -> dict[str, Any]:
        # TODO save hash of src/ directory
        dictified = {
            "results_identifier": asdict(self.identifier),
            "env": asdict(self.env),
            "schema": asdict(self.schema),
        }

        # let strategies encode themselves, since they may contain non
        # json serializable data
        strategies = []
        for strategy in self.schema.strategies:
            strat_dir = self.identifier.strategies_dir / strategy.name
            strat_dir.mkdir(exist_ok=True)
            strategies.append(strategy.serialize(directory=strat_dir))
        dictified["schema"]["strategies"] = strategies
        return dictified

    @property
    def experiment_metadata(self) -> dict[str, Any]:
        if self.identifier.metadata_path.exists():
            with self.identifier.metadata_path.open("r") as f:
                return json.load(f)  # type: ignore
        else:
            return self._construct_experiment_metadata()

    def add_inputs_and_outputs(
        self, trial: int, inputs: Model, outputs: Model, elapsed_time_s: float
    ) -> None:
        self._trial_results[trial] = TrialResult(
            inputs=inputs, outputs=outputs, elapsed_time_s=elapsed_time_s
        )

    def serialize(self) -> None:
        if self.frozen:
            raise RuntimeError("Cannot serialize frozen results.")
        if not self._trial_results:
            logging.warning("Attempting to serialize, but no results")
            return

        for trial in self._trial_results:
            trial_result = self._trial_results[trial]
            trial_result.inputs.save(
                self.identifier.trial_inputs_dir(trial, self.schema.n_trials)
            )
            trial_result.outputs.save(
                self.identifier.trial_outputs_dir(trial, self.schema.n_trials)
            )

            trial_metadata = {
                "trial": trial,
                "elapsed_time_s": trial_result.elapsed_time_s,
            }
            with self.identifier.trial_metadata_path(
                trial, self.schema.n_trials
            ).open("w") as f:
                json.dump(trial_metadata, f)

        self._trial_results.clear()

    def _trials_table(
        self, inputs_model: Type[Model], outputs_model: Type[Model]
    ) -> TrialsTable:
        if hasattr(self, "_cached_trials_table"):
            return self._cached_trials_table  # type: ignore

        # TODO: can use partitioning in the future
        trial_dirs = sorted(self.identifier.results_dir.glob("trial*"))
        self._trial_start = self.identifier.trial_dir_to_trial(trial_dirs[0])
        self._trial_end = self.identifier.trial_dir_to_trial(trial_dirs[-1])
        trials = list(range(self._trial_start, self._trial_end))
        displayable_inputs = [
            inp.as_display_dict(prefix="input.")
            for inp in self.inputs(trials, model=inputs_model)
        ]
        displayable_outputs = [
            out.as_display_dict(prefix="output.")
            for out in self.outputs(trials, model=outputs_model)
        ]
        trial_metas = []
        for trial_dir in trial_dirs:
            with (trial_dir / "trial-meta.json").open() as f:
                trial_metas.append(json.load(f))
        rows = [
            {**meta, **inp, **out}
            for meta, inp, out in zip(
                trial_metas, displayable_inputs, displayable_outputs
            )
        ]
        elapsed_time_s = sum(row["elapsed_time_s"] for row in rows)
        self._cached_trials_table = TrialsTable(
            rows=rows, elapsed_time_s=elapsed_time_s
        )
        return self._cached_trials_table

    def table(
        self, inputs_model: Type[Model], outputs_model: Type[Model]
    ) -> mo.ui.table:
        """Materializes the full data in memory."""
        return mo.ui.table(
            self._trials_table(inputs_model, outputs_model).rows,
            format_mapping={"elapsed_time_s": "{:E}"},
        )

    def inputs(
        self, trials: Sequence[int], model: Type[Model] | None = None
    ) -> list[Model]:
        model = model if model is not None else self._inputs_model
        return [
            model.load(
                self.identifier.trial_inputs_dir(trial, self.schema.n_trials)
            )
            for trial in trials
        ]

    def outputs(
        self, trials: Sequence[int], model: Type[Model] | None = None
    ) -> list[Model]:
        model = model if model is not None else self._outputs_model
        return [
            model.load(
                self.identifier.trial_outputs_dir(trial, self.schema.n_trials)
            )
            for trial in trials
        ]

    def execute_trial(self, inputs: Model) -> Model:
        return self._trial_function(inputs)
