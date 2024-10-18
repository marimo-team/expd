import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __(fields, mo, set_local_ns):
    import pathlib

    EXP_PRECONDITIONS_MET = False

    mo.stop(
        not pathlib.Path("src/trial.py").exists(),
        mo.md(
            """
                Failed to find `src/trial.py` notebook!

                `expd run` must be run from the top-level of the experiment
                directory, which should contain the `src` and `outputs` directories
                created by `expd init`.

                Are you in the wrong directory?
                """
        ).callout(kind="danger"),
    )

    from trial import trial as trial_cell, types

    _, defs = trial_cell.run()
    trial_fn = defs["trial"]

    _, typedefs = types.run()

    Inputs, Outputs = typedefs["Inputs"], typedefs["Outputs"]

    set_local_ns(typedefs)
    EXP_PRECONDITIONS_MET = True

    _inputs_table = mo.ui.table(
        [{"name": f.name, "type": f.type} for f in fields(Inputs)],
        selection=None,
    )
    mo.md(
        f"""
        # `expd run`

        ## Inputs

        {_inputs_table}
        """
    )
    return (
        EXP_PRECONDITIONS_MET,
        Inputs,
        Outputs,
        defs,
        pathlib,
        trial_cell,
        trial_fn,
        typedefs,
        types,
    )


@app.cell
def __(mo):
    get_experiment_state, set_experiment_state = mo.state(None)
    return get_experiment_state, set_experiment_state


@app.cell
def __(
    EXP_PRECONDITIONS_MET,
    mo,
    new_experiment,
    resume_experiment,
    set_experiment_state,
):
    mo.stop(not EXP_PRECONDITIONS_MET)

    def _on_new_experiment():
        new_experiment[0] = True
        resume_experiment[0] = False

    def _on_resume_experiment():
        new_experiment[0] = False
        resume_experiment[0] = True

    new_experiment_btn = mo.ui.button(
        label="New experiment", on_change=lambda _: set_experiment_state("new")
    )
    resume_experiment_btn = mo.ui.button(
        label="Resume experiment",
        on_change=lambda _: set_experiment_state("resume"),
    )

    mo.hstack(
        [new_experiment_btn, mo.md("_or_"), resume_experiment_btn],
        justify="start",
    )
    return new_experiment_btn, resume_experiment_btn


@app.cell
def __(get_experiment_state):
    experiment_state = get_experiment_state()
    return (experiment_state,)


@app.cell
def __(experiment_state, mo):
    experiment_file_browser = mo.ui.file_browser(
        initial_path="outputs/",
        label="_Choose the experiment folder_",
        multiple=False,
        selection_mode="directory",
        restrict_navigation=True,
    )

    mo.stop(experiment_state != "resume")
    experiment_file_browser
    return (experiment_file_browser,)


@app.cell
def __(experiment_file_browser):
    resumed_experiment_directory = (
        experiment_file_browser.value[0].id
        if experiment_file_browser.value
        else None
    )
    return (resumed_experiment_directory,)


@app.cell
def __(Inputs, experiment_state, get_strategy_dropdowns, mo):
    # If any item is gridded, then number of trials is pre-determined; could
    #   specify number of times through the grid
    # If all items are distributional, then need to specify number of trials

    # for each parameter -> need to choose strategy;
    # for each strategy selected, show params
    # if gridded, n_trials is locked
    strategy_dropdowns = get_strategy_dropdowns(Inputs)
    saved_params_container = [{}]
    mo.stop(experiment_state != "new")
    strategy_dropdowns
    return saved_params_container, strategy_dropdowns


@app.cell
def __(
    Inputs,
    get_strategy_params,
    saved_params_container,
    strategy_dropdowns,
):
    strategy_params, constructor = get_strategy_params(
        Inputs, strategy_dropdowns.value, saved_params_container[0]
    )
    strategy_params
    return constructor, strategy_params


@app.cell
def __(saved_params_container, strategy_dropdowns, strategy_params):
    _saved_params = {}
    for _name, _strat in strategy_dropdowns.value.items():
        if _strat is not None:
            _saved_params[_name] = {_strat: strategy_params.value[_name]}
    _saved_params
    saved_params_container[0] = _saved_params
    return


@app.cell
def __(constructor, strategy_params):
    constructed_strategies = constructor(strategy_params.value)
    return (constructed_strategies,)


@app.cell
def __(constructed_strategies, has_stochastic_strategies, mo):
    n_samples_element = mo.ui.number(
        start=1,
        stop=int(1e9),
        step=1,
        label="$n$ samples per stochastic strategy",
    )

    n_samples_element.right() if has_stochastic_strategies(
        constructed_strategies.values()
    ) else None
    return (n_samples_element,)


@app.cell
def __(mo, os):
    n_cores_element = mo.ui.number(start=1, stop=os.cpu_count(), step=1, label="number of trials to run in parallel")
    n_cores_element.right()
    return (n_cores_element,)


@app.cell
def __(compute_n_trials, constructed_strategies, mo, n_samples_element):
    n_trials = compute_n_trials(
        constructed_strategies.values(),
        n_samples_element.value if n_samples_element is not None else 1,
    )
    mo.md(f"Your experiment has {n_trials} trials").right()
    return (n_trials,)


@app.cell
def __(experiment_state, mo):
    seed = mo.ui.number(1, int(1e6), step=1, value=243, label="random seed")
    run_button = mo.ui.run_button(
        label=(
            "run experiment"
            if experiment_state == "new"
            else "resume experiment"
        )
    )
    mo.stop(experiment_state is None)
    (
        mo.hstack([seed, run_button], justify="end")
        if experiment_state == "new"
        else run_button.right()
    )
    return run_button, seed


@app.cell
def __():
    def trial_results_to_table_row(trial_index, elapsed_time_s, inputs, outputs):
        return {
            "trial": trial_index,
            "elapsed_time_s": float(elapsed_time_s),
            **inputs.as_display_dict(prefix="inputs."),
            **outputs.as_display_dict(prefix="outputs."),
        }
    return (trial_results_to_table_row,)


@app.cell
def __(mo, time, trial_results_to_table_row):
    def run_trial(
        trial_index,
        flat_inputs,
        results,
        Inputs,
        trial_fn,
        table_rows,
        fmt_mapping,
        progress_bar,
        results_lock,
        semaphore,
        ctx
    ):
        try:
            mo._runtime.context.types.initialize_context(ctx)
        except Exception:
            pass

        args = {
            name: value
            for name, value in zip(results.schema.inputs.columns, flat_inputs)
        }
        inputs = Inputs(**args)
        start = time.time()
        outputs = trial_fn(inputs)
        elapsed_time_s = time.time() - start
        with results_lock:
            table_rows.insert(
                0,
                trial_results_to_table_row(
                    trial_index, elapsed_time_s, inputs, outputs
                ),
            )
            # HACK
            progress_bar.closed = False
            progress_bar.update()
            results.add_inputs_and_outputs(
                trial_index, inputs, outputs, elapsed_time_s=elapsed_time_s
            )
            results.serialize()
            mo.output.replace_at_index(
                mo.ui.table(table_rows, format_mapping=fmt_mapping),
                # index 0 is the "running experiment ..."
                # index 1 is the progress bar
                2,
            )
        semaphore.release()
        return inputs, outputs, elapsed_time_s
    return (run_trial,)


@app.cell
def __(args, n_cores_element, threading):
    class MockThread:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

        def join(self):
            return

    Thread = threading.Thread

    if n_cores_element.value == 1:
        Thread = MockThread
    return MockThread, Thread


@app.cell
def __(
    Inputs,
    Outputs,
    Results,
    Thread,
    constructed_strategies,
    experiment_state,
    mo,
    n_cores_element,
    n_samples_element,
    n_trials,
    resumed_experiment_directory,
    run_button,
    run_trial,
    seed,
    set_seeds,
    strategy_iterator,
    threading,
    time,
    trial_fn,
):
    def run_experiment():
        table_rows = []
        results = None
        fmt_mapping = {"elapsed_time_s": "{:E}"}
        if not run_button.value:
            return None

        if experiment_state == "new":
            set_seeds(seed.value)
            strategies = constructed_strategies.values()
            results = Results.create(
                Inputs,
                Outputs,
                list(strategies),
                n_samples_element.value,
                n_trials,
            )
            mo.output.append(
                mo.md(
                    f"**Running new experiment** with key {results.identifier.key}"
                )
            )
        else:
            results = Results.load(resumed_experiment_directory, frozen=False)
            strategies = results.schema.strategies
            if results.trial == results.schema.n_trials:
                mo.output.append("The experiment is already complete.")
                return
            mo.output.append(
                mo.md(f"**Resuming experiment** at trial {results.trial}")
            )
            set_seeds(results.env.global_seed)

        it = strategy_iterator(
            strategies,
            n_samples_per_stochastic_strategy=results.schema.n_samples_per_stochastic_strategy,
        )
        i = 0
        while i < results.trial:
            next(it)
            i += 1

        start_s = time.time()
        threads = []
        semaphore = threading.Semaphore(n_cores_element.value)
        results_lock = threading.Lock()
        with mo.status.progress_bar(
            total=results.schema.n_trials - results.trial - 1
        ) as progress_bar:
            # Naive parallelization that should still go a long way because
            # scientific Python libraries typically release the GIL.
            for trial_index in range(results.trial, results.schema.n_trials):
                semaphore.acquire()
                threads.append(
                    Thread(
                        target=run_trial,
                        args=(
                            trial_index,
                            next(it),
                            results,
                            Inputs,
                            trial_fn,
                            table_rows,
                            fmt_mapping,
                            progress_bar,
                            results_lock,
                            semaphore,
                            mo._runtime.context.get_context()
                        ),
                    )
                )
                threads[-1].start()

        for thread in threads:
            thread.join()
        total_time_s = time.time() - start_s

        table = mo.ui.table(table_rows, format_mapping=fmt_mapping)
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Results"),
                    mo.md(f"**Experiment key**: {results.identifier.key}"),
                    mo.md(f"**Elapsed time**: {total_time_s:E} seconds"),
                    table,
                ]
            )
        )


    run_experiment()
    return (run_experiment,)


@app.cell
def __():
    import os
    import threading
    import time
    from dataclasses import asdict, dataclass, fields

    import marimo as mo

    from expd.dtypes.dtyping import set_local_ns
    from expd.dtypes.elements import (
        get_strategy_dropdowns,
        get_strategy_params,
    )
    from expd.results.results import Results
    from expd.strategies.utils import (
        compute_n_trials,
        has_stochastic_strategies,
        set_seeds,
        strategy_iterator,
    )
    return (
        Results,
        asdict,
        compute_n_trials,
        dataclass,
        fields,
        get_strategy_dropdowns,
        get_strategy_params,
        has_stochastic_strategies,
        mo,
        os,
        set_local_ns,
        set_seeds,
        strategy_iterator,
        threading,
        time,
    )


if __name__ == "__main__":
    app.run()
