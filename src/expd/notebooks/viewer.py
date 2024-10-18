import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# `expd view`""")
    return


@app.cell
def __(mo):
    import pathlib

    EXP_PRECONDITIONS_MET = False

    mo.stop(
        not (
            pathlib.Path("outputs/").exists()
            and pathlib.Path("outputs/").is_dir()
        ),
        mo.md(
            """
                Failed to find `outputs/` directory!

                `expd view` must be run from the top-level of the experiment
                directory, which should contain the `outputs` directory created
                by `expd init`.

                Are you in the wrong directory?
                """
        ).callout(kind="danger"),
    )

    # TODO might need inputs from original src notebook, under experiment_dir/src/trial.py

    EXP_PRECONDITIONS_MET = True
    return EXP_PRECONDITIONS_MET, pathlib


@app.cell
def __():
    from dataclasses import asdict

    import marimo as mo

    return asdict, mo


@app.cell(hide_code=True)
def __(EXP_PRECONDITIONS_MET, mo):
    mo.stop(not EXP_PRECONDITIONS_MET)
    mo.md("""**Experiment selection.**""")
    return


@app.cell
def __(EXP_PRECONDITIONS_MET, mo, pathlib):
    mo.stop(not EXP_PRECONDITIONS_MET)

    experiment_directory_browser = mo.ui.dropdown(
        sorted([str(p) for p in pathlib.Path("outputs/").glob("*")])
    )
    experiment_directory_browser
    return (experiment_directory_browser,)


@app.cell
def __():
    from expd.results.load_trial import load_models
    from expd.results.results import Results

    return Results, load_models


@app.cell
def __(experiment_directory_browser, mo):
    experiment_directory = None
    mo.stop(not experiment_directory_browser.value)
    experiment_directory = experiment_directory_browser.value
    return (experiment_directory,)


@app.cell
def __(Results, experiment_directory, load_models):
    results = Results.load(experiment_directory)

    trial_notebook_path = results.identifier.base_dir / "src" / "trial.py"
    Inputs, Outputs = load_models(trial_notebook_path)
    return Inputs, Outputs, results, trial_notebook_path


@app.cell
def __(experiment_directory, mo):
    mo.stop(experiment_directory is None)
    mo.md("""## Metadata""")
    return


@app.cell
def __(asdict, mo, results):
    metadata = mo.ui.tabs(
        {
            "Environment": asdict(results.env),
            "Schema": asdict(results.schema),
        }
    )
    metadata
    return (metadata,)


@app.cell
def __(experiment_directory, mo):
    mo.stop(experiment_directory is None)
    mo.md(
        f"""## Trials

        Your experiment results are tabulated below. The table
        shows the full values of "primitive" values, such as floats,
        ints, and strings, and truncated string representations of
        more complex objects such as NumPy arrays and images.

        To view the actual Python `Inputs` and `Outputs` objects
        that you defined in `trial.py` for a sequence of trials, select one or
        more rows below.
        """
    )
    return


@app.cell
def __(Inputs, Outputs, mo, results):
    _elapsed_time_s = results._trials_table(Inputs, Outputs).elapsed_time_s
    table = results.table(inputs_model=Inputs, outputs_model=Outputs)
    mo.vstack(
        [mo.md(f"The experiment took **{_elapsed_time_s:E}** seconds."), table]
    )
    return (table,)


@app.cell
def __(experiment_directory, mo, os):
    _full_path = os.path.realpath(experiment_directory)

    mo.accordion(
        {
            "Tip: executing trials programmatically": mo.md(
                f"""
                You can construct these values programmatically, in another
                marimo notebook or Python file, with the following code:

                ```python
                from expd import Results

                results = Results.load("{_full_path}")
                # trials below is a list of trials for which to obtain inputs/outputs
                inputs = results.inputs(trials=[0, 1])
                outputs = results.outputs(trials=[0, 1])
                ```

                For this to work, you'll need to run in a package environment
                matching the one serialized at `{_full_path}/requirements.txt`.

                You can invoke the trial function on the inputs you obtained,
                to check if they match the outputs.

                ```python
                trial_outputs = results.execute_trial(inputs[0])
                ```
                ***
                """
            )
        }
    )
    return


@app.cell
def __(mo, table):
    trials = []
    mo.stop(not table.value)

    trials = [row["trial"] for row in table.value]
    mo.md("""## Full inputs and outputs""")
    return (trials,)


@app.cell
def __(Inputs, Outputs, mo, results, trials):
    mo.stop(not trials)

    {
        i: {"input": inp, "output": out}
        for i, (inp, out) in enumerate(
            zip(
                results.inputs(trials, model=Inputs),
                results.outputs(trials, model=Outputs),
            )
        )
    }
    return


@app.cell
def __():
    import os

    return (os,)


if __name__ == "__main__":
    app.run()
