import marimo

__generated_with = "0.8.15"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# `expd explore`""")
    return


@app.cell(hide_code=True)
def __(mo, set_local_ns):
    import pathlib

    EXP_PRECONDITIONS_MET = False

    mo.stop(
        not pathlib.Path("src/trial.py").exists(),
        mo.md(
            """
                Failed to find `src/trial.py` notebook!

                `expd explore` must be run from the top-level of the experiment
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
    set_local_ns(typedefs)
    Inputs, Outputs = typedefs["Inputs"], typedefs["Outputs"]

    EXP_PRECONDITIONS_MET = True
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


@app.cell(hide_code=True)
def __(EXP_PRECONDITIONS_MET, Inputs, fields, mo):
    mo.stop(not EXP_PRECONDITIONS_MET)

    _inputs_table = mo.ui.table(
        [{"name": f.name, "type": f.type} for f in fields(Inputs)],
        selection=None,
    )
    mo.md(
        f"""
        ## Inputs

        {_inputs_table}
        """
    )
    return


@app.cell(hide_code=True)
def __(Inputs, get_explorer_elements, mo):
    args, constructors = get_explorer_elements(Inputs)
    args = mo.ui.dictionary(args).form(bordered=False)
    args
    return args, constructors


@app.cell(hide_code=True)
def __(Inputs, args, constructors, mo, trial_fn):
    outputs = None
    mo.stop(args.value is None, mo.md("Choose your arguments").callout())

    _constructed_args = {key: constructors[key](value) for key, value in args.value.items()}
    outputs = trial_fn(Inputs(**_constructed_args))
    return outputs,


@app.cell(hide_code=True)
def __(asdict, mo, outputs):
    mo.stop(outputs is None)
    mo.vstack([
        mo.md("## Outputs"),
        asdict(outputs)
    ])
    return


@app.cell(hide_code=True)
def __():
    from dataclasses import asdict, fields

    import marimo as mo

    from expd.dtypes.dtyping import set_local_ns
    from expd.dtypes.elements import get_explorer_elements
    return asdict, fields, get_explorer_elements, mo, set_local_ns


if __name__ == "__main__":
    app.run()
