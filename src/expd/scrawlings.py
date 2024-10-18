import marimo

__generated_with = "0.8.11"
app = marimo.App()


@app.cell
def __():
    from expd import dtypes as dt
    return dt,


@app.cell
def __(dt):
    from dataclasses import dataclass


    @dataclass
    class Inputs:
        x: int
        y: dt.NonNeg(int)
        #z: dt.Array(float, shape=(x, y))
    return Inputs, dataclass


@app.cell
def __(mo):
    mo.md(
        r"""
        ```python
        from dataclasses import dataclass

        @dataclass
        class Inputs:
            x: int
            y: dt.NonNeg(int)
            z: dt.Array(float, shape=(8, 8), ge=0, lt=10)
            s: dt.Sparse(float, shape=(8, 8), density=0.1, ge=0, lt=10)
        ```

        OR

        ```python
        from dataclasses import dataclass

        @dataclass
        class Inputs:
            x: int
            y: dt.Constrained[int, dt.NonNeg]
            z: dt.Constrained[float, dt.Shape(8, 8), dt.Ge(0), dt.Le(10)]
        ```
        """
    )
    return


@app.cell
def __(dt):
    dt.NonNeg(int)
    return


@app.cell
def __(Inputs, dt):
    types = dt.get_types(Inputs)

    t = types["x"]
    t
    return t, types


@app.cell
def __(types):
    from typing import get_args

    get_args(types["y"])
    return get_args,


@app.cell
def __(dt, types):
    dt.convert_annotated_to_bare_types(types["y"])
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
