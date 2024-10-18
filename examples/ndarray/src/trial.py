import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass
    import numpy as np

    import expd as ex

    @dataclass
    class Inputs(ex.Model):
        a: np.ndarray
        b: np.ndarray

    @dataclass
    class Outputs(ex.Model):
        a_times_b: np.ndarray
    return Inputs, Outputs, dataclass, ex, np


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(
            a_times_b=inputs.a @ inputs.b,
        )
    return (trial,)


@app.cell
def __(Inputs, np, trial):
    trial(Inputs(a=np.random.randn(2,2), b=np.ones((2,2))))
    return


if __name__ == "__main__":
    app.run()
