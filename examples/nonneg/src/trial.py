import marimo

__generated_with = "0.9.9"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass

    import expd as ex

    @dataclass
    class Inputs(ex.Model):
        a: ex.NonNeg[int]
        b: ex.NonPos[float]

    @dataclass
    class Outputs(ex.Model):
        a_plus_b: float
    return Inputs, Outputs, dataclass, ex


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(
            a_plus_b=inputs.a + inputs.b
        )
    return (trial,)


@app.cell
def __(Inputs, trial):
    trial(Inputs(a=1, b=-1))
    return


if __name__ == "__main__":
    app.run()
