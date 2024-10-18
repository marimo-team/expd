import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass

    import expd as ex

    @dataclass
    class Inputs(ex.Model):
        a: ex.T[list[bool], ex.Categorical([True, False], length=2)]

    @dataclass
    class Outputs(ex.Model):
        a: list[bool]
    return Inputs, Outputs, dataclass, ex


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(a=inputs.a)
    return (trial,)


@app.cell
def __(Inputs, trial):
    trial(Inputs(a=[True, True]))
    return


if __name__ == "__main__":
    app.run()
