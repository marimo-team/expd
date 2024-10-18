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
        a: int
        b: float
        c: bool

    @dataclass
    class Outputs(ex.Model):
        a_plus_b: float
        not_c: bool
    return Inputs, Outputs, dataclass, ex


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(
            a_plus_b=inputs.a + inputs.b,
            not_c=not inputs.c
        )
    return (trial,)


@app.cell
def __(Inputs, trial):
    trial(Inputs(a=1, b=1.5, c=True))
    return


if __name__ == "__main__":
    app.run()
