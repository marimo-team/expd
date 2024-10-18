import marimo

__generated_with = "0.8.14"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass

    import expd as ex

    @dataclass
    class Inputs(ex.Model):
        a: ex.T[float, ex.Bounded(low=0, high=2)]
        b: bool

    @dataclass
    class Outputs(ex.Model):
        a: int
        b: bool
        dictionary: dict[str, str]
    return Inputs, Outputs, ex


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(
            a=int(inputs.a),
            b=not inputs.b,
            dictionary={"hello": "world"},
        )
    return trial,


@app.cell
def __(Inputs, trial):
    trial(Inputs(a=1, b=True))
    return


if __name__ == "__main__":
    app.run()
