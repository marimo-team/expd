import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass

    import expd as ex

    files = [f"file_{i}" for i in range(12)]

    @dataclass
    class Inputs(ex.Model):
        file: ex.T[str, ex.Categorical(files)]

    @dataclass
    class Outputs(ex.Model):
        string_length: int
    return Inputs, Outputs, dataclass, ex, files


@app.cell
def trial(Inputs, Outputs):
    def trial(inputs: Inputs) -> Outputs:
        return Outputs(
            string_length=len(inputs.file)
        )
    return (trial,)


@app.cell
def __(Inputs, files, trial):
    trial(Inputs(file=files[0]))
    return


if __name__ == "__main__":
    app.run()
