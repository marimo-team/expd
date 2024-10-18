import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell
def types():
    # All your imports should be in this cell
    from dataclasses import dataclass
    import PIL

    import expd as ex

    @dataclass
    class Inputs(ex.Model):
        exponent: float

    @dataclass
    class Outputs(ex.Model):
        img: ex.Image
    return Inputs, Outputs, PIL, dataclass, ex


@app.cell
def trial(Inputs, Outputs, ex):
    def trial(inputs: Inputs) -> Outputs:
        import matplotlib.pyplot as plt
        import numpy as np
        
        x = np.linspace(0, 4, 100)
        # make sure to create a new figure
        fig, ax = plt.subplots()
        ax.plot(x, x ** inputs.exponent)
        return Outputs(
            img=ex.Image.from_figure(fig),
        )
    return (trial,)


@app.cell
def __(Inputs, trial):
    trial(Inputs(exponent=2))
    return


if __name__ == "__main__":
    app.run()
