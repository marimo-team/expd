# expd: experiment design

_This package is powered by a collection of [marimo](https://github.com/marimo-team/marimo) notebooks._

`expd` is a command-line and GUI utility for running reproducible computational
experiments, with independent trials optionally executed in parallel.

**Quickstart.** Install `expd` with

```bash
pip install expd
```

1. Create a project:

```bash
expd init my_project && cd my_project
```

2. Fill out the template [marimo notebook](https://github.com/marimo-team/marimo)
   to implement the trial function:

```bash
marimo edit src/trial.py
```

3. Run an experiment by executing the trial function across varying inputs:

```bash
expd run
```

4. View past experiment results:

```bash
expd view
```

## CLI

**Commands.** The `expd` CLI comes with four commands. Each command is
accompanied with a marimo notebook.

| command        | description                                                         |
| -------------- | ------------------------------------------------------------------- |
| `expd init`    | initialize project structure, creating a marimo notebook `trial.py` |
| `expd explore` | interactively explore the trial function implemented in `trial.py`  |
| `expd run`     | design and run an experiment based on `trial.py`                    |
| `expd view`    | view past `expd run` results                                        |

## Projects

Running `expd init my_project` creates a directory `my_project` with the
following structure:

```bash
my_project/
├── outputs
└── src
    └── trial.py
```

### Defining a trial

The file `trial.py` is a marimo notebook that implements the core logic of your
experiments. It starts as a template, which you'll need to fill out.

**Types.** In the cell named `types`, you'll define the inputs to and outputs
of each trial as dataclasses.

Each input must have a type annotation. The supported scalar annotations
are:

- `int`
- `float`
- `bool`

Numerical (NumPy) arrays are also supported with the annotation

- `np.ndarray`

In addition to numerical values, arbitrary categorical values are supported.
These can be singleton or lists of arbitrary Python objects, selected
from a given finite set. For example, define a string representing a filename
with

```python
from dataclasses import dataclass
import expd as ex

@dataclass
class Inputs(ex.Model):
   file: ex.T[str, ex.Categorical(options=list_of_files)]
```

or a boolean vector with

```python
from dataclasses import dataclass
import expd as ex

@dataclass
class Inputs(ex.Model):
   vector: ex.T[bool, ex.Categorical(options=[True, False], length=3)]
```

Make sure to see the [examples](examples/) for common use cases.

**Trial function.** In the cell named `trial`, you'll define the trial
function which maps inputs to outputs.

Other `expd` commands will use these cells to run your experiment and visualize
its results.

> [!TIP]
>
> `expd` only uses the `types` and `trial` cells. While prototyping, you may use
> other cells to execute the trial function on different inputs.
> `expd` won't execute these extra cells when running an experiment.

### Running an experiment

An **experiment** is a collection of independent executions of the trial
function with varying inputs.

Create an experiment with:

```
expd run
```

This opens the experiment runner notebook in your browser.

**Strategies.**
The runner notebook will prompt you to choose a generation **strategy** for
each input. You can choose to grid numerical values or sample them
randomly, and you can choose grid or sample subsets of categorical values.

**Number of trials.**
The number of trials in your experiment is the size of your experiment
grid times the number of samples per stochastic strategy, both of which
are configurable in the notebook UI.

**Experiment key.**
Every experiment is given a unique key based on a UTC timestamp, and the
results of the experiment are saved to the `outputs/` directory.

**Parallelization.** You may optionally choose to parallelize the execution
of your experiment across trials. To do this, select a number of cores
greater than one.

> [!NOTE]
>
> Currently, `expd` exploits parallelization using Python threads. This can
> yield speed-ups over a single threaded execution if your trial function
> calls expensive subroutines that release the GIL. Libraries such as NumPy,
> Torch, and TensorFlow release the GIL for many numerical operations.
>
> In the future we may add an option to use multiple processes.

**Resuming experiments.** If an experiment was interrupted (for example,
because your computer crashed, or you shut the experiment down), `expd`
can resume it from the last completed trial. Just click the `resume experiment`
button and choose the key that you'd like to resume.

### Viewing experiment results

View the results of a previous experiment by running `expd view` in a
project directory.

Experiment results are stored in the `outputs/` folder, with each
experiment stored in a directory keyed by a timestamp. Each experiment
stores each trial's inputs and outputs, as well as metadata that
describing the experiment set-up and the hardware specifications of
the machine on which the experiment was run.

The viewer notebook launched by `expd view` presents this information to
you in a friendly UI, and provides instructions on how to recreate any
given trial from an experiment.

```bash
├── outputs
│   └── 2024-10-17-22-51-20
│       ├── experiment_meta.json
│       ├── requirements.txt
│       ├── src
│       │   ├── __pycache__
│       │   │   └── trial.cpython-310.pyc
│       │   └── trial.py
│       ├── strategies
│       │   ├── categorical-uniform
│       │   └── scalar-grid
│       └── trials
│           ├── trial-0
│           │   ├── inputs
│           │   │   ├── a.ckpt
│           │   │   └── b.ckpt
│           │   ├── outputs
│           │   │   ├── a.ckpt
│           │   │   ├── b.ckpt
│           │   │   └── dictionary.ckpt
│           │   └── trial-meta.json

```

## Examples

The [examples](examples/) directory contains several example projects that
show how to define various kinds of inputs, including common use cases
such as cycling through a list of filenames and saving an image in the trial
function.


