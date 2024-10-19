from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import click


def _key_value_bullets(items: list[tuple[str, str]]) -> str:
    max_length = max(len(item[0]) for item in items)
    lines = []
    # "\b" tells click not to reformat our text
    for key, desc in items:
        lines.append("  " + key + " " * (max_length - len(key) + 6) + desc)
    return "\n".join(lines)


def notebook_source(name: str) -> str:
    # from expd.notebooks import <name>
    trial_template = getattr(
        __import__("expd.notebooks", fromlist=[name]), name
    )
    return inspect.getsource(trial_template)


def start_marimo(filepath: str, mode: Literal["edit", "run"]) -> None:
    if mode == "run":
        cmd = [
            "marimo",
            "run",
            "--include-code",
            filepath,
        ]
    else:
        cmd = [
            "marimo",
            "edit",
            filepath,
        ]

    envcopy = os.environ.copy()
    envcopy["PYTHONPATH"] = (
        os.path.join(os.getcwd(), "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )

    def handler(sig: int, frame: Any) -> None:
        del sig
        del frame
        # let marimo handle the interrupt
        pass

    signal.signal(signal.SIGINT, handler)
    process = subprocess.Popen(cmd, env=envcopy)
    process.wait()


def start_notebook(name: str, mode: Literal["edit", "run"]) -> None:
    source = notebook_source(name)
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(os.path.join(temp_dir.name, f"{name}.py"))
    path.write_text(source)
    start_marimo(str(path), mode=mode)


# A custom class to order commands in `help` in the order they're defined.
class OrderCommands(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        return list(self.commands)


@click.group(
    cls=OrderCommands,
    help="""expd: experiment design and execution.

    Start by running `expd init my_project`

    which will create a project directory called my_project.

    All other commands should be run from the root of the project directory.
    """,
)
def main() -> None:
    ...


@main.command(help="Create a project directory.")
@click.argument("name", required=True)
def init(name: str) -> None:
    project = Path(name)
    if Path("src").exists() and Path("outputs").exists():
        print(
            "Initialization failed: src/ and outputs/ already exist. "
            "You appear to be initializing a project from inside "
            "a project directory."
        )
        return

    try:
        project.mkdir(exist_ok=False)
    except FileExistsError:
        print(
            f"Initialization failed: a directory called {project} "
            "already exists."
        )
        return

    # create a src/ folder
    # create an outputs/ folder
    src = project / Path("src")
    outputs = project / Path("outputs")
    try:
        src.mkdir(exist_ok=False)
    except FileExistsError:
        print("Initialization failed: a directory called src/ already exists.")
        return

    try:
        outputs.mkdir(exist_ok=False)
    except FileExistsError:
        print(
            "Initialization failed: a directory called outputs/ already "
            "exists."
        )
        return

    trial = src / "trial.py"

    if trial.exists():
        sys.stderr.write(
            f"Trial notebook ({str(trial)}) already exists.\nHas this "
            "project already been initialized?\n"
        )
        return

    source = notebook_source("trial")
    with trial.open("w") as f:
        f.write(source)

    print("Project initialized!")
    print()
    print(f"Commands to run from {project}/:")
    print(
        "\n".join(
            _key_value_bullets(
                [
                    ("marimo edit src/trial.py", "edit the trial function"),
                    ("expd explore", "open the trial explorer"),
                    ("expd run", "run an experiment"),
                    ("expd view", "view experiment results"),
                ],
            ).split("\n")
        )
    )
    print()
    print(
        "Note: all commands should be run from the root of the "
        "project directory."
    )


@main.command(help="Interactively run the trial function in `src/trial.py`.")
def explore() -> None:
    # started in edit mode so users can make use of hot-reloading with
    # the trial notebook
    start_notebook("explorer", mode="edit")


@main.command(help="Run an experiment.")
def run() -> None:
    start_notebook("runner", mode="run")


@main.command(help="View experiment results.")
def view() -> None:
    start_notebook("viewer", mode="run")
