from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import marimo as mo

from expd.artifacts.protocols import infer_protocol
from expd.dtypes.dtyping import get_object_bare_types

if TYPE_CHECKING:
    import pathlib

PRIMITIVE_TYPES = [
    int,
    float,
    bool,
    str,
    bytes,
]


@dataclass
class Model:
    @staticmethod
    def checkpoint_filename(variable_name: str) -> str:
        return variable_name + ".ckpt"

    def save(self, directory: pathlib.Path) -> None:
        types = get_object_bare_types(self)
        for variable, bare_type in types.items():
            protocol = infer_protocol(bare_type)
            path = directory / Model.checkpoint_filename(variable)
            protocol.save(path, getattr(self, variable))

    @classmethod
    def load(cls, directory: pathlib.Path) -> Model:
        types = get_object_bare_types(cls)
        variable_values = {}
        for variable, bare_type in types.items():
            protocol = infer_protocol(bare_type)
            paths = directory.glob("*")
            found_path = None
            for path in paths:
                if path.name.startswith(Model.checkpoint_filename(variable)):
                    found_path = path
                    break
            assert found_path is not None
            variable_values[variable] = protocol.load(found_path)
        return cls(**variable_values)

    def as_display_dict(self, prefix: str = "") -> dict[str, Any]:
        display_dict = {}
        types = get_object_bare_types(self)
        for variable, bare_type in types.items():
            value = getattr(self, variable)
            if bare_type in PRIMITIVE_TYPES:
                display_dict[prefix + variable] = value
            else:
                stringified = str(value)
                if len(stringified) > 100:
                    display_dict[prefix + variable] = stringified[:100] + "..."
                else:
                    display_dict[prefix + variable] = stringified
        return display_dict

    def _mime_(self) -> tuple[str, str]:
        return mo.as_html(asdict(self))._mime_()
