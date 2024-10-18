from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from expd.artifacts.protocols import (
    ArtifactProtocol,
    from_name,
    infer_protocol,
)

if TYPE_CHECKING:
    from pathlib import Path

    from expd.dtypes.dtyping import AnnotatedType


@dataclass(frozen=True)
class Artifact:
    protocol: ArtifactProtocol | None = None


@dataclass
class ArtifactLocator:
    directory: Path
    name: str
    protocol: ArtifactProtocol

    def __post_init__(self) -> None:
        self._locator = self.directory / (
            self.name + "." + self.protocol.name + self.protocol.suffix
        )
        self._data: bytes | None = None
        self.suffix = self.protocol.suffix

    @staticmethod
    def from_path(path: Path) -> ArtifactLocator:
        directory = path.parent

        stem_parts = path.stem.split(".")
        variable_name = stem_parts[0]
        protocol_name = stem_parts[1]

        protocol = from_name(protocol_name)
        return ArtifactLocator(
            directory=directory, name=variable_name, protocol=protocol
        )

    @staticmethod
    def from_type(
        directory: Path, name: str, t: AnnotatedType
    ) -> ArtifactLocator:
        directory.mkdir(parents=True, exist_ok=True)
        artifact = t.metadata
        assert isinstance(artifact, Artifact)
        return ArtifactLocator(
            directory=directory,
            name=name,
            protocol=(
                artifact.protocol
                if artifact.protocol is not None
                else infer_protocol(t.bare_type)
            ),
        )

    def load(self) -> Any:
        return self.protocol.load(self._locator)

    def save(self, artifact: Any) -> None:
        return self.protocol.save(self._locator, artifact)
