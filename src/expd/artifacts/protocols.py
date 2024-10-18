from __future__ import annotations

import abc
import io
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type

import expd.dependencies as deps

if TYPE_CHECKING:
    import pathlib

    import PIL.Image


# TODO: user-defined protocols
@dataclass(frozen=True)
class ArtifactProtocol(abc.ABC):
    name: str
    suffix: str

    @staticmethod
    @abc.abstractmethod
    def supports(artifact: Any) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def load(path: pathlib.Path) -> Any:
        pass


@dataclass(frozen=True)
class PickleProtocol(ArtifactProtocol):
    name: str = "pickle"
    suffix: str = ".pkl"

    @staticmethod
    def supports(artifact: Any) -> bool:
        del artifact
        return True

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        with path.open("wb") as f:
            pickle.dump(artifact, f)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        with path.open("rb") as f:
            return pickle.load(f)


@dataclass(frozen=True)
class NumpyProtocol(ArtifactProtocol):
    name: str = "numpy"
    suffix: str = ".npy"

    @staticmethod
    def supports(artifact: Any) -> bool:
        if not deps.numpy():
            return False

        import numpy as np

        return isinstance(artifact, np.ndarray)

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        import numpy as np

        with path.open("wb") as f:
            np.save(f, artifact)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        import numpy as np

        with path.open("rb") as f:
            return np.load(f)


@dataclass(frozen=True)
class ScipySparseProtocol(ArtifactProtocol):
    name: str = "scipy-sparse"
    suffix: str = ".npz"

    @staticmethod
    def supports(artifact: Any) -> bool:
        if not deps.scipy():
            return False

        import scipy.sparse as sp  # type: ignore

        return isinstance(artifact, sp.spmatrix)

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        import scipy.sparse as sp

        with path.open("wb") as f:
            sp.save_npz(f, artifact)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        import scipy.sparse as sp

        with path.open("rb") as f:
            return sp.load_npz(f)


@dataclass(frozen=True)
class TorchProtocol(ArtifactProtocol):
    name: str = "torch"
    suffix: str = ".pt"

    @staticmethod
    def supports(artifact: Any) -> bool:
        if not deps.torch():
            return False

        import torch

        return isinstance(artifact, torch.Tensor)

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        import torch

        with path.open("wb") as f:
            torch.save(artifact, f)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        import torch

        with path.open("rb") as f:
            return torch.load(f)


@dataclass(frozen=True)
class ImageProtocol(ArtifactProtocol):
    name: str = "image-protocol"
    suffix: str = ".png"

    @staticmethod
    def supports(artifact: Any) -> bool:
        if not deps.pil():
            return False

        return isinstance(artifact, (bytes, io.BytesIO))

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        import PIL.Image

        if isinstance(artifact, Image):
            artifact.image.save(
                path.with_suffix(path.suffix + artifact.filetype)
            )
            return
        if isinstance(artifact, PIL.Image.Image):
            artifact.save(path)
            return
        if isinstance(artifact, io.BytesIO):
            bio = artifact
        elif isinstance(artifact, bytes):
            bio = io.BytesIO(artifact)
        else:
            raise ValueError(
                "ImageProtocol requires bytes, BytesIO, or PIL.Image"
                f"but received {type(artifact)}"
            )
        PIL.Image.open(bio).save(path)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        import PIL.Image

        return Image(PIL.Image.open(path), filetype=path.suffix)


@dataclass(frozen=True)
class TextProtocol(ArtifactProtocol):
    name: str = "text"
    suffix: str = ".txt"

    @staticmethod
    def supports(artifact: Any) -> bool:
        return isinstance(artifact, str)

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        with open(path, "w") as f:
            f.write(artifact)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        with path.open("r") as f:
            return f.read()


@dataclass(frozen=True)
class BytesProtocol(ArtifactProtocol):
    name: str = "bytes"
    suffix: str = ".bytes"

    @staticmethod
    def supports(artifact: Any) -> bool:
        return isinstance(artifact, bytes)

    @staticmethod
    def save(path: pathlib.Path, artifact: Any) -> None:
        with open(path, "wb") as f:
            f.write(artifact)

    @staticmethod
    def load(path: pathlib.Path) -> Any:
        with path.open("rb") as f:
            return f.read()


NAME_TO_PROTOCOL = {
    PickleProtocol().name: PickleProtocol(),
    NumpyProtocol().name: NumpyProtocol(),
    ScipySparseProtocol().name: ScipySparseProtocol(),
    TorchProtocol().name: TorchProtocol(),
    ImageProtocol().name: ImageProtocol(),
    TextProtocol().name: TextProtocol(),
    BytesProtocol().name: BytesProtocol(),
}


class Image:
    def __init__(self, image: PIL.Image.Image, filetype: str = ".png") -> None:
        self.image = image
        self.filetype = filetype

    @staticmethod
    def from_bytes(data: bytes) -> Image:
        import PIL.Image

        bio = io.BytesIO(data)
        return Image(PIL.Image.open(bio))

    @staticmethod
    def from_bytes_io(bio: io.BytesIO) -> Image:
        import PIL.Image

        return Image(PIL.Image.open(bio))

    @staticmethod
    def from_figure(figure: Any) -> Image:
        """Create an image from a matplotlib figure."""
        buffer = io.BytesIO()
        figure.savefig(buffer)
        buffer.seek(0)
        return Image.from_bytes_io(buffer)

    def _repr_png_(self) -> bytes | None:
        return self.image._repr_png_()


def infer_protocol(t: Type[Any]) -> ArtifactProtocol:
    if deps.numpy():
        import numpy as np

        if issubclass(t, np.ndarray):
            return NumpyProtocol()
    if deps.torch():
        import torch

        if issubclass(t, torch.Tensor):
            return TorchProtocol()
    if deps.scipy():
        import scipy.sparse as sp

        if issubclass(t, sp.spmatrix):
            return ScipySparseProtocol()

    if issubclass(t, Image):
        return ImageProtocol()
    if issubclass(t, str):
        return TextProtocol()
    elif issubclass(t, bytes):
        return BytesProtocol()
    return PickleProtocol()


def from_name(name: str) -> ArtifactProtocol:
    if name in NAME_TO_PROTOCOL:
        return NAME_TO_PROTOCOL[name]
    # TODO: Look for user-defined protocols
    raise ValueError("Unknown protocol %s", name)


@dataclass
class ArtifactProtocolContainer:
    name: str

    def __post_init__(self) -> None:
        self.protocol = from_name(self.name)
