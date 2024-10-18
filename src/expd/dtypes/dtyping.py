from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Type,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import expd.artifacts.protocols as protocols
from expd.artifacts.artifacts import Artifact
from expd.dtypes import dtypes as dt

LOCALNS = {}


def set_local_ns(localns: dict[str, Any]) -> None:
    global LOCALNS
    LOCALNS = localns


def convert_annotated_to_bare_types(t: Type[Any]) -> Type[Any]:
    # Return the first arg of an annotated type
    #
    # For a given `X[Y, Z, ...]`:
    # * `get_origin` returns `X`
    # * `get_args` return `(Y, Z, ...)`
    # For non-supported type objects, the return values are
    # `None` and `()` correspondingly.
    origin, args = get_origin(t), get_args(t)

    if origin is None:
        # No origin -> Not a generic/no arguments.
        return t

    if origin is Annotated:
        # Annotated -> Convert the first argument recursively.
        bare_type = get_args(t)[0]
        return convert_annotated_to_bare_types(bare_type)

    # Otherwise, it is a generic. Convert all arguments recursively.
    converted_args = [convert_annotated_to_bare_types(arg) for arg in args]
    return cast(Type[Any], origin[converted_args])


@dataclass(frozen=True)
class AnnotatedType:
    bare_type: Type[Any]
    metadata: dt.DType

    @staticmethod
    def from_type(t: Type[Any]) -> "AnnotatedType":
        bare_type = convert_annotated_to_bare_types(t)
        if hasattr(t, "__metadata__"):
            assert len(t.__metadata__) == 1
            metadata = t.__metadata__[0]
            assert isinstance(metadata, dt.DType)

            if isinstance(metadata, dt.Bounded) and bare_type is float:
                metadata = dt.Bounded(
                    low=metadata.low,
                    high=metadata.high,
                    _domain="float",
                )
            elif isinstance(metadata, dt.Bounded) and bare_type is int:
                metadata = dt.Bounded(
                    low=metadata.low,
                    high=metadata.high,
                    _domain="int",
                )

        else:
            metadata = dt.bare_type_to_dtype(t)
        return AnnotatedType(bare_type=bare_type, metadata=metadata)


def get_object_annotated_types(inputs: object) -> dict[str, AnnotatedType]:
    """dict(Field name => AnnotatedType)"""
    types = get_type_hints(
        inputs,
        localns={
            "T": dt.T,
            "Bounded": dt.Bounded,
            "NonNeg": dt.NonNeg,
            "NonPos": dt.NonPos,
            "Array": dt.Array,
            "Categorical": dt.Categorical,
            "Artifact": Artifact,
            "PickleProtocol": protocols.PickleProtocol,
            "NumpyProtocol": protocols.NumpyProtocol,
            "ScipySparseProtocol": protocols.ScipySparseProtocol,
            "TorchProtocol": protocols.TorchProtocol,
            "ImageProtocol": protocols.ImageProtocol,
            "BytesProtocol": protocols.BytesProtocol,
            **LOCALNS,
        },
        include_extras=True,
    )
    return {name: AnnotatedType.from_type(t) for name, t in types.items()}


def get_object_bare_types(inputs: object) -> dict[str, Type[Any]]:
    """dict(Field name => AnnotatedType)"""
    types = get_type_hints(
        inputs,
        localns={
            "T": dt.T,
            "Bounded": dt.Bounded,
            "NonNeg": dt.NonNeg,
            "NonPos": dt.NonPos,
            "Array": dt.Array,
            "Categorical": dt.Categorical,
            "Artifact": Artifact,
            "PickleProtocol": protocols.PickleProtocol,
            "NumpyProtocol": protocols.NumpyProtocol,
            "ScipySparseProtocol": protocols.ScipySparseProtocol,
            "TorchProtocol": protocols.TorchProtocol,
            "ImageProtocol": protocols.ImageProtocol,
            "BytesProtocol": protocols.BytesProtocol,
            **LOCALNS,
        },
        include_extras=True,
    )
    return {
        name: convert_annotated_to_bare_types(t) for name, t in types.items()
    }


def get_artifact_types(
    outputs: object,
) -> dict[str, AnnotatedType]:
    """dict(Field name => AnnotatedType) where annotated_type is an artifact"""
    types = get_object_annotated_types(outputs)
    return {
        name: at
        for name, at in types.items()
        if isinstance(at.metadata, Artifact)
    }
