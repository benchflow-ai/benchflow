"""The public surface is fully annotated and its doctests run.

benchflow ships ``py.typed``, so editors and type checkers trust its
annotations; a public function with an unannotated parameter or return reads
as ``Any`` there. This walks every function and every public method and
property of every class in ``benchflow.__all__`` and requires annotations,
except the ``__init__(*args, **kwargs)`` that ``typing.Protocol`` generates.
It also runs the docstring examples of the SDK modules.
"""

from __future__ import annotations

import doctest
import inspect
import typing
from pathlib import Path

import pytest

import benchflow as bf


def _is_protocol_init(cls: type, attr: str) -> bool:
    return attr == "__init__" and bool(getattr(cls, "_is_protocol", False))


def _missing(fn: typing.Callable[..., typing.Any]) -> list[str]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    missing = [
        p.name
        for p in sig.parameters.values()
        if p.name not in ("self", "cls") and p.annotation is inspect.Parameter.empty
    ]
    if fn.__name__ != "__init__" and sig.return_annotation is inspect.Signature.empty:
        missing.append("return")
    return missing


def _public_callables() -> list[tuple[str, typing.Callable[..., typing.Any]]]:
    out = []
    for name in bf.__all__:
        obj = getattr(bf, name)
        if inspect.isfunction(obj):
            out.append((name, obj))
        elif inspect.isclass(obj) and obj.__module__.startswith("benchflow"):
            for attr, member in vars(obj).items():
                if attr.startswith("_") and attr != "__init__":
                    continue
                if _is_protocol_init(obj, attr):
                    continue
                fn = (
                    member.__func__
                    if isinstance(member, (classmethod, staticmethod))
                    else member.fget
                    if isinstance(member, property)
                    else member
                )
                if inspect.isfunction(fn):
                    out.append((f"{name}.{attr}", fn))
    return out


def test_py_typed_ships() -> None:
    assert (Path(bf.__file__).parent / "py.typed").is_file()


def test_public_surface_is_annotated() -> None:
    gaps = {q: m for q, fn in _public_callables() if (m := _missing(fn))}
    assert gaps == {}


@pytest.mark.parametrize(
    "module",
    [
        "benchflow.batch",
        "benchflow.models",
        "benchflow.branch_api",
        "benchflow.jobs",
        "benchflow.job_export",
    ],
    ids=lambda m: m,
)
def test_sdk_docstring_examples_run(module: str) -> None:
    import importlib

    result = doctest.testmod(importlib.import_module(module), verbose=False)
    assert result.failed == 0
    assert result.attempted > 0
