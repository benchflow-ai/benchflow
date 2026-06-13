"""Parametrized Sandbox-Protocol conformance for every backend.

Guards the ``@runtime_checkable`` ``Sandbox`` Protocol from drifting away from
the concrete backends. The Protocol is meant to be the literal common surface
of every ``BaseSandbox`` subclass; if a member is added to the Protocol that a
backend does not implement (the Cua-only ``read_file``/``write_file``/``host``/
``expose_ports`` regression), ``isinstance(instance, Sandbox)`` flips to
``False`` and these tests fail.

Instances are built with ``cls.__new__(cls)`` — no ``__init__``, so no Docker
daemon, Daytona/Modal cloud, or Cua desktop is touched. ``runtime_checkable``
only inspects attribute/method presence on the class, which ``__new__`` already
exposes, so this is a pure structural check.
"""

from __future__ import annotations

import pytest

from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox.protocol import Sandbox

# Each backend module imports its provider SDK lazily (inside methods), so the
# classes themselves import without the optional extras installed.
_BACKENDS = [
    ("benchflow.sandbox.docker", "DockerSandbox"),
    ("benchflow.sandbox.daytona", "DaytonaSandbox"),
    ("benchflow.sandbox.modal_impl", "ModalSandbox"),
    ("benchflow.sandbox.cua", "CuaSandbox"),
    ("benchflow.sandbox.macos_ios_simulator", "MacosIosSimulatorSandbox"),
]


def _load_backend(module_path: str, class_name: str) -> type[BaseSandbox]:
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@pytest.mark.parametrize(("module_path", "class_name"), _BACKENDS)
def test_backend_is_a_base_sandbox(module_path: str, class_name: str) -> None:
    cls = _load_backend(module_path, class_name)
    assert issubclass(cls, BaseSandbox)


@pytest.mark.parametrize(("module_path", "class_name"), _BACKENDS)
def test_backend_satisfies_sandbox_protocol(module_path: str, class_name: str) -> None:
    cls = _load_backend(module_path, class_name)
    instance = cls.__new__(cls)  # no __init__: no daemon/cloud/desktop touched
    assert isinstance(instance, Sandbox), (
        f"{class_name} does not satisfy the Sandbox Protocol — a Protocol "
        "member was added that this backend does not implement, or a backend "
        "dropped a member the contract requires."
    )
