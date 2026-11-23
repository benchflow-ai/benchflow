"""BENCHFLOW_MODAL_RLIMITS leaves BenchFlow's read of the kit's record alone (#1248 review, N2).

The held prefix runs the image's prlimit, timeout and setpriv ahead of a root
command, and is refused on a kit route whose proven folders lack /usr/bin.
The kit-probe route (the kit's own shell, the kit's bin alone) is the read
that finds the kit: a refusal there reads as "no kit", and root's later steps
would run on the image's shell and PATH. That route gets no prefix.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tenacity")

from benchflow.sandbox._base import KIT_PROBE_ROUTE
from benchflow.sandbox.modal_impl import DOCKER_RLIMITS, ModalSandbox


def _held_sandbox() -> ModalSandbox:
    sandbox = object.__new__(ModalSandbox)
    sandbox._rlimits = dict(DOCKER_RLIMITS)
    return sandbox


def test_the_kit_probe_route_runs_without_the_prefix():
    argv = ["/opt/runtime/bin/sh", "-c", "read the record"]
    assert _held_sandbox().hold_argv(argv, KIT_PROBE_ROUTE) == argv
