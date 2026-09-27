"""Sandbox protocol value types are immutable."""

from __future__ import annotations

import pytest

from benchflow.sandbox.protocol import ExecResult, ImageRef


def test_exec_result_is_frozen():
    r = ExecResult(return_code=1, stdout="", stderr="err")
    with pytest.raises(AttributeError):
        r.return_code = 2  # type: ignore[misc]


def test_image_ref_is_frozen():
    ref = ImageRef(tag="v1")
    with pytest.raises(AttributeError):
        ref.tag = "v2"  # type: ignore[misc]
