"""The "Modal sets ... in every sandbox" line is a warning once per run (#1248).

Modal sets its thread counts in every sandbox, and a count the task's
Dockerfile leaves unset keeps Modal's value (a variable can't be unset
there). Few Dockerfiles set the MKL, OpenBLAS, BLIS or ONNX Runtime counts, so
the warning came for nearly every task, once per sandbox. It now comes once
per run for each set of names and reason, and at debug level after that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import modal_impl
from benchflow.sandbox.modal_impl import ModalSandbox


def _sandbox(dockerfile: Path) -> ModalSandbox:
    """A ModalSandbox whose task folder holds *dockerfile* (or not)."""
    sandbox = object.__new__(ModalSandbox)
    sandbox.task_env_config = SimpleNamespace(docker_image=None)
    sandbox.environment_dir = dockerfile.parent
    sandbox._persistent_env = {}
    return sandbox


@pytest.fixture(autouse=True)
def _fresh_run(monkeypatch):
    monkeypatch.setattr(modal_impl, "_KEPT_ENV_SAID", set())


def _lines(caplog) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if "in every sandbox" in r.getMessage()
    ]


def test_the_kept_thread_counts_are_a_warning_once_per_run(tmp_path, caplog):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nENV OMP_NUM_THREADS=1\n")
    caplog.set_level(logging.DEBUG, logger=modal_impl.logger.name)

    for _ in range(3):
        image = _sandbox(dockerfile)._read_image_env()
        assert image.kept, image

    lines = _lines(caplog)
    assert [level for level, _ in lines] == [
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ], lines
    assert "MKL_NUM_THREADS" in lines[0][1], lines
    assert "OMP_NUM_THREADS" not in lines[0][1].split(" in every")[0], lines


def test_another_reason_is_its_own_warning(tmp_path, caplog):
    """A task whose Dockerfile BenchFlow can't read is a different case: it
    gets its own warning."""
    readable = tmp_path / "Dockerfile"
    readable.write_text("FROM python:3.12-slim\n")
    caplog.set_level(logging.DEBUG, logger=modal_impl.logger.name)

    _sandbox(readable)._read_image_env()
    _sandbox(tmp_path / "missing" / "Dockerfile")._read_image_env()

    assert [level for level, _ in _lines(caplog)] == [
        logging.WARNING,
        logging.WARNING,
    ], _lines(caplog)
