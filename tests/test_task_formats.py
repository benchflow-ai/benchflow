"""Task formats (benchflow.task.formats): foreign task folders materialized into native packages on load."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.task import formats

NATIVE_TASK_MD = """---
schema_version: "1.3"
task:
  name: demo/{name}
agent:
  timeout_sec: 60
verifier:
  timeout_sec: 60
---

Say hello.
"""


def _write_native(pkg: Path, name: str) -> None:
    (pkg / "environment").mkdir(parents=True)
    (pkg / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (pkg / "verifier").mkdir()
    test = pkg / "verifier" / "test.sh"
    test.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
    test.chmod(0o755)
    (pkg / "task.md").write_text(NATIVE_TASK_MD.format(name=name))


class DemoFormat:
    """Claims folders holding a `demo.txt`; the native package is named after its content."""

    name = "demo"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def detect(self, task_dir: Path) -> bool:
        return (task_dir / "demo.txt").is_file()

    def materialize(self, task_dir: Path, out_root: Path) -> Path:
        self.calls.append(task_dir)
        name = (task_dir / "demo.txt").read_text().strip()
        pkg = out_root / name
        if not pkg.exists():
            _write_native(pkg, name)
        return pkg


@pytest.fixture
def demo(tmp_path, monkeypatch):
    fmt = DemoFormat()
    monkeypatch.setattr(formats, "_registered", [])
    monkeypatch.setattr(formats, "_entry_point_formats", [])
    monkeypatch.setenv(formats.CACHE_ENV, str(tmp_path / "cache"))
    formats.register_task_format(fmt)
    return fmt


def _demo_task(root: Path, dirname: str, name: str) -> Path:
    d = root / dirname
    d.mkdir(parents=True)
    (d / "demo.txt").write_text(name)
    return d


def test_native_and_non_task_dirs_pass_through(tmp_path, demo):
    native = tmp_path / "native"
    _write_native(native, "native")
    assert formats.detect_task_format(native) is None
    assert formats.materialize_task_dir(native) == native
    assert formats.materialize_task_dir(tmp_path / "missing") == tmp_path / "missing"


def test_claimed_dir_materializes_under_format_cache(tmp_path, demo):
    src = _demo_task(tmp_path / "src", "a", "alpha")
    out = formats.materialize_task_dir(src)
    assert out == tmp_path / "cache" / "demo" / "alpha"
    assert (out / "task.md").is_file()
    assert demo.calls == [src.resolve()]


def test_materialized_package_still_claimed_is_an_error(tmp_path, demo):
    class Loop(DemoFormat):
        name = "loop"

        def materialize(self, task_dir: Path, out_root: Path) -> Path:
            return task_dir

    formats._registered[:] = [Loop()]
    src = _demo_task(tmp_path / "src", "a", "alpha")
    with pytest.raises(RuntimeError, match="still claimed"):
        formats.materialize_task_dir(src)


def test_register_rejects_non_formats(demo):
    with pytest.raises(TypeError):
        formats.register_task_format(object())  # type: ignore[arg-type]


def test_broken_entry_point_is_ignored(monkeypatch):
    class BrokenEP:
        name = "broken"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(formats, "_registered", [])
    monkeypatch.setattr(formats, "_entry_point_formats", None)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [BrokenEP()])
    # Only the built-in formats remain.
    assert [f.name for f in formats.task_formats()] == ["taskmd"]


def test_evaluation_runs_claimed_tasks_as_native_packages(tmp_path, demo):
    tasks = tmp_path / "tasks"
    _demo_task(tasks, "src-b", "beta")
    _demo_task(tasks, "src-a", "alpha")
    _write_native(tasks / "native-c", "native-c")
    (tasks / "not-a-task").mkdir()
    job = Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs")
    dirs = job._get_task_dirs()
    # ordered by source folder name; claimed folders run as their materialized packages
    assert [d.name for d in dirs] == ["native-c", "alpha", "beta"]
    assert dirs[0] == tasks / "native-c"
    assert dirs[1].parent == tmp_path / "cache" / "demo"


def test_evaluation_single_claimed_task_root(tmp_path, demo):
    src = _demo_task(tmp_path, "one", "alpha")
    job = Evaluation(tasks_dir=src, jobs_dir=tmp_path / "jobs")
    assert [d.name for d in job._get_task_dirs()] == ["alpha"]
    cfg = EvaluationConfig(exclude_tasks={"alpha"})
    job = Evaluation(tasks_dir=src, jobs_dir=tmp_path / "jobs2", config=cfg)
    assert job._get_task_dirs() == []


def test_rollout_config_uses_materialized_package(tmp_path, demo):
    from benchflow.rollout import RolloutConfig

    src = _demo_task(tmp_path, "one", "alpha")
    cfg = RolloutConfig(task_path=src, agent="oracle")
    assert cfg.task_path == tmp_path / "cache" / "demo" / "alpha"
