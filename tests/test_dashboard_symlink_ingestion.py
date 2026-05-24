"""Symlink-defence regression tests for dashboard data ingestion.

Covers issues #390 (rollout artifact ingestion) and #416 (experiments and labs
ingestion). Each test plants a symlink under the dashboard's ingestion root
that points to an out-of-tree secret and asserts:

  * the secret content does NOT appear in the produced payload
  * a warning is emitted naming the rejected symlink

The dashboard package isn't an installed module; ``generate.py`` loads itself
via a file-spec loader. We do the same in these tests.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATE = ROOT / "dashboard" / "generate.py"

SECRET = "SECRET_FROM_HOST_DASHBOARD_LEAK"


def _load_generate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "dashboard_generate_under_test", GENERATE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generate() -> ModuleType:
    return _load_generate()


def _secret_outside(tmp_path: Path) -> Path:
    secret = tmp_path / "host-secret.txt"
    secret.write_text(SECRET)
    return secret


# ---------------------------------------------------------------------------
# #390 — rollout artifact ingestion
# ---------------------------------------------------------------------------


def test_task_artifacts_skips_symlink_in_artifacts_subdir(
    tmp_path: Path,
    generate: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`_task_artifacts` must not embed the target of a symlinked artifact."""
    secret = _secret_outside(tmp_path)
    rollout = tmp_path / "rollout"
    (rollout / "artifacts").mkdir(parents=True)
    (rollout / "artifacts" / "leak.txt").symlink_to(secret)

    # Drop a real, in-tree artifact so we still exercise the happy path.
    (rollout / "artifacts" / "ok.txt").write_text("legitimate-artifact")

    with caplog.at_level(logging.WARNING):
        arts = generate._task_artifacts(rollout)

    contents = [a.get("content") or "" for a in arts]
    assert SECRET not in "\n".join(contents), (
        "dashboard ingested host secret through symlinked artifact"
    )
    # The in-tree artifact survived.
    assert any("legitimate-artifact" in (c or "") for c in contents)
    # Exactly one warning per skipped link.
    assert any("leak.txt" in r.message for r in caplog.records)


def test_task_artifacts_skips_symlink_at_rollout_root(
    tmp_path: Path, generate: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """Top-level rollout files are also iterated; a symlink there must not leak."""
    secret = _secret_outside(tmp_path)
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "result.json").symlink_to(secret)

    with caplog.at_level(logging.WARNING):
        arts = generate._task_artifacts(rollout)

    assert all(SECRET not in (a.get("content") or "") for a in arts)
    assert any("result.json" in r.message for r in caplog.records)


def test_file_payload_refuses_symlink(
    tmp_path: Path, generate: ModuleType
) -> None:
    """The shared ``_file_payload`` helper must reject symlinks directly."""
    secret = _secret_outside(tmp_path)
    link = tmp_path / "link.txt"
    link.symlink_to(secret)

    content, lines, truncated, _lang = generate._file_payload(link)
    assert content is None
    assert lines == 0
    assert truncated is False


# ---------------------------------------------------------------------------
# #416 — experiments and labs ingestion
# ---------------------------------------------------------------------------


def test_collect_experiments_skips_symlinks_in_experiments_dir(
    tmp_path: Path,
    generate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A symlink under experiments/ must not surface in any experiment file."""
    secret = _secret_outside(tmp_path)
    exp_root = tmp_path / "fake-repo"
    (exp_root / "experiments").mkdir(parents=True)
    (exp_root / "labs").mkdir()
    (exp_root / "experiments" / "ablation_link.md").symlink_to(secret)
    # Add a legitimate file so the bucket is non-empty.
    (exp_root / "experiments" / "ablation_real.md").write_text("real-experiment")

    monkeypatch.setattr(generate, "ROOT", exp_root)

    with caplog.at_level(logging.WARNING):
        timeline = generate.collect_experiments()

    flat_content = "\n".join(
        (f.get("content") or "") for entry in timeline for f in entry["files"]
    )
    assert SECRET not in flat_content
    assert "real-experiment" in flat_content
    assert any("ablation_link.md" in r.message for r in caplog.records)


def test_collect_experiments_skips_symlinks_in_labs_subdir(
    tmp_path: Path,
    generate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Labs walks an attacker-controllable subtree — must not follow links."""
    secret = _secret_outside(tmp_path)
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "labs" / "my-lab").mkdir(parents=True)
    (fake_repo / "labs" / "my-lab" / "leak.md").symlink_to(secret)
    (fake_repo / "labs" / "my-lab" / "notes.md").write_text("real-lab-note")

    monkeypatch.setattr(generate, "ROOT", fake_repo)

    with caplog.at_level(logging.WARNING):
        timeline = generate.collect_experiments()

    labs_entries = [e for e in timeline if e["source"].startswith("labs/")]
    assert labs_entries, "labs entry missing — fixture broken"
    flat_content = "\n".join(
        (f.get("content") or "") for e in labs_entries for f in e["files"]
    )
    assert SECRET not in flat_content
    assert "real-lab-note" in flat_content
    assert any("leak.md" in r.message for r in caplog.records)
