from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_build_ledger_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "skillsbench-fill" / "build_ledger.py"
    spec = importlib.util.spec_from_file_location("skillsbench_fill_ledger", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _base_row(cell_id: str) -> dict:
    return {
        "cell_id": cell_id,
        "model": "gemini-3.5-flash",
        "skill_mode": "with",
        "task": "citation-check",
        "trial_slot": 1,
        "status": "queued",
    }


def test_ledger_review_partial_overwrites_stale_runner_state(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against stale runner partial de-crediting reviews."""
    build_ledger = _load_build_ledger_module()
    cell = "gemini-3.5-flash__with__citation-check__t1"
    _write_json(tmp_path / "grid.json", {"target": 1, "rows": [_base_row(cell)]})
    _write_json(
        tmp_path / "state" / f"{cell}.json",
        {"cell_id": cell, "status": "completed", "partial": True},
    )
    _write_json(
        tmp_path / "review" / f"{cell}.json",
        {
            "cell_id": cell,
            "verdict": "pass",
            "health": "healthy",
            "trial_id": "abc",
            "partial_trajectory": False,
        },
    )
    _write_json(
        tmp_path / "published" / f"{cell}.json",
        {"cell_id": cell, "hf_path": "root/citation-check__abc", "tid": "abc"},
    )

    assert build_ledger.main(["--root", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "experiments_ledger.json").read_text())["rows"][0]

    assert row["status"] == "published"
    assert row["partial"] is False


def test_ledger_uncredits_published_partial_without_overlay(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against terminal credit for raw partial publishes."""
    build_ledger = _load_build_ledger_module()
    cell = "gemini-3.5-flash__with__citation-check__t1"
    _write_json(tmp_path / "grid.json", {"target": 1, "rows": [_base_row(cell)]})
    _write_json(
        tmp_path / "review" / f"{cell}.json",
        {
            "cell_id": cell,
            "verdict": "pass",
            "health": "healthy",
            "trial_id": "abc",
            "partial_trajectory": True,
            "timeout_complete_artifacts": True,
        },
    )
    _write_json(
        tmp_path / "published" / f"{cell}.json",
        {"cell_id": cell, "hf_path": "root/citation-check__abc", "tid": "abc"},
    )

    assert build_ledger.main(["--root", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "experiments_ledger.json").read_text())["rows"][0]

    assert row["status"] == "review_pass"
    assert row["uncredited_hf_path"] == "root/citation-check__abc"


def test_ledger_uncredits_publish_tid_mismatch(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against crediting a different trial than reviewed."""
    build_ledger = _load_build_ledger_module()
    cell = "gemini-3.5-flash__with__citation-check__t1"
    _write_json(tmp_path / "grid.json", {"target": 1, "rows": [_base_row(cell)]})
    _write_json(
        tmp_path / "review" / f"{cell}.json",
        {
            "cell_id": cell,
            "verdict": "pass",
            "health": "healthy",
            "trial_id": "reviewed",
            "partial_trajectory": False,
        },
    )
    _write_json(
        tmp_path / "published" / f"{cell}.json",
        {"cell_id": cell, "hf_path": "root/citation-check__other", "tid": "other"},
    )

    assert build_ledger.main(["--root", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "experiments_ledger.json").read_text())["rows"][0]

    assert row["status"] == "review_pass"
    assert row["uncredited_hf_path"] == "root/citation-check__other"
