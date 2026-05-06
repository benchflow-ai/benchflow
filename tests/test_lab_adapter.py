"""Smoke tests for the LAB adapter.

Exercises the translation logic on a synthetic task fixture so we can
verify the generated layout without cloning harveyai/harvey-labs or
calling Gemini.  No network, no Docker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ADAPTER_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "lab"
sys.path.insert(0, str(ADAPTER_DIR))

from adapter.translate import (  # noqa: E402
    discover_tasks,
    sanitize_task_id,
    write_task,
)


def _make_lab_task(root: Path, parts: list[str], cfg: dict, docs: dict[str, str]) -> Path:
    """Materialise a synthetic LAB task on disk."""
    d = root / "tasks" / Path(*parts)
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps(cfg))
    (d / "documents").mkdir()
    for name, body in docs.items():
        (d / "documents" / name).write_text(body)
    return d


@pytest.fixture
def fake_lab(tmp_path: Path) -> Path:
    """A two-task LAB clone: one flat, one nested with a scenario."""
    cfg_flat = {
        "title": "Extract Counterparty",
        "work_type": "extract",
        "tags": ["M&A"],
        "instructions": "Extract the counterparty name into counterparty.md.",
        "deliverables": {"counterparty.md": "counterparty.md"},
        "criteria": [
            {
                "id": "C-001",
                "title": "Counterparty named",
                "match_criteria": "PASS if counterparty is named.",
                "deliverables": ["counterparty.md"],
            }
        ],
    }
    _make_lab_task(tmp_path, ["corporate-ma", "extract-counterparty"],
                   cfg_flat, {"contract.txt": "Buyer: Acme. Seller: Beta."})

    cfg_nested = dict(cfg_flat, title="Scenario task")
    _make_lab_task(tmp_path, ["real-estate", "extract-key-terms", "scenario-01"],
                   cfg_nested, {"psa.txt": "Sale Price: $1M"})
    return tmp_path


def test_sanitize_task_id_joins_parts():
    assert sanitize_task_id(["a", "b", "c"]) == "a__b__c"


def test_sanitize_task_id_lowercases_and_strips():
    assert sanitize_task_id(["My Task ", "scenario 01"]) == "my-task__scenario-01"


def test_sanitize_task_id_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_task_id([])


def test_discover_finds_flat_and_nested(fake_lab: Path):
    tasks = discover_tasks(fake_lab)
    assert len(tasks) == 2
    rids = {t.relative_id for t in tasks}
    assert rids == {
        "corporate-ma/extract-counterparty",
        "real-estate/extract-key-terms/scenario-01",
    }


def test_discover_preserves_config(fake_lab: Path):
    tasks = discover_tasks(fake_lab)
    flat = next(t for t in tasks if "scenario" not in t.relative_id)
    assert flat.config["title"] == "Extract Counterparty"
    assert flat.config["criteria"][0]["id"] == "C-001"


def test_write_task_creates_required_layout(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    target = write_task(tasks[0], out)
    for rel in [
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/documents",
        "tests/test.sh",
        "tests/rubric_judge.py",
        "tests/criteria.json",
        "tests/task_desc.txt",
        "solution/solve.sh",
    ]:
        assert (target / rel).exists(), f"missing {rel}"


def test_write_task_copies_documents(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    docs = (out / tasks[0].task_id / "environment" / "documents")
    assert (docs / "contract.txt").read_text().startswith("Buyer:")


def test_write_task_carries_rubric(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    crit = json.loads(
        (out / tasks[0].task_id / "tests" / "criteria.json").read_text()
    )
    assert crit[0]["id"] == "C-001"
    assert "Counterparty" in crit[0]["title"]


def test_write_task_instruction_preamble_first(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    instr = (out / tasks[0].task_id / "instruction.md").read_text()
    # preamble + actual task body
    assert instr.startswith("You are an AI agent")
    assert "Extract the counterparty" in instr


def test_rubric_judge_script_parses(fake_lab: Path, tmp_path: Path):
    """Make sure the embedded rubric_judge.py is valid Python."""
    import ast
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    src = (out / tasks[0].task_id / "tests" / "rubric_judge.py").read_text()
    ast.parse(src)


def test_test_sh_executable(fake_lab: Path, tmp_path: Path):
    """test.sh must be marked executable so test.sh works inside the verifier."""
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    test_sh = (out / tasks[0].task_id / "tests" / "test.sh")
    mode = test_sh.stat().st_mode & 0o777
    assert mode & 0o100, f"test.sh not user-executable (mode={oct(mode)})"


def test_idempotent_without_force(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    target1 = write_task(tasks[0], out)
    # Drop a marker in the existing dir; without force=True, write_task
    # must not stomp on it.
    marker = target1 / "marker.txt"
    marker.write_text("preserved")
    target2 = write_task(tasks[0], out, force=False)
    assert target1 == target2
    assert marker.exists()


def test_force_overwrites(fake_lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    tasks = discover_tasks(fake_lab)
    write_task(tasks[0], out)
    marker = out / tasks[0].task_id / "marker.txt"
    marker.write_text("preserved")
    write_task(tasks[0], out, force=True)
    assert not marker.exists()
