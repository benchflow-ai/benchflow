"""Guards FF-sw-other skill judge answer boundary at 363d815b."""

import json
import runpy
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parents[1] / "src/benchflow/templates/judge.py.tmpl"


def load_judge(tmp_path, monkeypatch):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "case.json").write_text(
        json.dumps(
            {
                "question": "Reply with Paris",
                "ground_truth": "Paris",
                "expected_behavior": [],
            }
        )
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setenv("BENCHFLOW_VERIFIER_DIR", str(verifier))
    monkeypatch.setenv("BENCHFLOW_AGENT_LOG_DIR", str(logs))
    monkeypatch.setenv("BENCHFLOW_WORKSPACE", str(app))
    script = tmp_path / "judge.py"
    script.write_text(TEMPLATE.read_text().replace("${judge_model}", "gpt-4o-mini"))
    return runpy.run_path(str(script)), logs, app, verifier


@pytest.mark.parametrize(
    "source",
    [
        "prompt",
        "transcript",
        "workspace",
        "link",
        "parent",
        "missing",
        "wrong",
        "correct",
    ],
)
def test_exact_match_only_accepts_published_final_reply(tmp_path, monkeypatch, source):
    judge, logs, app, verifier = load_judge(tmp_path, monkeypatch)
    if source == "prompt":
        (logs / "prompt.txt").write_text("Reply with Paris")
    elif source == "transcript":
        (logs / "acp_trajectory.jsonl").write_text(
            json.dumps({"role": "user", "content": "Paris"})
        )
    elif source == "workspace":
        (app / "input.txt").write_text("Paris")
    elif source == "link":
        (logs / "output.txt").symlink_to(verifier / "case.json")
    elif source == "parent":
        logs.rmdir()
        logs.symlink_to(app, target_is_directory=True)
        (app / "output.txt").write_text("Paris")
    if source not in {"link", "parent", "missing"}:
        (logs / "output.txt").write_text("Paris" if source == "correct" else "London")
    assert judge["judge_exact_match"]() == (1.0 if source == "correct" else 0.0)
