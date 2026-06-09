from __future__ import annotations

import click
from typer.testing import CliRunner

from benchflow.cli.main import app


def test_skills_eval_rejects_model_agent_length_mismatch_before_header(tmp_path):
    skill_dir = tmp_path / "skill"
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "evals.json").write_text(
        '{"skill_name":"citation-management",'
        '"cases":[{"id":"case-1","question":"Q?","ground_truth":"A"}]}'
    )

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "eval",
            str(skill_dir),
            "--agent",
            "gemini",
            "--model",
            "gemini-2.5-flash",
            "--model",
            "extra-model",
            "--no-baseline",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    output = click.unstyle(result.output)
    assert result.exit_code == 1
    assert "--model may be provided once for all agents or once per --agent" in output
    assert "got 2 models" in output
    assert "for 1 agents" in output
    assert "Skill eval:" not in output
    assert "Traceback" not in output
