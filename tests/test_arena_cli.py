"""Multi-agent floor is the standard `bench eval run --agents` (plural of --agent)."""

from __future__ import annotations

import textwrap

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def _roster(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(textwrap.dedent("""
        agents:
          - { name: codex, agent: codex-acp }
    """))
    return p


def test_eval_run_exposes_agents_and_drive():
    r = runner.invoke(app, ["eval", "run", "--help"])
    assert r.exit_code == 0
    assert "--agents" in r.output and "--drive" in r.output


def test_agents_requires_environment_manifest(tmp_path):
    # --agents without --environment-manifest fails closed (no docker touched).
    r = runner.invoke(app, ["eval", "run", "--agents", str(_roster(tmp_path))])
    assert r.exit_code == 1


def test_agents_mutually_exclusive_with_agent(tmp_path):
    r = runner.invoke(app, [
        "eval", "run", "--agents", str(_roster(tmp_path)),
        "--agent", "codex-acp", "--environment-manifest", "x.toml",
    ])
    assert r.exit_code == 1


def test_arena_run_is_deprecated_alias(tmp_path):
    # The old `arena run` still works (hidden alias) and requires the manifest.
    r = runner.invoke(app, ["arena", "run", "--agents", str(_roster(tmp_path))])
    assert r.exit_code != 0  # missing required --environment-manifest
