"""`benchflow arena run` is registered and wired to the manifest runner."""

from __future__ import annotations

import textwrap

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def test_arena_group_registered():
    r = runner.invoke(app, ["arena", "--help"])
    assert r.exit_code == 0
    assert "run" in r.output


def test_arena_run_help_lists_agents_option():
    r = runner.invoke(app, ["arena", "run", "--help"])
    assert r.exit_code == 0
    assert "--agents" in r.output


def test_arena_run_bootstrap_error_exits_nonzero(tmp_path, monkeypatch):
    # A manifest with no service + no image_dir → bootstrap fails closed (exit 1),
    # without touching docker.
    p = tmp_path / "agents.yaml"
    p.write_text(textwrap.dedent("""
        prompt: play
        agents:
          - { name: codex, agent: codex-acp }
    """))
    r = runner.invoke(app, ["arena", "run", "--agents", str(p)])
    assert r.exit_code == 1
