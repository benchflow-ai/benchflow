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


def test_eval_run_agents_threads_floor_flags(tmp_path, monkeypatch):
    # The floor-shape flags fold onto the standard `eval run` and reach the runner.
    from benchflow.cli import arena as arena_mod

    captured = {}
    monkeypatch.setattr(arena_mod, "run_floor_from_cli", lambda **kw: captured.update(kw))
    r = runner.invoke(app, [
        "eval", "run", "--agents", str(_roster(tmp_path)),
        "--environment-manifest", "benchmarks/casinobench/environment.toml",
        "--game", "six-deck-blackjack-s17", "--multiplayer",
        "--url-env", "CASINO_URL", "--seat-env", "CASINOBENCH_SEAT_ID",
        "--standings-path", "/_admin/standings", "--events-path", "/_admin/events",
    ])
    assert r.exit_code == 0, r.output
    assert captured["multiplayer"] is True
    assert captured["game"] == "six-deck-blackjack-s17"
    assert captured["url_env"] == "CASINO_URL"
    assert captured["seat_env"] == "CASINOBENCH_SEAT_ID"
    assert captured["standings_path"] == "/_admin/standings"
    assert captured["events_path"] == "/_admin/events"
