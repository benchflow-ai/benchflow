"""The decoupled roster: `--agents roster.yaml` is the file form of repeated
`--agent/--model` — ONLY the agents list, no task/service/sandbox/out/prompt
(those follow the standard single-agent `bench eval run` flags)."""

from __future__ import annotations

import textwrap

import pytest

from benchflow.arena.roster import Roster


def _write(tmp_path, body: str):
    p = tmp_path / "roster.yaml"
    p.write_text(textwrap.dedent(body))
    return p


@pytest.mark.parametrize("key", ["task", "services", "sandbox", "out", "prompt", "drive"])
def test_run_level_keys_rejected_with_migration_hint(tmp_path, key):
    p = _write(
        tmp_path,
        f"""
        {key}: something
        agents:
          - {{ name: x, agent: codex-acp }}
        """,
    )
    with pytest.raises(ValueError, match="bench eval run"):
        Roster.from_yaml(p)


def test_seat_id_defaults_to_agent_dash_model(tmp_path):
    # No `name:` → the player id is <agent>-<model> (provider prefix stripped),
    # so the floor/viewer identify agent + model. count appends -0..-(n-1).
    p = _write(
        tmp_path,
        """
        agents:
          - { agent: codex-acp, model: gpt-5.5, count: 2 }
          - { agent: deepagents, model: deepseek/deepseek-v4-pro }
        """,
    )
    ids = [s.seat_id for s in Roster.from_yaml(p).seats()]
    assert ids == ["codex-acp-gpt-5.5-0", "codex-acp-gpt-5.5-1", "deepagents-deepseek-v4-pro"]


def test_explicit_name_overrides_agent_model_id(tmp_path):
    p = _write(tmp_path, """
        agents:
          - { name: hero, agent: codex-acp, model: gpt-5.5 }
    """)
    assert [s.seat_id for s in Roster.from_yaml(p).seats()] == ["hero"]


def test_roster_is_a_pure_agents_list(tmp_path):
    p = _write(
        tmp_path,
        """
        agents:
          - { name: codex, agent: codex-acp, model: gpt-5.5, count: 2 }
          - { name: claude, agent: claude-agent-acp }
        """,
    )
    seats = Roster.from_yaml(p).seats()
    assert [s.seat_id for s in seats] == ["codex-0", "codex-1", "claude"]
    assert seats[0].config.name == "codex-acp"
