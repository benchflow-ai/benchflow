"""A session-log copy whose redaction raises is never skipped in silence.

Review of PR #122 (F2b-R122C, 2026-11-14, F1; the lead's guard): when the
redaction raised, both copies were dropped, the mirror warned once with the
exception's text and teardown logged at info as if the sandbox were gone,
and the trial's count was short with no word of it (or with the wrong one,
codex-acp's "no Codex session log was left"). Now the copy that failed is
named in a warning by the exception's class, never its message, and the
trial is marked: ``usage_incomplete`` with a reason that names the class,
``cost_incomplete`` beside a cost, in result.json, the run card and
``bench results``. Every value is made up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.rollout import _session_log, _session_mirror
from benchflow.rollout._cost import INCOMPLETE_REDACTION
from tests.test_codex_acp_tokens import THREE_CALLS, _codex_log, _summed
from tests.test_en374_salvage import OPUS, _tokens, _usage
from tests.test_en374_session_mirror import LOST, _acp_rollout
from tests.test_en374_usage_incomplete import _respond

SECRET = "zq9-FAKE-plain-credential-value-77"
MESSAGE = "zq9-FAKE-text-the-redaction-was-holding"
REASON = INCOMPLETE_REDACTION.format(error="RuntimeError")


def _boom(*_args, **_kwargs):
    raise RuntimeError(MESSAGE)


def _with_downloads(sandbox):
    """The local sandbox, able to hand files back, so the full copy at
    teardown is made (the test sandbox alone cannot, and the copy made while
    the turns ran is placed instead)."""
    import shutil

    async def download_file(src, dst):
        shutil.copyfile(src, dst)

    sandbox.download_file = download_file
    return sandbox


def _logged(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_mirror_that_cannot_redact_marks_the_count(
    tmp_path, monkeypatch, caplog
):
    """A Claude Code trial over ACP: turn 1 reports, turn 2 is cut off when
    the sandbox is lost, and the copy made while the turns ran could not be
    redacted, so nothing counted turn 2."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS, Evaluation

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(_session_log, "scrub_secrets", _boom)
    rollout, trial, sandbox, home = _acp_rollout(
        tmp_path, monkeypatch, "claude-agent-acp"
    )
    rollout._agent_env = {"BENCHFLOW_PROVIDER_API_KEY": SECRET}
    rollout._resolved_prompts = ["Build the site.", "Now add tests."]
    object.__setattr__(rollout._config, "model", OPUS)
    log = home / ".claude" / "projects" / "-app" / "session-1.jsonl"

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        _respond(log, "msg_1", _usage(1500, 0, 3000, 120))
        session.record_prompt_usage(
            {
                "inputTokens": 1500,
                "outputTokens": 120,
                "cachedReadTokens": 0,
                "cachedWriteTokens": 3000,
                "totalTokens": 4620,
            }
        )
        session.record_user_prompt(prompt_list[1])
        _respond(log, "msg_2", _usage(20, 3000, 400, 300))
        await asyncio.sleep(0.4)  # the copy ticks, and fails, meanwhile
        sandbox.lost = True
        raise RuntimeError(LOST)

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    with pytest.raises(RuntimeError, match=LOST):
        await rollout.execute()
    rollout._env = None
    await rollout._cleanup_before_teardown()
    rollout._build_result()

    text = (trial / "result.json").read_text()
    data = json.loads(text)
    agent = data["agent_result"]
    assert (agent["n_input_tokens"], agent["n_output_tokens"]) == (1500, 120)
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == REASON
    assert agent["cost_usd"] is not None and agent["cost_incomplete"] is True
    assert data["final_metrics"]["extra"]["usage_incomplete_reason"] == REASON
    assert MESSAGE not in text and MESSAGE not in _logged(caplog)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("could not be redacted (RuntimeError)" in r.getMessage() for r in warned)

    # The run card carries summary.json's counts.
    summary = usage_summary({trial.name: data})
    holder = SimpleNamespace(_run_card={})
    Evaluation._log_run_card_after(
        holder,
        trial.parent,
        {},
        None,
        usage={key: summary.get(key) for key in RUN_CARD_USAGE_FIELDS},
    )
    card = json.loads((trial.parent / "run-card.json").read_text())
    assert card["after_run"]["usage"]["trials_usage_incomplete"] == 1
    assert card["after_run"]["usage"]["trials_cost_incomplete"] == 1

    shown = CliRunner().invoke(
        app, ["results", str(trial.parent)], env={"COLUMNS": "300"}
    )
    assert shown.exit_code == 0, shown.output
    plain = " ".join(re.sub("[─-╿]", " ", shown.stdout).split())
    assert f"~${agent['cost_usd']:.4f}+" in plain
    assert f"floor: {REASON}" in plain


@pytest.mark.asyncio
async def test_a_teardown_copy_that_cannot_redact_names_the_cause(
    tmp_path, monkeypatch, caplog
):
    """A codex-acp trial whose turn ended: both copies fail their redaction,
    and the record says so, not that no Codex log was left."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(_session_log, "scrub_secrets", _boom)
    rollout, trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    _with_downloads(sandbox)
    rollout._agent_env = {"BENCHFLOW_PROVIDER_API_KEY": SECRET}

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        _codex_log(home, THREE_CALLS)
        await asyncio.sleep(0.2)
        session.record_prompt_usage(
            {"inputTokens": 13500, "outputTokens": 84, "totalTokens": 13584}
        )
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    await rollout.execute()
    await rollout._cleanup_before_teardown()
    rollout._build_result()

    text = (trial / "result.json").read_text()
    agent = json.loads(text)["agent_result"]
    assert agent["total_tokens"] == 13584
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == REASON
    assert not (trial / "agent" / "codex-sessions").exists()
    assert MESSAGE not in text and MESSAGE not in _logged(caplog)
    assert "the copy made at teardown was not kept" in _logged(caplog)


@pytest.mark.asyncio
async def test_a_mirror_failure_with_a_teardown_copy_kept_is_complete(
    tmp_path, monkeypatch, caplog
):
    """The copy made while the turn ran fails, the full copy at teardown
    does not: the count is the full copy's, whole, and the failure is still
    said."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(_session_mirror, "_clean", _boom)
    rollout, trial, sandbox, home = _acp_rollout(tmp_path, monkeypatch, "codex-acp")
    _with_downloads(sandbox)
    rollout._agent_env = {"BENCHFLOW_PROVIDER_API_KEY": SECRET}

    async def prompts(_client, session, prompt_list, *_args, **_kwargs):
        session.record_user_prompt(prompt_list[0])
        _codex_log(home, THREE_CALLS)
        await asyncio.sleep(0.2)
        session.record_prompt_usage(
            {"inputTokens": 13500, "outputTokens": 84, "totalTokens": 13584}
        )
        return [], 0

    monkeypatch.setattr(rollout._planes, "execute_prompts", prompts)
    await rollout.execute()
    await rollout._cleanup_before_teardown()
    usage = rollout._usage_metrics
    assert (trial / "agent" / "codex-sessions").is_dir()
    assert _tokens(usage) == _summed(THREE_CALLS)
    assert not usage.get("usage_incomplete")
    assert "the copy made while its turns ran was not kept" in _logged(caplog)
    assert MESSAGE not in _logged(caplog)
