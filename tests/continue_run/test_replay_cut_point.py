"""Replay cut-point tests — rollout-branching RFC §3.5, FrontierPhysics#73.

``max_exchanges`` replays at most the first K recorded exchanges, then switches
the proxy to live passthrough exactly as if the recording had ended there. The
router exposes cut-point accounting (``n_replayed_exchanges`` +
``cut_point_digest``), a cut continue-run's ``source_provenance`` gains a
``cut_point`` block, and ``stage_tags`` + ``cut_stage`` name a cut by recorded
stage. Unit tests against the existing continue_run test doubles — no Docker,
no API keys.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.continue_run.orchestrator import (
    build_agent_env,
    build_rollout_config,
    cut_point_provenance,
    resolve_cut_point,
    stitched_trajectory_lines,
)
from benchflow.continue_run.replay_proxy import (
    ReplayCutPointError,
    ReplayRouter,
    request_body_digest,
    validate_max_exchanges,
)
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder

from ._helpers import completion, exchange, write_run_folder

runner = CliRunner()


def _recorded(n: int = 3):
    """n exchanges with distinct contents and distinct request bodies."""
    return [
        exchange(completion(content=f"turn-{i}"), n_request_messages=i + 1)
        for i in range(n)
    ]


def _request(i: int) -> dict:
    """The agent request matching ``_recorded``'s i-th turn (no divergence)."""
    return {"messages": [{"role": "user"}] * (i + 1)}


# ── ReplayRouter: max_exchanges cut ───────────────────────────────────────


def test_cut_serves_exactly_k_then_live():
    recorded = _recorded(3)
    live_requests = []

    def forwarder(req):
        live_requests.append(req)
        return completion(content="LIVE")

    router = ReplayRouter(recorded, live_forwarder=forwarder, max_exchanges=2)

    r1 = router.next_response(_request(0))
    assert r1.source == "replay"
    assert r1.body["choices"][0]["message"]["content"] == "turn-0"
    assert router.exhausted is False

    r2 = router.next_response(_request(1))
    assert r2.source == "replay"
    assert r2.body["choices"][0]["message"]["content"] == "turn-1"
    # the cut behaves like the natural end of the recording
    assert router.exhausted is True

    r3 = router.next_response(_request(2))
    assert r3.source == "live"
    assert r3.body["choices"][0]["message"]["content"] == "LIVE"
    # exactly K recorded turns served; the third recording is never replayed
    assert router.n_replayed_exchanges == 2
    assert len(live_requests) == 1
    assert len(router.live_exchanges) == 1
    assert router.divergences == 0


def test_cut_without_forwarder_errors_like_natural_exhaustion():
    router = ReplayRouter(_recorded(3), live_forwarder=None, max_exchanges=1)
    assert router.next_response(_request(0)).source == "replay"
    result = router.next_response(_request(1))
    assert result.source == "error"
    assert result.status == 503
    assert result.body["error"]["type"] == "replay_exhausted"


def test_cut_at_n_equals_full_prefix_behavior():
    recorded = _recorded(2)
    cut = ReplayRouter(
        recorded, live_forwarder=lambda req: completion(content="L"), max_exchanges=2
    )
    full = ReplayRouter(recorded, live_forwarder=lambda req: completion(content="L"))
    for router in (cut, full):
        sources = [router.next_response(_request(i)).source for i in range(3)]
        assert sources == ["replay", "replay", "live"]
    assert cut.n_replayed_exchanges == full.n_replayed_exchanges == 2
    assert cut.cut_point_digest == full.cut_point_digest


@pytest.mark.parametrize("bad", [0, -1, 4])
def test_invalid_cut_fails_closed_at_configuration_time(bad):
    with pytest.raises(ReplayCutPointError, match=r"max_exchanges must be in 1\.\.3"):
        ReplayRouter(_recorded(3), max_exchanges=bad)


def test_validate_max_exchanges_none_means_full_prefix():
    assert validate_max_exchanges(None, 5) == 5
    assert validate_max_exchanges(3, 5) == 3


# ── cut-point accounting ──────────────────────────────────────────────────


def test_cut_point_digest_matches_independent_sha256():
    recorded = _recorded(3)
    router = ReplayRouter(
        recorded, live_forwarder=lambda req: completion(content="L"), max_exchanges=2
    )
    assert router.cut_point_digest is None  # nothing replayed yet
    for i in range(3):
        router.next_response(_request(i))

    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                recorded[1].request.body, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert router.n_replayed_exchanges == 2
    assert router.cut_point_digest == expected
    assert request_body_digest(recorded[1].request.body) == expected


def test_cut_point_digest_deterministic_across_runs():
    recorded = _recorded(3)
    digests = []
    for _ in range(2):
        router = ReplayRouter(
            recorded,
            live_forwarder=lambda req: completion(content="L"),
            max_exchanges=2,
        )
        for i in range(2):
            router.next_response(_request(i))
        digests.append(router.cut_point_digest)
    assert digests[0] == digests[1]
    assert digests[0] is not None


def test_natural_end_exposes_cut_point_accounting_too():
    recorded = _recorded(2)
    router = ReplayRouter(recorded, live_forwarder=lambda req: completion(content="L"))
    for i in range(3):
        router.next_response(_request(i))
    assert router.n_replayed_exchanges == 2
    assert router.cut_point_digest == request_body_digest(recorded[1].request.body)


# ── provenance: the cut_point block in source_provenance ──────────────────


def _build_config(tmp_path, run, **kwargs):
    task = tmp_path / "real-task"
    task.mkdir(exist_ok=True)
    return build_rollout_config(
        run,
        task_path=task,
        live_model="gemini-3.1-flash-lite-preview",
        agent_env=build_agent_env("http://host:1/v1"),
        timeout=123,
        output_dir=tmp_path / "out",
        rollout_name="demo-task__continued",
        **kwargs,
    )


def test_provenance_gains_cut_point_block(tmp_path):
    folder = write_run_folder(tmp_path / "run", exchanges=_recorded(3))
    run = load_run_folder(folder)
    cfg = _build_config(tmp_path, run, max_exchanges=2)
    cut = cfg.source_provenance["cut_point"]
    assert cut["n_replayed_exchanges"] == 2
    assert cut["cut_point_digest"] == request_body_digest(run.exchanges[1].request.body)
    assert "branch_stage" not in cut
    # existing provenance fields are preserved alongside the new block
    assert cfg.source_provenance["kind"] == "benchflow-continue"
    assert cfg.source_provenance["n_recorded_exchanges"] == 3


def test_provenance_natural_end_documents_cut_point(tmp_path):
    folder = write_run_folder(tmp_path / "run", exchanges=_recorded(3))
    run = load_run_folder(folder)
    cfg = _build_config(tmp_path, run)  # no max_exchanges
    cut = cfg.source_provenance["cut_point"]
    assert cut["n_replayed_exchanges"] == 3
    assert cut["cut_point_digest"] == request_body_digest(run.exchanges[2].request.body)


# ── stage-named cuts ──────────────────────────────────────────────────────


def test_stage_named_cut_resolves_and_records_branch_stage():
    tags = {"env-ready": 1, "post-research": 2}
    resolved, stage = resolve_cut_point(
        None, stage_tags=tags, cut_stage="post-research"
    )
    assert (resolved, stage) == (2, "post-research")
    block = cut_point_provenance(
        _recorded(3), max_exchanges=resolved, branch_stage=stage
    )
    assert block["branch_stage"] == "post-research"
    assert block["n_replayed_exchanges"] == 2


def test_unknown_cut_stage_names_available_stages():
    tags = {"env-ready": 1, "post-research": 2}
    with pytest.raises(RunFolderError, match="env-ready, post-research"):
        resolve_cut_point(None, stage_tags=tags, cut_stage="pre-verify")


def test_cut_stage_without_stage_tags_fails_closed():
    with pytest.raises(RunFolderError, match="stage_tags"):
        resolve_cut_point(None, cut_stage="post-research")


def test_cut_stage_and_max_exchanges_are_exclusive():
    with pytest.raises(RunFolderError, match="not both"):
        resolve_cut_point(2, stage_tags={"env-ready": 1}, cut_stage="env-ready")


def test_resolve_cut_point_passthrough_without_stage():
    assert resolve_cut_point(None) == (None, None)
    assert resolve_cut_point(4) == (4, None)


# ── stitched trajectory: prefix truncated at the cut ──────────────────────


def test_stitched_prefix_truncated_at_cut(tmp_path):
    original = tmp_path / "orig.jsonl"
    original.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    lines = stitched_trajectory_lines(
        original, [exchange(completion(content="L"))], max_recorded=2
    )
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
    last = json.loads(lines[2])
    assert last["response"]["body"]["choices"][0]["message"]["content"] == "L"


# ── CLI: --max-exchanges reaches the orchestrator ─────────────────────────


def _patch_continue_run(monkeypatch, tmp_path, captured):
    import benchflow.continue_run.orchestrator as orch

    async def fake_continue_run(folder, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            rollout_dir=tmp_path / "r",
            n_recorded=2,
            n_live=1,
            divergences=0,
            rewards={"reward": 1.0},
            error=None,
        )

    monkeypatch.setattr(orch, "continue_run", fake_continue_run)


def test_cli_max_exchanges_reaches_orchestrator(tmp_path, monkeypatch):
    captured: dict = {}
    _patch_continue_run(monkeypatch, tmp_path, captured)
    res = runner.invoke(
        app, ["eval", "continue", str(tmp_path), "--max-exchanges", "2"]
    )
    assert res.exit_code == 0, res.output
    assert captured["max_exchanges"] == 2


def test_cli_max_exchanges_defaults_to_all_recorded(tmp_path, monkeypatch):
    captured: dict = {}
    _patch_continue_run(monkeypatch, tmp_path, captured)
    res = runner.invoke(app, ["eval", "continue", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert captured["max_exchanges"] is None


def test_cli_out_of_range_cut_exits_clean(tmp_path):
    folder = write_run_folder(tmp_path / "run", exchanges=_recorded(2))
    res = runner.invoke(app, ["eval", "continue", str(folder), "--max-exchanges", "99"])
    assert res.exit_code == 1
    assert "max_exchanges must be in 1..2" in res.output
    assert "Traceback (most recent call last)" not in res.output
