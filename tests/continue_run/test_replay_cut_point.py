"""Replay cut-point tests.

Guards the replay cut-point ("feat(continue): replay cut-point — replay first
K exchanges, then go live"; docs/rollout-branching-rfc.md WS-3;
FrontierPhysics#73). PR number to be added on submission.

``max_exchanges`` replays at most the first K recorded exchanges, then switches
the proxy to live passthrough exactly as if the recording had ended there. The
router exposes cut-point accounting (``n_replayed_exchanges`` +
``cut_point_digest``), a cut continue-run's ``source_provenance`` gains a
``cut_point`` block (configured at build time, reconciled with the served
counts post-run in host proxy mode), and ``stage_tags`` + ``cut_stage`` name a
cut by recorded stage — ``stage_tags[stage]`` is the 1-based count of
exchanges that had completed when the stage closed. Unit tests against the
existing continue_run test doubles — no Docker, no API keys.
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
    served_cut_point,
    stitched_trajectory_lines,
    summarize_llm_trajectory_usage,
    update_continued_metadata,
    write_stitched_trajectory,
)
from benchflow.continue_run.replay_proxy import (
    ReplayCutPointError,
    ReplayRouter,
    request_body_digest,
    validate_max_exchanges,
)
from benchflow.continue_run.run_folder import load_run_folder

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
    # config-time blocks are computed from the recording, and say so — sandbox
    # proxy mode (truncated upload, no live router to read back) keeps this
    # basis in the final artifacts.
    assert cut["accounting"] == "configured"
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
    assert cut["accounting"] == "configured"


# ── host-mode reconciliation: the served cut_point block ──────────────────


def test_served_cut_point_records_what_the_router_actually_served():
    """A run that went live before reaching the configured cut is visible.

    The router's served counters had no production readers — the block now
    records n_replayed_exchanges / cut_point_digest as observed (accounting
    "served") plus the requested configured_max_exchanges, so a
    served-vs-configured divergence is detectable in artifacts.
    """
    recorded = _recorded(3)
    router = ReplayRouter(
        recorded, live_forwarder=lambda req: completion(content="L"), max_exchanges=3
    )
    # the agent only ever replays 2 of the configured 3 exchanges
    for i in range(2):
        router.next_response(_request(i))

    block = served_cut_point(router, configured_max_exchanges=3)

    assert block == {
        "n_replayed_exchanges": 2,
        "cut_point_digest": request_body_digest(recorded[1].request.body),
        "accounting": "served",
        "configured_max_exchanges": 3,
    }


def test_served_cut_point_natural_end_and_stage_shape():
    recorded = _recorded(2)
    router = ReplayRouter(recorded, live_forwarder=lambda req: completion(content="L"))
    for i in range(3):
        router.next_response(_request(i))

    block = served_cut_point(router, branch_stage="post-research")

    assert block == {
        "n_replayed_exchanges": 2,
        "cut_point_digest": request_body_digest(recorded[1].request.body),
        "accounting": "served",
        "branch_stage": "post-research",
    }


def test_update_continued_metadata_reconciles_cut_point_in_both_files(tmp_path):
    """Host mode's post-run patch replaces the configured block with served."""
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    configured = {
        "n_replayed_exchanges": 3,
        "cut_point_digest": "sha256:configured",
        "accounting": "configured",
    }
    (rollout_dir / "config.json").write_text(
        json.dumps({"model": None, "source": {"cut_point": configured}})
    )
    (rollout_dir / "result.json").write_text(
        json.dumps({"model": None, "source": {"cut_point": configured}})
    )
    traj = tmp_path / "llm_trajectory.jsonl"
    traj.write_text("")
    served = {
        "n_replayed_exchanges": 2,
        "cut_point_digest": "sha256:served",
        "accounting": "served",
        "configured_max_exchanges": 3,
    }

    update_continued_metadata(
        rollout_dir,
        live_model="live-model",
        usage=summarize_llm_trajectory_usage(traj, n_recorded=2),
        environment="docker",
        cut_point=served,
    )

    config = json.loads((rollout_dir / "config.json").read_text())
    result = json.loads((rollout_dir / "result.json").read_text())
    assert config["source"]["cut_point"] == served
    assert result["source"]["cut_point"] == served


# ── stage-named cuts ──────────────────────────────────────────────────────


def test_stage_named_cut_resolves_and_records_branch_stage():
    """stage_tags[stage] = the 1-based count of exchanges completed when the
    stage closed; a cut at that stage replays exactly that many."""
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
    with pytest.raises(ReplayCutPointError, match="env-ready, post-research"):
        resolve_cut_point(None, stage_tags=tags, cut_stage="pre-verify")


def test_cut_stage_without_stage_tags_fails_closed():
    with pytest.raises(ReplayCutPointError, match="stage_tags"):
        resolve_cut_point(None, cut_stage="post-research")


def test_cut_stage_and_max_exchanges_are_exclusive():
    with pytest.raises(ReplayCutPointError, match="not both"):
        resolve_cut_point(2, stage_tags={"env-ready": 1}, cut_stage="env-ready")


@pytest.mark.parametrize("bad_tag", [0, -3])
def test_stage_tag_below_one_fails_closed(bad_tag):
    """A stage tag is a 1-based completed-exchange count — 0/negative is a
    caller bug, rejected as ReplayCutPointError at resolve time."""
    with pytest.raises(ReplayCutPointError, match="1-based"):
        resolve_cut_point(
            None, stage_tags={"env-ready": bad_tag}, cut_stage="env-ready"
        )


def test_resolve_cut_point_passthrough_without_stage():
    assert resolve_cut_point(None) == (None, None)
    assert resolve_cut_point(4) == (4, None)


# ── stitched trajectory: prefix = the K parsed (replayed) exchanges ───────


def test_stitched_prefix_truncated_at_cut():
    recorded_lines = ['{"a": 1}', '{"b": 2}', '{"c": 3}']
    lines = stitched_trajectory_lines(
        recorded_lines, [exchange(completion(content="L"))], max_recorded=2
    )
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
    last = json.loads(lines[2])
    assert last["response"]["body"]["choices"][0]["message"]["content"] == "L"


def test_stitched_prefix_counts_parsed_exchanges_not_raw_lines(tmp_path):
    """A malformed recorded line is never replayed — and never stitched.

    Replay counts *parsed* exchanges (load_llm_exchanges skips malformed
    lines), so the stitched prefix must be the raw lines of the first K
    parsed exchanges. Truncating by raw file-line index used to embed the
    malformed (never-replayed) line and drop a replayed one, and shifted the
    recorded/live usage-accounting boundary.
    """
    recorded = _recorded(3)
    folder = write_run_folder(tmp_path / "run", exchanges=recorded)
    traj = folder / "trajectory" / "llm_trajectory.jsonl"
    good_lines = traj.read_text().splitlines()
    traj.write_text(
        "\n".join([good_lines[0], "this is not json", *good_lines[1:]]) + "\n"
    )
    run = load_run_folder(folder)
    assert run.n_recorded_exchanges == 3  # malformed line skipped at load

    live = [exchange(completion(content="LIVE"))]
    stitched = write_stitched_trajectory(
        tmp_path / "rollout", run.exchange_lines, live, max_recorded=2
    )

    lines = stitched.read_text().splitlines()
    # exactly the 2 replayed exchanges' verbatim lines + the live suffix
    assert len(lines) == 3
    assert lines[0] == good_lines[0]
    assert lines[1] == good_lines[1]
    assert "this is not json" not in stitched.read_text()
    assert (
        json.loads(lines[2])["response"]["body"]["choices"][0]["message"]["content"]
        == "LIVE"
    )
    # the recorded/live usage split lands on the true replay boundary
    usage = summarize_llm_trajectory_usage(stitched, n_recorded=2)
    assert usage.recorded_total_tokens == 4  # 2 replayed exchanges x 2 tokens
    assert usage.live_total_tokens == 2  # 1 live exchange x 2 tokens


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
