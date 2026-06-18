#!/usr/bin/env python3
"""Tests for the deterministic review-pack builder (build_integration_review_pack).

Stdlib only. These run under plain ``python3`` with NO ``benchflow`` install:
the review-pack is built over flat-fixture rollouts (the grader lazy-imports the
benchflow enforcers only for a PRODUCTION rollout). A test that needs a real
production rollout dir is guarded by :data:`_HAVE_BENCHFLOW` and skipped.

Run either way::

    python3 tests/test_build_review_pack.py     # stdlib runner (no pytest)
    pytest tests/test_build_review_pack.py       # standard pytest discovery
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))
sys.path.insert(0, str(_REPO_ROOT / ".github" / "scripts"))

import build_integration_review_pack as pack_mod  # noqa: E402

try:  # pragma: no cover - environment-dependent
    import benchflow  # noqa: F401

    _HAVE_BENCHFLOW = True
except Exception:  # pragma: no cover
    _HAVE_BENCHFLOW = False

_HEAD = "abc1234def5678"


class SkipTest(Exception):
    """Raised to skip a test under the stdlib runner / signal pytest.skip."""


# ------------------------------------------------------------------
# Fixtures: hand-built flat rollouts + a matrix plan.
# ------------------------------------------------------------------


def _write_flat_rollout(
    dest: Path,
    *,
    task: str,
    agent: str,
    sandbox: str,
    skill_mode: str,
    head_sha: str,
    reward: float = 1.0,
    status: str = "pass",
    with_usage: bool = True,
    error: str | None = None,
    required_env: list[str] | None = None,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "status": status,
        "reward": reward,
        "verifier_started_after_agent": True,
        "timing": {
            "started_at": "2026-06-01T00:00:00Z",
            "ended_at": "2026-06-01T00:05:00Z" if status != "timeout" else None,
            "duration_seconds": 300,
        },
        "tool_usage": {"bash": 2},
    }
    if with_usage:
        result["token_usage"] = {"input_tokens": 1000, "output_tokens": 200}
    if error is not None:
        result["error"] = error
    (dest / "result.json").write_text(json.dumps(result))
    cfg: dict = {
        "task_id": task,
        "harness": agent,
        "scenario": "agent_matrix",
        "sandbox": sandbox,
        "skill_mode": skill_mode,
        "source_ref": head_sha,
        "head_sha": head_sha,
    }
    if required_env is not None:
        cfg["required_env"] = required_env
    (dest / "run_config.json").write_text(json.dumps(cfg))


def _cell(task: str, agent: str, **kw) -> dict:
    base = {
        "id": kw.get(
            "id",
            f"{task}-{kw.get('sandbox', 'daytona')}-"
            f"{kw.get('skill_mode', 'no-skill')}-{agent}",
        ),
        "task": task,
        "agent": agent,
        "model": kw.get("model"),
        "sandbox": kw.get("sandbox", "daytona"),
        "skill_mode": kw.get("skill_mode", "no-skill"),
        "network_mode": kw.get("network_mode", "default-off"),
        "audit_skills": kw.get("audit_skills", False),
        "expect_reward": kw.get("expect_reward", "any"),
        "scenario": kw.get("scenario", "agent_matrix"),
    }
    return base


def _plan(head_sha: str, matrix: list[dict], **kw) -> dict:
    return {
        "schema_version": "1",
        "head_sha": head_sha,
        "base_ref": "main",
        "scope": kw.get("scope", "high-3"),
        "buckets": kw.get("buckets", ["agents"]),
        "trust_boundary": kw.get("trust_boundary", False),
        "network_lane": kw.get("network_lane", False),
        "baseline": "pinned",
        "source_sha": kw.get("source_sha"),
        "matrix": matrix,
        "residual_risk": kw.get("residual_risk", []),
        "rejected_overflow": None,
    }


# ------------------------------------------------------------------
# Matrix normalization (SPEC schema + legacy planner cells schema).
# ------------------------------------------------------------------


def test_normalize_cell_spec_schema() -> None:
    cell = pack_mod.normalize_cell(_cell("weighted-gdp-calc", "openhands"))
    assert cell.task == "weighted-gdp-calc"
    assert cell.agent == "openhands"
    assert cell.network_mode == "default-off"
    # The single task is carried into ``include`` for cross-schema matching.
    assert cell.include == ("weighted-gdp-calc",)


def test_matrix_cells_accepts_legacy_cells_key() -> None:
    plan = {
        "head_sha": _HEAD,
        "cells": [
            {
                "id": "matrix-openhands",
                "scenario": "agent_matrix",
                "agent": "openhands",
                "sandbox": "daytona",
                "skill_mode": "no-skill",
                "include": ["weighted-gdp-calc"],
            }
        ],
    }
    cells = pack_mod._matrix_cells(plan)
    assert len(cells) == 1
    assert cells[0].task == "weighted-gdp-calc"
    assert cells[0].include == ("weighted-gdp-calc",)


# ------------------------------------------------------------------
# Slot classification + verdict (X-SLOTS / FAIL-CLOSED).
# ------------------------------------------------------------------


def test_review_pack_missing_slot_is_not_mergeable() -> None:
    matrix = [
        _cell("weighted-gdp-calc", "openhands"),
        _cell("weighted-gdp-calc", "gemini"),
    ]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        # Produce ONLY the openhands cell; the gemini slot is missing.
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha=_HEAD,
        )
        review = pack_mod.build_review(plan, arts, None)
        verdict = review["verdict"]
        assert verdict.verdict == pack_mod.VERDICT_NOT_MERGEABLE
        assert any("missing slot" in b for b in verdict.blockers)
        statuses = {s.cell_id: s.status for s in review["slots"]}
        assert statuses["weighted-gdp-calc-daytona-no-skill-openhands"] == "healthy"
        assert statuses["weighted-gdp-calc-daytona-no-skill-gemini"] == "missing"


def test_review_pack_full_coverage_is_mergeable() -> None:
    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha=_HEAD,
        )
        review = pack_mod.build_review(plan, arts, None)
        verdict = review["verdict"]
        assert verdict.verdict in pack_mod._OK_VERDICTS
        assert verdict.blockers == []


def test_review_pack_stale_sha_is_flagged_not_mergeable() -> None:
    # Stale = the rollout's TASK-SOURCE sha differs from the plan's pinned
    # task-source sha (NOT the benchflow head_sha).
    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan("NEWSHA0000000", matrix, source_sha="PINNEDSOURCE0000")
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha="OLDSOURCE1111111",  # rollout task-source sha != pinned
        )
        review = pack_mod.build_review(plan, arts, None)
        assert review["slots"][0].status == "stale"
        assert review["verdict"].verdict == pack_mod.VERDICT_NOT_MERGEABLE
        assert any("stale slot" in b for b in review["verdict"].blockers)


def test_review_pack_task_source_sha_not_compared_to_benchflow_head() -> None:
    # Regression (e2e bug): a healthy rollout whose recorded task-source sha
    # differs from the benchflow head_sha must NOT be marked stale when no
    # task-source pin is given. head_sha is the benchflow commit, unrelated to
    # the skillsbench task-source sha; comparing them flagged every real rollout.
    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan("BENCHFLOWHEAD999", matrix)  # no source_sha pinned
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha="SKILLSBENCHSRC123",  # task-source sha != benchflow head
        )
        review = pack_mod.build_review(plan, arts, None)
        assert review["slots"][0].status != "stale"
        assert not any("stale" in b for b in review["verdict"].blockers)


def test_review_pack_infra_timeout_is_capability_attributed_quarantine() -> None:
    # An unscored infra-timeout rollout: R-TELEMETRY/R-REAL fail -> unhealthy slot
    # (deterministic reject), and C-ATTRIB classifies experiment-fidelity. The
    # cell is a blocker (unhealthy) and the attribution is recorded in the pack.
    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha=_HEAD,
            reward=None,  # type: ignore[arg-type]
            status="timeout",
            with_usage=False,
            error="Missing REQUIRED_TASK_API_KEY prevented task execution.",
            required_env=["REQUIRED_TASK_API_KEY"],
        )
        review = pack_mod.build_review(plan, arts, None)
        slot = review["slots"][0]
        assert slot.status == "unhealthy"
        assert review["verdict"].verdict == pack_mod.VERDICT_NOT_MERGEABLE
        # The agent_judge_summary attaches the experiment-fidelity attribution.
        rows = pack_mod.agent_judge_summary(review["slots"])
        attribution = rows[0]["attribution"]
        assert attribution is not None
        assert attribution["label"] == "experiment-fidelity"


# ------------------------------------------------------------------
# Full review-pack/ layout on disk + the CLI verdict contract.
# ------------------------------------------------------------------


def test_write_pack_emits_full_layout() -> None:
    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arts = tmp / "arts"
        arts.mkdir()
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha=_HEAD,
        )
        out = tmp / "review-pack"
        review = pack_mod.build_review(plan, arts, None)
        pack_mod.write_pack(out, plan, arts, None, review)
        expected_files = {
            "manifest.json",
            "matrix_expected.json",
            "matrix_observed.json",
            "metrics.json",
            "agent_judge_summary.json",
            "skill_catalog_summary.json",
            "parity_summary.json",
            "hardening_summary.md",
            "red_flags.md",
            "verdict.md",
        }
        present = {p.name for p in out.iterdir()}
        assert expected_files <= present, expected_files - present
        assert (out / "rollouts" / "index.json").is_file()
        # verdict.md carries the SKILL-order sections + user-facing verdict.
        verdict_md = (out / "verdict.md").read_text()
        assert "# Verdict" in verdict_md
        for section in (
            "Blockers",
            "Coverage",
            "Evidence",
            "Residual risk",
            "Required reruns",
        ):
            assert f"## {section}" in verdict_md
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["scope"] == "high-3"
        assert manifest["head_sha"] == _HEAD


def test_cli_prints_verdict_and_exit_code() -> None:
    matrix = [
        _cell("weighted-gdp-calc", "openhands"),
        _cell("weighted-gdp-calc", "gemini"),
    ]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arts = tmp / "arts"
        arts.mkdir()
        # Missing gemini -> not mergeable -> nonzero exit.
        _write_flat_rollout(
            arts / "openhands",
            task="weighted-gdp-calc",
            agent="openhands",
            sandbox="daytona",
            skill_mode="no-skill",
            head_sha=_HEAD,
        )
        plan_path = tmp / "plan.json"
        plan_path.write_text(json.dumps(plan))
        rc = pack_mod.main(
            [
                "--matrix",
                str(plan_path),
                "--artifacts",
                str(arts),
                "--out",
                str(tmp / "rp"),
            ]
        )
        assert rc == 1
        verdict_md = (tmp / "rp" / "verdict.md").read_text()
        assert pack_mod.VERDICT_NOT_MERGEABLE in verdict_md


def test_cli_fails_closed_on_unreadable_matrix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp) / "arts"
        arts.mkdir()
        rc = pack_mod.main(["--matrix", "{not valid json", "--artifacts", str(arts)])
        # FAIL CLOSED: an unreadable plan is not mergeable (nonzero exit).
        assert rc == 1


# ------------------------------------------------------------------
# Hardening summary surfaces the network policy (public-mode blocker).
# ------------------------------------------------------------------


def test_hardening_summary_flags_public_network_on_sandbox_pr() -> None:
    matrix = [
        _cell(
            "jax-computing-basics",
            "openhands",
            network_mode="public",
            id="jax-public",
        )
    ]
    cells = [pack_mod.normalize_cell(c) for c in matrix]
    md = pack_mod.hardening_summary_md(
        slots=[], cells=cells, verifier_or_sandbox_pr=True
    )
    assert "V-NETWORK=fail" in md
    assert "jax-public" in md


def test_hardening_summary_allowlist_variant_is_hardened() -> None:
    raw = _cell("citation-check", "openhands", network_mode="allowlist", id="cit-allow")
    raw["allowed_hosts"] = ["pubmed.ncbi.nlm.nih.gov", "scholar.google.com"]
    cells = [pack_mod.normalize_cell(raw)]
    md = pack_mod.hardening_summary_md(
        slots=[], cells=cells, verifier_or_sandbox_pr=False
    )
    assert "V-NETWORK=pass" in md


# ------------------------------------------------------------------
# Production rollout via scenarios.synth_rollout (needs benchflow).
# ------------------------------------------------------------------


def test_review_pack_over_synth_production_rollout() -> None:
    if not _HAVE_BENCHFLOW:
        raise SkipTest("benchflow not installed; production synth path skipped")
    import scenarios

    matrix = [_cell("weighted-gdp-calc", "openhands")]
    plan = _plan(_HEAD, matrix)
    with tempfile.TemporaryDirectory() as tmp:
        arts = Path(tmp)
        dest = arts / "openhands"
        scenarios.synth_rollout(
            dest,
            task_name="weighted-gdp-calc",
            reward=1.0,
            n_tool_calls=8,
            total_tokens=120_000,
            trajectory=[
                {
                    "phase": "agent",
                    "type": "tool_call",
                    "tool_calls": [
                        {"name": "bash", "arguments": {"command": "python solve.py"}}
                    ],
                }
            ],
            atif={"schema_version": "ATIF-v1.0", "steps": [{"source": "agent"}]},
            adp_lines=[{"step": 0}],
        )
        # synth_rollout writes a production result.json (no run_config); add the
        # config dims the slot matcher + freshness check need, plus the timing
        # the R-TELEMETRY gate reads from the run config.
        (dest / "run_config.json").write_text(
            json.dumps(
                {
                    "task_id": "weighted-gdp-calc",
                    "harness": "openhands",
                    "scenario": "agent_matrix",
                    "sandbox": "daytona",
                    "skill_mode": "no-skill",
                    "head_sha": _HEAD,
                    "source_ref": _HEAD,
                    "started_at": "2026-06-01T00:00:00Z",
                    "ended_at": "2026-06-01T00:05:00Z",
                }
            )
        )
        review = pack_mod.build_review(plan, arts, None)
        assert review["slots"][0].status == "healthy", review["slots"][0].detail
        assert review["verdict"].verdict in pack_mod._OK_VERDICTS


# ------------------------------------------------------------------
# Stdlib runner (so this file passes under plain python3, no pytest).
# ------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # report any error
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
