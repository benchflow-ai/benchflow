#!/usr/bin/env python3
"""Tests for the deterministic rubric grader + review pack.

Stdlib only. These run under plain ``python3`` with NO ``benchflow`` install:
the flat-fixture grading tests use only the lazy-import-light code paths, and
any test that needs a PRODUCTION rollout dir (which lazy-imports the benchflow
enforcers) is guarded by :data:`_HAVE_BENCHFLOW` and skipped when the import
fails.

Run either way::

    python3 tests/test_rubric_checks.py     # stdlib runner (no pytest needed)
    pytest tests/test_rubric_checks.py      # standard pytest discovery
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))
sys.path.insert(0, str(_REPO_ROOT / ".github" / "scripts"))

import rubric_checks  # noqa: E402

_FIXTURES = (
    _REPO_ROOT
    / ".agents"
    / "skills"
    / "benchflow-experiment-review"
    / "evals"
    / "files"
)

# Production-path tests need the benchflow enforcers (agent_judge ->
# benchflow.rewards.llm). Detect availability; skip those tests otherwise.
try:  # pragma: no cover - environment-dependent
    import benchflow  # noqa: F401

    _HAVE_BENCHFLOW = True
except Exception:  # pragma: no cover
    _HAVE_BENCHFLOW = False


class SkipTest(Exception):
    """Raised to skip a test under the stdlib runner / signal pytest.skip."""


def _grade(fixture: str) -> dict:
    return rubric_checks.grade_rollout(_FIXTURES / fixture)


def _gate(report: dict, gate_id: str) -> dict:
    for gate in report["gates"]:
        if gate["id"] == gate_id:
            return gate
    raise AssertionError(f"gate {gate_id} not in report")


# ------------------------------------------------------------------
# Single source of truth + schema sniff.
# ------------------------------------------------------------------


def test_rubric_gates_are_the_contract_table() -> None:
    ids = [g.id for g in rubric_checks.RUBRIC_GATES]
    # Every CONTRACT-A gate id is present exactly once.
    expected = {
        "R-REAL",
        "R-OUTCOME",
        "R-ARTIFACT",
        "R-TELEMETRY",
        "S-WITHSKILL",
        "S-NOSKILL",
        "V-TAMPER",
        "V-EMPTYPATCH",
        "V-LIFECYCLE",
        "V-ENVHARDEN",
        "V-REWARDHACK",
        "V-NETWORK",
        "P-REWARD",
        "P-SCHEMA",
        "C-ATTRIB",
        "P-PROV",
        "P-PATHS",
        "X-SLOTS",
    }
    assert set(ids) == expected, set(ids) ^ expected
    assert len(ids) == len(set(ids)), "duplicate gate id"
    # Enforcement tiers are constrained to the contract's four values.
    tiers = {g.enforcement for g in rubric_checks.RUBRIC_GATES}
    assert tiers <= {"deterministic", "quarantine", "codex", "residual"}, tiers
    # The non-deterministic gates must NOT be deterministic (never faked).
    for gate_id in ("V-LIFECYCLE", "V-ENVHARDEN", "V-REWARDHACK"):
        assert rubric_checks.GATES_BY_ID[gate_id].enforcement != "deterministic"


def test_schema_sniff_flat_vs_production() -> None:
    flat = {"reward": 1.0, "token_usage": {"input_tokens": 1}, "status": "pass"}
    prod = {"rewards": {"reward": 1.0}, "n_tool_calls": 3}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        assert rubric_checks._is_flat_fixture(flat, d) is True
        assert rubric_checks._is_flat_fixture(prod, d) is False
        # Top-level reward, no token_usage, but a sibling run_config.json -> flat.
        (d / "run_config.json").write_text("{}")
        assert (
            rubric_checks._is_flat_fixture({"reward": None, "status": "timeout"}, d)
            is True
        )


# ------------------------------------------------------------------
# Flat-fixture grading (the four eval fixtures).
# ------------------------------------------------------------------


def test_clean_pass_all_green() -> None:
    report = _grade("clean-pass")
    assert report["schema"] == "flat"
    assert report["deterministic_reject"] is False
    assert report["quarantines"] == []
    assert _gate(report, "R-REAL")["status"] == "pass"
    assert _gate(report, "R-OUTCOME")["status"] == "pass"
    assert _gate(report, "R-TELEMETRY")["status"] == "pass"
    assert _gate(report, "V-TAMPER")["status"] == "pass"
    assert _gate(report, "S-WITHSKILL")["status"] == "pass"


def test_reward_hack_is_deterministic_reject_via_tamper() -> None:
    # The reward-hack fixture edits verifier.py and writes results/score.json:
    # V-TAMPER (deterministic) must fire and the rollout must reject.
    report = _grade("reward-hack")
    assert report["schema"] == "flat"
    assert report["deterministic_reject"] is True
    tamper = _gate(report, "V-TAMPER")
    assert tamper["status"] == "fail"
    assert "verifier.py" in tamper["detail"] or "score.json" in tamper["detail"]


def test_no_skill_leak_fails_s_noskill() -> None:
    report = _grade("no-skill-leak")
    assert report["schema"] == "flat"
    s_noskill = _gate(report, "S-NOSKILL")
    assert s_noskill["status"] == "fail"
    # Leakage markers from the trajectory are surfaced.
    assert "invoke_skill" in s_noskill["detail"]
    assert report["deterministic_reject"] is True
    # S-WITHSKILL must be NA on a no-skill cell.
    assert _gate(report, "S-WITHSKILL")["status"] == "na"


def test_infra_timeout_telemetry_and_attribution() -> None:
    report = _grade("infra-timeout-missing-usage")
    assert report["schema"] == "flat"
    # Missing token usage -> R-TELEMETRY fail (deterministic reject).
    assert _gate(report, "R-TELEMETRY")["status"] == "fail"
    assert _gate(report, "R-REAL")["status"] == "fail"
    assert report["deterministic_reject"] is True
    # C-ATTRIB quarantines as experiment-fidelity (missing required resource).
    attrib = _gate(report, "C-ATTRIB")
    assert attrib["status"] == "quarantine"
    assert "experiment-fidelity" in attrib["detail"]
    assert any("C-ATTRIB" in q for q in report["quarantines"])


def test_non_deterministic_gates_are_never_faked() -> None:
    # On every fixture, V-LIFECYCLE/V-ENVHARDEN/V-REWARDHACK report as na markers
    # with their non-deterministic enforcement tier (never pass/fail).
    for fixture in ("clean-pass", "reward-hack", "no-skill-leak"):
        report = _grade(fixture)
        for gate_id in ("V-LIFECYCLE", "V-ENVHARDEN", "V-REWARDHACK"):
            gate = _gate(report, gate_id)
            assert gate["status"] == "na", (fixture, gate_id, gate)
            assert gate["enforcement"] in {"codex", "residual"}


# ------------------------------------------------------------------
# Capability attribution classifier (model-capability|experiment-fidelity|...).
# ------------------------------------------------------------------


def test_classify_capability_experiment_fidelity_on_infra_timeout() -> None:
    # The infra-timeout fixture (missing required env, unscored, no tokens) is an
    # experiment-fidelity failure, never a healthy model failure.
    ev = rubric_checks.load_evidence(_FIXTURES / "infra-timeout-missing-usage")
    attribution = rubric_checks.classify_capability(ev)
    assert attribution.label == "experiment-fidelity"
    assert any("required resource" in r for r in attribution.reasons)


def test_classify_capability_model_capability_on_scored_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "rollout"
        dest.mkdir()
        (dest / "result.json").write_text(
            json.dumps(
                {
                    "status": "fail",
                    "reward": 0.0,
                    "timing": {
                        "started_at": "2026-06-01T00:00:00Z",
                        "ended_at": "2026-06-01T00:03:00Z",
                        "duration_seconds": 180,
                    },
                    "token_usage": {"input_tokens": 800, "output_tokens": 100},
                    "tool_usage": {"bash": 3},
                }
            )
        )
        (dest / "run_config.json").write_text(json.dumps({"skill_mode": "no-skill"}))
        ev = rubric_checks.load_evidence(dest)
        attribution = rubric_checks.classify_capability(ev)
        assert attribution.label == "model-capability"


def test_classify_capability_not_applicable_on_pass() -> None:
    ev = rubric_checks.load_evidence(_FIXTURES / "clean-pass")
    assert rubric_checks.classify_capability(ev).label == "not-applicable"


# ------------------------------------------------------------------
# Network hardening (STATIC per-task config; Q3 SCOPE-GATED LANE).
# ------------------------------------------------------------------


def test_network_hardening_default_no_network_passes() -> None:
    gate_id, status, _ = rubric_checks.network_hardening({"network_mode": "no-network"})
    assert gate_id == "V-NETWORK"
    assert status == "pass"
    # An undeclared policy is NA (the runtime default applies).
    assert rubric_checks.network_hardening({})[1] == "na"


def test_network_hardening_allowlist_requires_hosts() -> None:
    empty = rubric_checks.network_hardening(
        {"network_mode": "allowlist", "allowed_hosts": []}
    )
    assert empty[1] == "fail"
    ok = rubric_checks.network_hardening(
        {"network_mode": "allowlist", "allowed_hosts": ["pubmed.ncbi.nlm.nih.gov"]}
    )
    assert ok[1] == "pass"


def test_network_hardening_public_is_blocker_on_verifier_sandbox_pr() -> None:
    # A bare public mode is a quarantine on an unrelated PR ...
    normal = rubric_checks.network_hardening({"network_mode": "public"})
    assert normal[1] == "quarantine"
    # ... but a hard fail (blocker) when the PR touches the isolation boundary.
    sandbox = rubric_checks.network_hardening(
        {"network_mode": "public"}, verifier_or_sandbox_pr=True
    )
    assert sandbox[1] == "fail"


# ------------------------------------------------------------------
# Pinned-baseline reward-band parity (subprocess wrapper; needs benchflow).
# ------------------------------------------------------------------


def test_parity_baseline_band_reports_fail_for_missing_roots() -> None:
    # With nonexistent roots the underlying checker FAILS closed (no matching
    # result.json), and the wrapper surfaces that as a fail rather than crashing.
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        result = rubric_checks.parity_baseline_band(empty, empty, ["weighted-gdp-calc"])
        assert result.status == "fail"


# ------------------------------------------------------------------
# Schema adapter maps a PRODUCTION (scenarios.synth_rollout) dir.
# ------------------------------------------------------------------


def test_adapter_maps_production_synth_rollout() -> None:
    if not _HAVE_BENCHFLOW:
        raise SkipTest("benchflow not installed; production path skipped")
    import scenarios

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "rollout"
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
        ev = rubric_checks.load_evidence(dest)
        assert ev.schema == "production"
        assert ev.reward == 1.0
        assert ev.n_tool_calls == 8
        assert ev.total_tokens == 120_000
        report = rubric_checks.grade_rollout(dest)
        # Realness + artifacts hold on a well-formed production rollout.
        assert _gate(report, "R-REAL")["status"] == "pass"
        assert _gate(report, "R-ARTIFACT")["status"] == "pass"
        assert _gate(report, "V-TAMPER")["status"] == "pass"


def test_production_timing_read_from_toplevel_started_finished() -> None:
    # Regression (e2e bug): a real production result.json records timing at the
    # TOP LEVEL (started_at / finished_at) plus per-phase seconds under `timing`
    # (total), NOT under timing.{started_at,ended_at} (the flat-fixture shape).
    # Reading the flat shape left timing empty and false-failed R-TELEMETRY on a
    # reward-1.0, 7-minute real rollout.
    if not _HAVE_BENCHFLOW:
        raise SkipTest("benchflow not installed; production path skipped")
    import json as _json

    import scenarios

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "rollout"
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
        )
        # Stamp the production timing shape onto result.json (synth omits it).
        rj = dest / "result.json"
        result = _json.loads(rj.read_text())
        result["started_at"] = "2026-06-18 04:37:45.327504"
        result["finished_at"] = "2026-06-18 04:44:47.885050"
        result["timing"] = {"agent_execution": 150.1, "verifier": 15.9, "total": 422.6}
        rj.write_text(_json.dumps(result))

        ev = rubric_checks.load_evidence(dest)
        assert ev.started_at and (ev.ended_at or ev.duration_seconds)
        report = rubric_checks.grade_rollout(dest)
        assert _gate(report, "R-TELEMETRY")["status"] == "pass"


def test_adapter_flat_and_production_agree_on_realness() -> None:
    # The flat clean-pass fixture and an equivalent production synth rollout both
    # normalize to a REAL measurement (same R-REAL verdict from one adapter).
    flat = _grade("clean-pass")
    assert _gate(flat, "R-REAL")["status"] == "pass"
    if not _HAVE_BENCHFLOW:
        raise SkipTest("benchflow not installed; production half skipped")
    import scenarios

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "rollout"
        scenarios.synth_rollout(dest, reward=1.0, n_tool_calls=2, total_tokens=1440)
        prod = rubric_checks.grade_rollout(dest)
        assert _gate(prod, "R-REAL")["status"] == "pass"


# NOTE: the review-pack (slot classification, verdict ladder, full layout) is
# graded by ``build_integration_review_pack`` and tested in
# ``tests/test_build_review_pack.py``. This file is scoped to the rubric grader
# (RUBRIC_GATES + the per-rollout predicates) it owns.


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
