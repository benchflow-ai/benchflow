"""test.yml's jobs that run BenchFlow's Docker tests build the pinned bfx, and
the three the release waits on run on the release gate's Docker.

0.10.0rc4's tag run (44180275236) failed before any publish step: 118 unit
tests per Python, 56 deterministic-integration tests and the grading-integrity
job. A checkout has no bfx (git ignores src/benchflow/_bfx/bin), so every model
agent's Docker trial was refused before its turn (integrity_unavailable); and
on the runner image's Docker CE 28.0.4 with Compose 2.38.2 the tests that
build a Compose service, or check the network policy at a sandbox's start,
failed too. The release gate's VMs had bfx and the gate's Docker."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
GATE_DOCKER_JOBS = ("unit", "deterministic-integration", "grading-integrity")
BFX_JOBS = (*GATE_DOCKER_JOBS, "unit-latest-os")


def _runs(job: str) -> list[str]:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    return [step.get("run") or "" for step in jobs[job]["steps"]]


def _first(runs: list[str], needle: str) -> int:
    hits = [i for i, run in enumerate(runs) if needle in run]
    assert hits, needle
    return hits[0]


def test_every_docker_test_job_builds_the_pinned_bfx_before_its_tests():
    for job in BFX_JOBS:
        runs = _runs(job)
        build = _first(runs, "scripts/build_bfx.sh")
        assert "BFX_REQUIRE_PIN=1" in runs[build], job
        assert (
            _first(runs, "rustup toolchain install") < build < _first(runs, "pytest")
        ), job


def test_no_job_runs_pytest_without_bfx():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    without = sorted(
        name
        for name, job in jobs.items()
        if any("pytest" in (step.get("run") or "") for step in job.get("steps", []))
        and not any(
            "build_bfx.sh" in (step.get("run") or "") for step in job.get("steps", [])
        )
    )
    assert without == [], without


def test_the_live_smoke_rollout_builds_the_pinned_bfx_from_the_pr_head():
    """Guards EN-925 (PR #1251): integration-light's rollout-smoke had no bfx,
    so its one model-agent trial was refused before the turn on every internal
    PR from 11-17 (PR #1244's run 44199910131, PR #1250's 44657087309)."""
    job = yaml.safe_load((WORKFLOW.parent / "integration-light.yml").read_text())[
        "jobs"
    ]["rollout-smoke"]
    runs = [step.get("run") or "" for step in job["steps"]]
    overlay = _first(runs, 'git checkout "$HEAD_SHA" -- src/benchflow')
    assert 'git checkout "$HEAD_SHA" -- bfx' in runs[overlay]
    build = _first(runs, "scripts/build_bfx.sh")
    assert "BFX_REQUIRE_PIN=1" in runs[build]
    assert (
        overlay
        < _first(runs, "rustup toolchain install")
        < build
        < _first(runs, "bench eval run")
    )


def test_the_jobs_the_release_waits_on_run_on_the_gates_docker():
    gate = (ROOT / "tools" / "release_gate" / "bootstrap_vm.sh").read_text()
    packages = (
        re.search(r"^UBUNTU_PACKAGES=\(\n(.*?)^\)", gate, re.M | re.S).group(1).split()
    )
    versions = {
        name: version for name, _, version in (p.partition("=") for p in packages)
    }
    for job in GATE_DOCKER_JOBS:
        runs = _runs(job)
        docker = _first(runs, "tools/release_gate/bootstrap_vm.sh")
        assert "UBUNTU_PACKAGES" in runs[docker], job
        assert docker < _first(runs, "pytest"), job
        # it checks it got them: the engine and the Compose build the direct path is proven on
        assert versions["docker.io"].split("-")[0] in runs[docker], job
        assert versions["docker-compose-v2"] in runs[docker], job
    assert all("bootstrap_vm.sh" not in run for run in _runs("unit-latest-os"))
