"""Integration suite — the pytest lane home (ADR 0001).

Each release-blocker lane in suites/release.yaml resolves to a `lane_<id>`
marker here; `run_suite.py --run-lane <id>` runs `pytest -m lane_<id>`.
Assertions live in scenarios.py.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios  # noqa: E402

_HAVE_DOCKER = shutil.which("docker") is not None or bool(
    os.environ.get("BENCHFLOW_IT_DOCKER")
)


@pytest.mark.lane
@pytest.mark.lane_network_mode_enforcement
@pytest.mark.skipif(not _HAVE_DOCKER, reason="docker not available")
def test_network_mode_allowlist_conformance():
    """allowlist permits only listed hosts; everything else denied (ENG-265)."""
    issues = scenarios.egress_conformance_issues(
        ["example.com"], blocked_hosts=["www.cloudflare.com"]
    )
    assert issues == [], f"egress allowlist not conformant: {issues}"
