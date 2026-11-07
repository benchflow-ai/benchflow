"""BenchFlow's probes as a non-root verifier user never run the agent's
programs, in a real container (merge wave batch 32b, round-2 review A, M-1).

Round 1's M-1 fix kept the exec's PATH whenever the kit's RECORD existed.
That is root's kit route; an exec as another user carries that user's own
PATH, the image's, whose first folder here is the agent's. So the
``required_tools`` probe and the verifier-uid probe, run as the task's
non-root verifier user, ran the agent's ``sh`` and ``id``: a step from the
agent's user to the verifier's, and a forged answer. They pin the system
PATH now, as V3 did. Skipped without a Docker daemon.
"""

from __future__ import annotations

import pytest

from benchflow._utils.trusted_path import TrustedPathReport
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import SYSTEM_PATH
from tests.test_c8_protected_runtime_container import (  # noqa: F401
    _agent_turn,
    _dx,
    _marks,
    _setup,
    box,
)

VERIFIER = "verifier"


@pytest.mark.asyncio
async def test_probes_as_a_non_root_verifier_run_none_of_the_agents_programs(box):  # noqa: F811
    sandbox = await _setup(box)
    made = _dx(box, f"useradd -u 2001 -M {VERIFIER} || adduser -D -u 2001 {VERIFIER}")
    assert made.returncode == 0, made.stderr
    _agent_turn(box)
    assert _marks(box) == []

    probe = await sandbox.exec(
        lockdown._on_path('printf "%s\\n" "$PATH"; command -v id'),
        user=VERIFIER,
        timeout_sec=60,
    )
    assert probe.return_code == 0, probe.stderr
    assert probe.stdout.split() == [SYSTEM_PATH, "/usr/bin/id"]

    system = await sandbox.exec(
        lockdown._system_probe("echo probe-ran; id -u"), user=VERIFIER, timeout_sec=60
    )
    assert system.stdout.split() == ["probe-ran", "2001"], system.stderr

    assert await lockdown._verifier_uid(sandbox, VERIFIER) == "2001"

    # The required_tools contract: the probe answers for itself, so a tool
    # the trusted PATH lacks is still reported missing.
    verifier = type(
        "Verifier", (), {"user": VERIFIER, "required_tools": ["ls", "no-such-tool"]}
    )()
    report = TrustedPathReport(path=SYSTEM_PATH)
    with pytest.raises(RuntimeError, match="no-such-tool"):
        await lockdown._check_required_path_tools(sandbox, report, verifier)

    assert _marks(box) == []
