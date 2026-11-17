"""An optional program is left missing, not stubbed, by PATH hardening.

Guards the fix against V3-CONFIRM-6's design (548ac115), where a
program found only in a folder the verifier's PATH leaves out got a refusal
stub even when the task listed it in ``[verifier] optional_tools``.
Terminal-Bench 4.1's atrx-vep-crispr looks ``vep`` up with ``shutil.which``
and falls back to ``perl /app/.../vep`` when it is absent; the lookup found
the stub, the verifier ran it, and the trial was refused with ``vep`` in
``refused_programs_run``. An optional program now gets no stub: its folder is
still left out, a lookup finds nothing, and the task takes its own fallback.
A required program that is missing still refuses, and a program the task
does not list still gets its stub.

Real Docker containers (the fixtures of test_verifier_environment_refusals).
"""

from __future__ import annotations

import pytest

from tests.test_verifier_environment_refusals import (  # noqa: F401
    _sh,
    _verify,
    _write,
    box,
    image,
)


def _plant(box) -> None:  # noqa: F811
    # The agent-writable folder first on the image's PATH holds the program.
    _write(box, "/opt/probe/bin/vep", "#!/bin/sh\necho agents-vep\nexit 0\n")
    _sh(box, "chmod 755 /opt/probe/bin/vep")


def _dropped(record) -> list[str]:
    return [entry["folder"] for entry in record["dropped"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["command -v vep", "which vep"])
async def test_an_optional_program_is_missing_and_the_task_falls_back(
    box,  # noqa: F811
    tmp_path,
    lookup,
):
    _plant(box)
    rewards, error, record = await _verify(
        box,
        tmp_path,
        f"#!/bin/sh\nif {lookup} >/dev/null 2>&1; then r=0; else r=1; fi\n"
        "echo $r > /logs/verifier/reward.txt\n",
        optional_tools=("vep",),
    )
    assert (rewards, error) == ({"reward": 1.0}, None)
    assert record["refused_programs_run"] == []
    # The agent's program stays out of reach: its folder is still dropped.
    assert "/opt/probe/bin" in _dropped(record)


@pytest.mark.asyncio
async def test_a_program_the_task_does_not_list_is_still_refused(
    box,  # noqa: F811
    tmp_path,
):
    _plant(box)
    rewards, error, record = await _verify(
        box,
        tmp_path,
        "#!/bin/sh\nvep >/dev/null 2>&1\necho 1 > /logs/verifier/reward.txt\n",
    )
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert record["refused_programs_run"] == [
        {"program": "vep", "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
async def test_a_missing_required_program_still_refuses(
    box,  # noqa: F811
    tmp_path,
):
    _plant(box)
    rewards, error, _ = await _verify(
        box,
        tmp_path,
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
        required_tools=("vep",),
        optional_tools=("other",),
    )
    assert rewards is None
    assert error is not None and "vep" in error
