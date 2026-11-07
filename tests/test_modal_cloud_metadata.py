"""Cloud metadata on Modal: proven unreachable instead of blocked.

Before this, a non-root agent on an open network was stopped on every Modal
trial: the block's iptables rules cannot go in there (root has no NET_ADMIN,
"not_installed"), although a Modal sandbox has no route to any metadata
endpoint. Measured on 2026-10-26 from a public-network Modal sandbox, as root
and as uid 1000: every IPv4 endpoint timed out on ports 80 and 443, every
IPv6 one had no route (ENETUNREACH), and 1.1.1.1:443 connected. A sandbox
class that declares ``cloud_metadata_unreachable`` now gets that proven each
trial by a connect to every endpoint, recorded as ``not_reachable``; any
answer, a missing tool or a failed exec records ``unproven``, which stops the
trial. Docker and Daytona are unaffected: they declare nothing.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.cloud_metadata import (
    CLOUD_METADATA_V4,
    CLOUD_METADATA_V6,
    NOT_REACHABLE,
)
from benchflow.sandbox.lockdown import (
    _cloud_metadata_reach_cmd,
    _cloud_metadata_reach_results,
    _reach_target,
    block_cloud_metadata,
)
from tests.receipt_fakes import receipt_answer

PORTS = (80, 443)


def _report(result_for) -> str:
    return "".join(
        f"BF metadata-reach {address} {port} {result_for(address, port)}\n"
        for address in (*CLOUD_METADATA_V4, *CLOUD_METADATA_V6)
        for port in PORTS
    )


def _measured(address, port):
    """What the 2026-10-26 Modal measurement saw."""
    return "unreachable" if ":" in address else "timeout"


class _Unreachable:
    """A provider sandbox that declares no route to the endpoints (Modal)."""

    cloud_metadata_unreachable = True
    cloud_metadata_provider_managed = True
    supports_system_shell = True

    def __init__(self, stdout: str, code: int = 0) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.stdout = stdout
        self.code = code

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return receipt_answer(
            command,
            lambda read: ExecResult(
                stdout=self.stdout, stderr="", return_code=self.code
            ),
        )


def test_the_reach_command_classifies_each_connect(tmp_path):
    """The shell text itself, with a planted ``timeout`` standing in for the
    network: a timeout, no route, a refusal and an open port."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "timeout").write_text(
        "#!/bin/sh\n"
        "# argv: -s USR2 <secs> bash -c <wrapper> bf-probe <address>/<port>\n"
        'case "$8" in\n'
        "  *169.254.169.254/80*) exit 0;;\n"
        "  *168.63.129.16/443*) echo 'bash: connect: Connection refused' >&2; exit 1;;\n"
        "  *100.100.100.200/80*) echo 'bash: connect: No route to host' >&2; exit 1;;\n"
        "  *:*) echo 'bash: connect: Network is unreachable' >&2; exit 1;;\n"
        "  *) echo 'BF metadata-deadline expired' >&2; exit 124;;\n"
        "esac\n"
    )
    (stubs / "timeout").chmod(0o755)
    ran = subprocess.run(
        ["/bin/sh", "-c", _cloud_metadata_reach_cmd()],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stderr
    probes = _cloud_metadata_reach_results(ran.stdout)
    assert probes is not None, ran.stdout
    assert len(probes) == 2 * (len(CLOUD_METADATA_V4) + len(CLOUD_METADATA_V6))
    assert probes["169.254.169.254:80"] == "reached"
    assert probes["168.63.129.16:443"] == "refused"
    assert probes["100.100.100.200:80"] == "unreachable"
    assert probes["[fd00:ec2::254]:443"] == "unreachable"
    assert probes["169.254.170.2:80"] == "timeout"


def test_an_incomplete_report_is_not_a_proof():
    full = _report(_measured)
    assert _cloud_metadata_reach_results(full) is not None
    assert _cloud_metadata_reach_results("\n".join(full.splitlines()[:-1])) is None
    assert _cloud_metadata_reach_results("BF metadata-reach skipped\n") is None


@pytest.mark.asyncio
async def test_no_answer_from_any_endpoint_is_not_reachable():
    env = _Unreachable(_report(_measured))
    record = await block_cloud_metadata(env, "agent")
    assert record["state"] == NOT_REACHABLE == "not_reachable"
    assert "no block is needed" in record["why"]
    assert record["probes"]["169.254.169.254:80"] == "timeout"
    assert record["probes"][_reach_target("fd00:ec2::254", 80)] == "unreachable"
    assert "root_agent" not in record
    # One exec, as root, through the agent's own launch (its privilege drop).
    [(command, kwargs)] = env.calls
    assert kwargs["user"] == "root"
    assert "setpriv" in command and "--reuid=agent" in command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_for, why",
    [
        (
            lambda a, p: "reached" if a == "169.254.169.254" else _measured(a, p),
            "got an answer",
        ),
        (
            lambda a, p: "refused" if a == "168.63.129.16" else _measured(a, p),
            "got an answer",
        ),
        (lambda a, p: "failed" if p == 443 else _measured(a, p), "got an answer"),
    ],
)
async def test_any_answer_stops_a_non_root_agent(result_for, why):
    env = _Unreachable(_report(result_for))
    record = await block_cloud_metadata(env, "agent")
    assert record["state"] == "unproven"
    assert why in record["why"]


@pytest.mark.asyncio
async def test_a_probe_that_cannot_run_is_unproven():
    record = await block_cloud_metadata(
        _Unreachable("BF metadata-reach skipped\n"), "agent"
    )
    assert record["state"] == "unproven"
    assert "no bash and timeout" in record["why"]
    record = await block_cloud_metadata(_Unreachable("", code=1), "agent")
    assert record["state"] == "unproven"


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "root"])
async def test_a_root_agent_is_proven_too_or_keeps_the_providers_policy(user):
    env = _Unreachable(_report(_measured))
    record = await block_cloud_metadata(env, user)
    assert record["state"] == NOT_REACHABLE and record["root_agent"] is True
    [(command, kwargs)] = env.calls
    assert kwargs["user"] == "root" and "setpriv" not in command
    # A proof that does not hold leaves a root agent where it was before.
    env = _Unreachable(_report(lambda a, p: "reached"))
    record = await block_cloud_metadata(env, user)
    assert record["state"] == "provider-managed"


@pytest.mark.asyncio
async def test_a_backend_that_declares_nothing_is_never_probed():
    """Docker and Daytona keep the block: no reach probe runs for them."""

    class _Provider:
        cloud_metadata_provider_managed = True

        async def exec(self, command, **kwargs):
            raise AssertionError(f"nothing runs for a provider-managed root: {command}")

    record = await block_cloud_metadata(_Provider(), None)
    assert record["state"] == "provider-managed"


def test_modal_declares_no_route_and_docker_and_daytona_do_not():
    pytest.importorskip("tenacity")
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox

    assert ModalSandbox.cloud_metadata_unreachable is True
    for cls in (DockerSandbox, DaytonaSandbox):
        assert getattr(cls, "cloud_metadata_unreachable", False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("require_block", [False, True])
async def test_not_reachable_lets_the_agent_start_and_is_recorded(
    tmp_path, require_block
):
    """The agent starts on not_reachable (the task's strict opt-in too: the
    endpoints are proven out of reach), and result.json says not_reachable."""
    import contextlib

    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        cloud_metadata_record,
    )
    from tests.test_cloud_metadata_block import _connect_patches, _rollout

    trial = _rollout(
        tmp_path, CloudMetadataPlan(armed=True, require_block=require_block)
    )
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record(NOT_REACHABLE, "no route", probes={})

    with contextlib.ExitStack() as stack:
        for patched in _connect_patches(trial, order, block):
            stack.enter_context(patched)
        await trial.connect()

    # The block step (here: the proof) comes right before the agent starts.
    assert order[-2:] == ["block agent", "agent"]
    assert trial._cloud_metadata_result == {"state": "not_reachable"}
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "not_reachable"


def test_the_reach_command_is_constant_and_small():
    """The probe is the same text every trial (no value from the run in it),
    well inside one exec's transport bound."""
    assert _cloud_metadata_reach_cmd() == _cloud_metadata_reach_cmd()
    assert len(_cloud_metadata_reach_cmd().encode()) < 8 * 1024
