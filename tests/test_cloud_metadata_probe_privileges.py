"""Batch 18: supplemental probes cannot become an agent privilege bypass."""

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from tests.test_cloud_metadata_block import AGENT_REFUSED, ROOT_OK, FakeEnv
from tests.test_cloud_metadata_block import sandbox as sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "placement",
    ["system", "absolute", "relative", "empty", "spaces", "symlink", "delegate"],
)
async def test_audit_candidates_inherit_no_new_privs(sandbox, shell, tool, placement):
    """Guards batch 18 at 36ef925b: reconnect probes run mutable candidates.

    Observe the real kernel flag in harmless scripts; no set-ID transition
    or firewall operation is simulated by the unit fixture.
    """
    from benchflow.sandbox import lockdown

    sandbox.caps("0" * 16)
    folder = (
        sandbox.bin
        if placement == "system"
        else sandbox.root / ("with spaces" if placement == "spaces" else "agent-tools")
    )
    folder.mkdir(exist_ok=True)
    target = folder / tool
    body = (
        "#!/bin/sh\nwhile read -r key value rest; do "
        'case $key in NoNewPrivs:) printf "%s\\n" "$value" >> "$PRIV_MARKER";; esac; '
        "done < /proc/self/status\nexit 1\n"
    )
    target.write_text(body)
    target.chmod(0o755)
    if placement in {"symlink", "delegate"}:
        real = folder / "ordinary.real"
        target.rename(real)
        if placement == "symlink":
            target.symlink_to(real)
        else:
            target.write_text(f'#!/bin/sh\nexec {shlex.quote(str(real))} "$@"\n')
            target.chmod(0o755)
    path = f"{folder}:{sandbox.bin}:{sandbox._system_bin()}"
    if placement in {"relative", "empty"}:
        path = "." if placement == "relative" else ""
    command = lockdown._cloud_metadata_agent_check_cmd(
        "BFS2DUMMY", proc=str(sandbox.proc), connect_probe=False
    )
    command = f"cd {shlex.quote(str(folder))} && {command}"
    marker = sandbox.root / "probe-privileges"

    class Env:
        supports_system_shell = True

        async def exec(self, launch, user=None, **kwargs):
            if user == "root":
                assert kwargs.get("system_shell") is True
                for flag in (
                    "--no-new-privs",
                    "--bounding-set=-net_admin,-net_raw",
                    "--inh-caps=-all",
                    "--ambient-caps=-all",
                ):
                    assert flag in launch
                # Model only the privileged uid/bounding transition. Execute
                # the real no_new_privs barrier and real candidate scripts;
                # the root-container repro checks the entire shared launcher.
                body = (
                    "/usr/bin/setpriv --no-new-privs --inh-caps=-all --ambient-caps=-all -- /bin/sh -c "
                    + shlex.quote(command)
                )
            else:
                assert user == "agent"
                body = command
            return SimpleNamespace(
                stdout=sandbox.run(body, shell, PATH=path, PRIV_MARKER=str(marker)),
                return_code=0,
            )

    run = getattr(lockdown, "_cloud_metadata_agent_exec", None)
    if run is None:
        result, _ = await lockdown._cloud_metadata_exec(
            Env(), command, user="agent", timeout_sec=30
        )
    else:
        result, _ = await run(Env(), command, "agent", timeout_sec=30)
    assert result is not None
    assert marker.exists(), "the candidate must actually run"
    assert set(marker.read_text().splitlines()) == {"1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_system_shell", [True, False])
@pytest.mark.parametrize(
    "failure",
    [
        "nonzero",
        "missing-code",
        "boolean",
        "string",
        "float",
        "timeout",
        "empty",
        "duplicate",
    ],
)
async def test_optional_tcp_fallback_uses_the_shared_barrier_and_fresh_proof(
    supports_system_shell, failure
):
    """Guards batch 18 at 36ef925b: an incomplete audit is no TCP receipt."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    class Env(FakeEnv):
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-connect" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is supports_system_shell
                for flag in (
                    "--no-new-privs",
                    "--bounding-set=-net_admin,-net_raw",
                    "--inh-caps=-all",
                    "--ambient-caps=-all",
                ):
                    assert flag in command
                if failure == "timeout":
                    raise TimeoutError("dummy TCP proof timeout")
                return SimpleNamespace(
                    stdout={
                        "empty": "",
                        "duplicate": "BF metadata-connect refused\n" * 2,
                    }.get(failure, "BF metadata-connect refused\n"),
                    return_code={
                        "nonzero": 1,
                        "missing-code": None,
                        "boolean": False,
                        "string": "0",
                        "float": 0.0,
                    }.get(failure, 0),
                )
            if "BF metadata-agent " in command:
                return SimpleNamespace(stdout=AGENT_REFUSED, return_code=7)
            return await super().exec(command, user=user, **kwargs)

    env = Env(ROOT_OK, "BF metadata-hits 1\n")
    env.supports_system_shell = supports_system_shell
    record = await block_cloud_metadata(env, "agent")
    assert record["state"] == "unproven"
    assert record["agent_privilege_audit"] == "unavailable"
