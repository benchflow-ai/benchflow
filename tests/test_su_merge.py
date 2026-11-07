"""Guard the SU merge of 85b87bc9: preserve env inside S2's launch barrier."""

from __future__ import annotations

import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests._firewall_tools import audit_image_tools  # noqa: F401

pytestmark = pytest.mark.usefixtures("audit_image_tools")


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("su_kind", ["busybox", "shadow", "util-linux"])
def test_bounded_su_preserves_the_authorized_environment(tmp_path, shell, su_kind):
    """Guards the SU/85b87bc9 merge without weakening its mandatory barrier."""
    from benchflow.sandbox.lockdown import build_priv_drop_cmd

    tools = tmp_path / "tools"
    tools.mkdir()
    setpriv = tools / "setpriv"
    setpriv.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --help ]; then echo 'bounding-set inh-caps ambient-caps no-new-privs'; exit; fi\n"
        'printf \'%s\\n\' "$*" > "$PROBE_FLAGS"\n'
        'while [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n'
    )
    su = tools / "su"
    su.write_text(
        "#!/bin/sh\nlogin= shell=/bin/sh command=\n"
        'while [ "$#" -gt 0 ]; do case $1 in\n'
        " -l) login=1;; -s) shift; shell=$1;; -c) shift; command=$1;; esac; shift; done\n"
        'if [ -n "$login" ]; then exec /usr/bin/env -i "$shell" -c "$command"; fi\n'
        'if [ "$SU_KIND" = shadow ]; then export PATH=/usr/bin:/bin; fi\n'
        'if [ "$SU_KIND" = util-linux ]; then export MAIL=dummy; fi\n'
        'exec "$shell" -c "$command"\n'
    )
    for script in (setpriv, su):
        script.chmod(0o755)
    flags = tmp_path / "flags"
    path = f"{tools}:/usr/bin:/bin"
    # Root's launch pins its own PATH (C8): the stand-ins are the tools there.
    with patch("benchflow.sandbox._root_launch.AGENT_LAUNCH_PATH", path):
        command = build_priv_drop_cmd(
            'test "$BF_SU_MARK" = present && test "$PATH" = "$EXPECTED_PATH" '
            '&& test -z "${BF_ROOT_ONLY-}${BENCHFLOW_SU_NAMES-}${BENCHFLOW_SU_PATH-}${MAIL-}" '
            "&& printf 'authorized environment preserved\\n'",
            "agent",
        )
    result = subprocess.run(
        [shell, "-c", "BF_ROOT_ONLY=dummy; " + command],
        env={
            "PATH": path,
            "EXPECTED_PATH": path,
            "BF_SU_MARK": "present",
            "SU_KIND": su_kind,
            "PROBE_FLAGS": str(flags),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "authorized environment preserved\n"
    recorded = shlex.split(flags.read_text())
    assert {
        "--no-new-privs",
        "--bounding-set=-net_admin,-net_raw",
        "--inh-caps=-all",
        "--ambient-caps=-all",
    } <= set(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize("system_shell", [True, False])
@pytest.mark.parametrize(
    "mode", ["valid", "missing", "duplicate", "corrupt", "nonzero"]
)
async def test_firewall_reads_require_a_successful_raw_receipt(system_shell, mode):
    """Guards SU's cccac709 audit against unverified output and failed execs."""
    from benchflow.sandbox._exec_output import raw_output_requested
    from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command
    from benchflow.sandbox.firewall_power import _exec, _root_route

    class Env:
        supports_system_shell = system_shell

        async def exec(self, command, **kwargs):
            assert kwargs.get("system_shell", False) is system_shell
            assert kwargs["user"] == "root"
            assert raw_output_requested()
            wrapped = unwrap_receipt_command(command)
            assert wrapped is not None
            assert wrapped[0] == _root_route("printf dummy")
            output = encode_receipt(b"dummy", 7 if mode == "nonzero" else 0, wrapped[1])
            if mode == "missing":
                output = "dummy"
            elif mode == "duplicate":
                output *= 2
            elif mode == "corrupt":
                output = output.replace("64756d6d79", "64756d6d78")
            return SimpleNamespace(stdout=output, stderr="", return_code=0)

    output, failure = await _exec(Env(), "printf dummy", 10)
    if mode == "valid" and system_shell:
        assert (output, failure) == ("dummy", "")
    else:
        assert output == "" and failure


@pytest.mark.asyncio
async def test_firewall_transport_failure_cannot_be_clear():
    """Guards cccac709: a transport status cannot certify a successful scan."""
    from benchflow.sandbox.firewall_power import _exec

    class Env:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(
                stdout="BF fwpower-root net_admin=0\n", return_code=7
            )

    output, failure = await _exec(Env(), "printf dummy", 10)
    assert output == "" and failure


def test_root_scan_never_executes_the_image_python(tmp_path):
    """Guards SU's 8f4442b0 merge against PT's forbidden image-Python fallback."""
    from benchflow.sandbox.firewall_power import _scan_cmd
    from tests.test_firewall_power import Stub

    stub = Stub(tmp_path)
    marker = tmp_path / "image-python-ran"
    stub.tool("python3", f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n")
    output = stub.run(_scan_cmd(whole_fs=True, fs_root=str(stub.fs)))
    assert not marker.exists()
    assert "BF fwpower-scan scope=all caps=" in output
    assert "complete=0" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("available", [False, True])
async def test_root_python_requires_the_runtime_authority(
    monkeypatch, asynchronous, available
):
    """Guards c877980b: an executable at the kit path is not its custody proof."""
    from benchflow import _runtime_tree as runtime
    from benchflow.sandbox.firewall_power import _protected_python

    environment = object()

    def resolve(env):
        assert env is environment
        return ["/tmp/protected-python", "-I", "-S"] if available else None

    async def async_resolve(env):
        return resolve(env)

    monkeypatch.setattr(
        runtime, "runtime_kit_python", async_resolve if asynchronous else resolve
    )
    assert await _protected_python(environment) == (
        "/tmp/protected-python" if available else None
    )


@pytest.mark.asyncio
async def test_broken_runtime_cannot_fall_back_to_an_image_python(monkeypatch):
    """Guards c877980b: a failed kit proof must remain a startup refusal."""
    from benchflow import _runtime_tree as runtime
    from benchflow.sandbox.firewall_power import _protected_python

    async def broken(env):
        raise RuntimeError("dummy kit custody failure")

    monkeypatch.setattr(runtime, "runtime_kit_python", broken)
    with pytest.raises(RuntimeError, match="custody failure"):
        await _protected_python(object())


@pytest.mark.asyncio
async def test_scan_uses_the_proven_runtime_entrypoint(monkeypatch):
    """Guards c877980b through the production coordinator, not an import check."""
    from benchflow import _runtime_tree as runtime
    from benchflow.sandbox.firewall_power import check_firewall_power
    from tests._receipt_fakes import receipt_aware

    seen = []

    async def resolve(env):
        seen.append(env)
        return ["/tmp/proven-python", "-I", "-S"]

    monkeypatch.setattr(runtime, "runtime_kit_python", resolve)

    class Env:
        @receipt_aware
        async def exec(self, command, **kwargs):
            assert "/tmp/proven-python -I -S -B" in command
            return SimpleNamespace(
                stdout="BF fwpower-root net_admin=0\n", return_code=0
            )

    environment = Env()
    record, refusal = await check_firewall_power(environment, "agent", holds=("dummy",))
    assert seen == [environment]
    assert refusal is None and record["state"] == "clear"
