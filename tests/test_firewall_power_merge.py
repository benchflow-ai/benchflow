"""Guards the S3 merge of 56e689d8 against regressions in f02e4607."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import firewall_power as fp
from tests._receipt_fakes import receipt_aware


@pytest.mark.parametrize("user", ["root", "agent"])
@pytest.mark.parametrize("code", [1, 125, 126])
async def test_failed_audit_status_never_becomes_success(user, code):
    """Guard f02e4607: plausible stdout cannot cancel a failed checked exec."""

    class Env:
        supports_system_shell = True

        @receipt_aware
        async def exec(self, command, **kwargs):
            return SimpleNamespace(
                stdout="BF fwpower-root net_admin=0\n", return_code=code
            )

    out, failed = await fp._exec(Env(), "printf fixture", 10, user=user)
    assert not out and failed


async def test_direct_session_audit_requires_a_receipt():
    """Guard f02e4607: a direct agent audit also needs checked raw output."""

    class Env:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout="BF fwpower-agent nnp=1\n", return_code=0)

    out, failed = await fp._exec(Env(), "printf fixture", 10, user="agent")
    assert not out and failed


def test_audit_preserves_the_shared_runtime_route(monkeypatch, tmp_path):
    """Guard f02e4607: execute the shared route's shell, HOME and tool PATH."""
    import os
    import shlex
    import subprocess

    from benchflow.sandbox import _base, lockdown

    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    shell = tmp_path / "sh"
    shell.write_text('#!/bin/sh\nprintf "protected-shell\\n"; exec /bin/sh "$@"\n')
    shell.chmod(0o755)
    path = f"{tmp_path}:/usr/bin:/bin"
    home = "/tmp/protected-root-home"
    monkeypatch.setattr(
        _base,
        "with_system_path",
        lambda command: (
            f"PATH={shlex.quote(path)} HOME={home}; export PATH HOME; " + command
        ),
    )
    command = fp._root_route('printf "%s\\n%s\\n" "$PATH" "$HOME"')
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["protected-shell", path, home]


@pytest.mark.parametrize("tool", ["env", "sh"])
def test_route_bootstrap_never_runs_a_writable_helper(monkeypatch, tmp_path, tool):
    """Guard b0322920: shared system PATH is not yet PT's sealed tool tree."""
    import os
    import shlex
    import subprocess

    from benchflow.sandbox import _base, lockdown

    marker = tmp_path / "executed"
    helper = tmp_path / tool
    helper.write_text(f"#!/bin/sh\nprintf unsafe > {shlex.quote(str(marker))}\n")
    helper.chmod(0o777)
    monkeypatch.setattr(_base, "SYSTEM_PATH", f"{tmp_path}:/usr/bin:/bin")
    # PT's route pins an explicit PATH; put the planted tool on that route.
    monkeypatch.setattr(
        _base,
        "with_system_path",
        lambda command: (
            f"PATH={shlex.quote(str(tmp_path))}:/usr/bin:/bin; export PATH; " + command
        ),
    )
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    result = subprocess.run(
        ["/bin/sh", "-c", fp._root_route("printf safe")],
        capture_output=True,
        text=True,
    )
    assert not marker.exists()
    assert result.returncode == 126
    assert f"BF fwpower-error tool={tool}" in result.stdout
