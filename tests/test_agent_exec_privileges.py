"""S2-CONF-9: the shared launch cannot fall back to an unrestricted exec."""

from __future__ import annotations

import os
import shlex
import subprocess

import pytest

from benchflow.sandbox.lockdown import build_priv_drop_cmd


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("checked_drop", [None, False, True])
@pytest.mark.parametrize(
    "implementation", ["full", "su", "busybox", "missing", "reject"]
)
def test_privilege_barrier_precedes_every_agent_entry(
    tmp_path, shell, implementation, monkeypatch, checked_drop
):
    """Guards S2-CONF-9 at 056be2ad: unsupported drops never run agent code.

    The fixture models setpriv's feature/argv contract, not a uid transition;
    genuine set-ID and file-capability execs are covered by the VM repro.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "argv"
    marker = tmp_path / "agent-ran"
    setpriv = bin_dir / "setpriv"
    setpriv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --help ]; then\n'
        f" case {implementation} in\n"
        " full|reject) echo 'reuid bounding-set no-new-privs';;\n"
        " su) echo 'bounding-set no-new-privs';;\n"
        " busybox) echo 'no-new-privs';;\n"
        " missing) exit 127;;\n"
        " esac; exit 0\nfi\n"
        'printf "%s\\n" "$@" > "$TRACE"\n'
        f"[ {implementation} != reject ] || exit 1\n"
        'while [ "$1" != -- ]; do shift; done\nshift\n'
        'exec "$@"\n'
    )
    setpriv.chmod(0o755)
    # A model of su's identity transition keeps the test on the runner's uid.
    su = bin_dir / "su"
    su.write_text(
        "#!/bin/sh\nshell=/bin/sh\n"
        'while [ "$#" -gt 0 ]; do case $1 in\n'
        " -s) shift; shell=$1;; -c) shift; command=$1;; esac; shift; done\n"
        'exec "$shell" -c "$command"\n'
    )
    su.chmod(0o755)
    launch = f"printf ran > {shlex.quote(str(marker))}"
    # The launch pins its own PATH before the drop (C8): the
    # stand-in setpriv and su sit on that PATH, not the caller's.
    from benchflow.sandbox import _root_launch

    monkeypatch.setattr(_root_launch, "AGENT_LAUNCH_PATH", f"{bin_dir}:/usr/bin:/bin")
    options = {} if checked_drop is None else {"drop_net_admin": checked_drop}
    result = subprocess.run(
        [shell, "-c", build_priv_drop_cmd(launch, "nobody", **options)],
        env={**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "TRACE": str(trace)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    if implementation in {"busybox", "missing", "reject"}:
        assert result.returncode != 0
        assert not marker.exists(), result
    else:
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "ran"
        argv = trace.read_text().splitlines()
        boundary = argv.index("--")
        options = argv[:boundary]
        assert "--no-new-privs" in options
        assert "--bounding-set=-net_admin,-net_raw" in options
        assert "--inh-caps=-all" in options
        assert "--ambient-caps=-all" in options
        if implementation == "su":
            assert argv[boundary + 1 : boundary + 4] == ["su", "-m", "-s"]
            assert argv[-1] == "nobody"
            assert "-l" not in argv[boundary + 1 :]


@pytest.mark.parametrize("user", ["agent", "bench-user", "65534"])
@pytest.mark.parametrize("checked_drop", [False, True])
def test_both_identity_drop_branches_use_the_same_barrier(user, checked_drop):
    """Guards S2-CONF-9 at 056be2ad for named and numeric sandbox users."""
    command = build_priv_drop_cmd("exec /bin/true", user, drop_net_admin=checked_drop)
    for option in (
        "--no-new-privs",
        "--bounding-set=-net_admin,-net_raw",
        "--inh-caps=-all",
        "--ambient-caps=-all",
    ):
        assert command.count(option) == 2
