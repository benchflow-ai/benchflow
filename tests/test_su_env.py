"""The agent keeps its environment on images without util-linux setpriv.

Guards PY8 SU: ``build_priv_drop_cmd`` fell back to ``su -l <user> -c`` where
the image has no util-linux setpriv (Alpine and BusyBox images, a Debian
without it), and through ``setpriv --no-new-privs -- su -l`` under BusyBox's
setpriv (S3's ``nnp`` drop). A login shell clears the environment, so the
agent env the root shell had just sourced (keys, proxy settings) was gone and
the agent failed before doing anything ("Authentication required").

These tests run the built command in a real shell, the way ``DockerProcess``
starts it (a root shell that sources the agent env file, then runs the
command), with stand-ins for setpriv and su in ``tmp_path``: the su stand-in
follows BusyBox's, util-linux's and shadow's su (``-l`` clears the
environment as a login shell does; ``-m`` keeps it, and shadow's su still
resets PATH). The fake agent writes a marker line and the names of its
environment's variables, never their values. Every image kind must give the
agent the environment the setpriv path gives it. The real su programs run in
the deterministic tier (tests/test_deterministic_su_env.py).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.sandbox.lockdown import build_priv_drop_cmd

USER = "agent"

STUB_SETPRIV = """#!/bin/sh
if [ "$1" = --help ]; then printf '%s\\n' "$SETPRIV_HELP"; exit 0; fi
printf 'setpriv %s\\n' "$*" >> "$SU_LOG"
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done
shift
exec "$@"
"""

# Stand-in for su run by root. -l clears the environment as a login shell
# does; -m and -p keep it (shadow's su still resets PATH from ENV_PATH);
# neither sets HOME and SHELL (and USER and LOGNAME for a non-root user).
# Without -s, -m picks $SHELL, as BusyBox's and util-linux's su do.
# util-linux's su runs a PAM session: Debian's pam_mail sets MAIL and
# pam_env sets what /etc/environment holds (PATH on Ubuntu).
STUB_SU = """#!/bin/sh
printf 'su %s\\n' "$*" >> "$SU_LOG"
login= keep= shell= cmd= user=root
while [ "$#" -gt 0 ]; do
  case $1 in
    -l|-|--login) login=1;;
    -m|-p|--preserve-environment) keep=1;;
    -s|--shell) shift; shell=$1;;
    -c|--command) shift; cmd=$1;;
    -*) echo "su: unknown option $1" >&2; exit 2;;
    *) user=$1;;
  esac
  shift
done
if [ -z "$shell" ] && [ -n "$keep" ] && [ -z "$login" ]; then shell=$SHELL; fi
[ -n "$shell" ] || shell=/bin/bash
echo "su switched to $user" >> "$SU_LOG"
if [ -n "$login" ]; then
  exec env -i HOME="/home/$user" USER="$user" LOGNAME="$user" SHELL="$shell" \\
    PATH=/usr/bin:/bin "$shell" -c "$cmd"
fi
if [ -z "$keep" ]; then
  export HOME="/home/$user" SHELL="$shell" USER="$user" LOGNAME="$user"
fi
if [ "$SU_KIND" = shadow ]; then export PATH=/usr/bin:/bin; fi
if [ "$SU_KIND" = util-linux ]; then
  export MAIL="/var/mail/$user" PATH=/usr/local/bin:/usr/bin:/bin
fi
exec "$shell" -c "$cmd"
"""

UTIL_LINUX_HELP = (
    " --no-new-privs              disallow granting new privileges\n"
    " --inh-caps <caps,...>       set inheritable capabilities\n"
    " --bounding-set <caps>       set capability bounding set\n"
    " --reuid <uid|user>          set real and effective uid\n"
    " --init-groups               initialize supplementary groups"
)
# BusyBox v1.36.1's (alpine:3.20): no --reuid, no --bounding-set, but --nnp.
BUSYBOX_HELP = (
    "Usage: setpriv [OPTIONS] PROG ARGS\n\n"
    "--nnp,--no-new-privs\tIgnore setuid/setgid bits and file capabilities\n"
    "--inh-caps CAP,CAP\tSet inheritable capabilities"
)

# (setpriv --help text or None for no setpriv, the su, drop_net_admin)
SETPRIV_KINDS = {
    "util-linux-setpriv": (UTIL_LINUX_HELP, "util-linux", False),
    "util-linux-setpriv-drop": (UTIL_LINUX_HELP, "util-linux", True),
}
SU_KINDS = {
    "bounded-busybox-su": (
        UTIL_LINUX_HELP.replace("--reuid", "--uid"),
        "busybox",
        False,
    ),
    "bounded-util-linux-su": (
        UTIL_LINUX_HELP.replace("--reuid", "--uid"),
        "util-linux",
        False,
    ),
    "bounded-shadow-su": (UTIL_LINUX_HELP.replace("--reuid", "--uid"), "shadow", True),
    "no-setpriv-busybox-su": (None, "busybox", False),
    "no-setpriv-util-linux-su": (None, "util-linux", False),
    "no-setpriv-shadow-su": (None, "shadow", False),
    "busybox-setpriv-su": (BUSYBOX_HELP, "busybox", False),
    "busybox-setpriv-nnp-drop": (BUSYBOX_HELP, "busybox", True),
    "busybox-setpriv-shadow-su-nnp-drop": (BUSYBOX_HELP, "shadow", True),
}
# What the setpriv path itself sets; the su path may differ only there.
SET_BY_THE_DROP = {"HOME", "USER", "LOGNAME", "SHELL", "PATH"}


def _launch(tmp_path: Path, kind: tuple[str | None, str, bool], *, root_sh: str):
    help_text, su_kind, drop = kind
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    tools = (
        "bash",
        "sh",
        "grep",
        "cat",
        "printf",
        "env",
        "cut",
        "sort",
        "rm",
        "sed",
        "tr",
    )
    for name in tools:
        found = shutil.which(name)
        assert found, name
        (bin_dir / name).symlink_to(found)
    if help_text is not None:
        (bin_dir / "setpriv").write_text(STUB_SETPRIV)
        (bin_dir / "setpriv").chmod(0o755)
    (bin_dir / "su").write_text(STUB_SU)
    (bin_dir / "su").chmod(0o755)
    cwd = tmp_path / "work"
    cwd.mkdir()
    out = tmp_path / "agent.txt"
    log = tmp_path / "su.log"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f"echo bash-env >> {shlex.quote(str(log))}\n")
    # The fake agent: a marker line and the variable names, never values.
    agent = (
        f"{{ printf 'marker %s\\n' \"${{BF_AGENT_MARK:+present}}\";"
        " printf 'home %s\\n' \"$HOME\"; printf 'path %s\\n' \"$PATH\";"
        " printf 'cwd %s\\n' \"$PWD\";"
        " env | cut -d= -f1 | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' | sort;"
        f" }} > {shlex.quote(str(out))}"
    )
    # Root's launch pins its own PATH (C8): the stand-ins are the tools there.
    with patch("benchflow.sandbox._root_launch.AGENT_LAUNCH_PATH", str(bin_dir)):
        cmd = build_priv_drop_cmd(agent, USER, drop_net_admin=drop)
    # As DockerProcess.start: the agent env goes to a file the root shell
    # sources (and removes) just before the drop. BF_ROOT_ONLY is root's
    # own (not exported); neither path may pass it.
    env_file = tmp_path / "agent.env"
    env_file.write_text(
        "export BF_AGENT_MARK=fake-key-value\n"
        f"export BASH_ENV={shlex.quote(str(bash_env))}\n"
        "export HTTPS_PROXY=http://127.0.0.1:18080\n"
    )
    root_cmd = (
        f"BF_ROOT_ONLY=1; cd {shlex.quote(str(cwd))} &&"
        f" . {shlex.quote(str(env_file))} && rm -f {shlex.quote(str(env_file))}"
        f" && {cmd}"
    )
    result = subprocess.run(
        [root_sh, "-c", root_cmd],
        capture_output=True,
        text=True,
        env={
            "PATH": str(bin_dir),
            "HOME": "/root",
            # One line: the agent lists variable names line by line.
            "SETPRIV_HELP": (help_text or "").replace("\n", " "),
            "SU_KIND": su_kind,
            "SU_LOG": str(log),
        },
        timeout=30,
    )
    if help_text is None or "bounding-set" not in help_text:
        # S2-CONF-9 supersedes SU's unrestricted and BusyBox nnp fallbacks.
        assert result.returncode != 0
        assert "requires util-linux setpriv" in result.stderr
        assert not out.exists()
        assert not log.exists()  # no su, startup file or agent code ran
        return None, set(), [], str(bin_dir), str(cwd)
    assert result.returncode == 0, (result.returncode, result.stderr)
    lines = out.read_text().splitlines()
    facts = dict(ln.split(" ", 1) for ln in lines[:4])
    keys = set(lines[4:])
    calls = log.read_text().splitlines() if log.exists() else []
    return facts, keys, calls, str(bin_dir), str(cwd)


def _root_shells() -> list[str]:
    # DockerProcess starts the root shell with bash; a direct Daytona
    # sandbox may run it under sh.
    return [s for s in ("/bin/bash", "/bin/sh") if Path(s).exists()]


@pytest.mark.parametrize("root_sh", _root_shells())
@pytest.mark.parametrize("name", sorted(SU_KINDS))
def test_the_su_path_gives_the_agent_the_setpriv_paths_environment(
    tmp_path, name, root_sh
):
    reference, ref_keys, _, bin_dir, cwd = _launch(
        tmp_path / "ref", SETPRIV_KINDS["util-linux-setpriv"], root_sh=root_sh
    )
    facts, keys, calls, su_bin, su_cwd = _launch(
        tmp_path / "su", SU_KINDS[name], root_sh=root_sh
    )
    if facts is None:
        return  # _launch proved the mandatory refusal before any user code

    # The command went through su (not a setpriv that switched users).
    assert any(c.startswith("su ") for c in calls), calls
    assert facts["marker"] == "present", (facts, sorted(keys))
    assert "HTTPS_PROXY" in keys and "BASH_ENV" in keys
    assert keys == ref_keys, sorted(keys ^ ref_keys)
    assert (keys ^ ref_keys) <= SET_BY_THE_DROP
    assert facts["home"] == reference["home"] == f"/home/{USER}"
    # Root's PATH, as on the setpriv path, even where su resets it (shadow).
    assert reference["path"] == bin_dir and facts["path"] == su_bin
    # No login shell: the working directory stays, as with setpriv.
    assert reference["cwd"] == cwd and facts["cwd"] == su_cwd


@pytest.mark.parametrize("root_sh", _root_shells())
@pytest.mark.parametrize("name", sorted({**SETPRIV_KINDS, **SU_KINDS}))
def test_nothing_of_roots_own_reaches_the_agent(tmp_path, name, root_sh):
    kind = {**SETPRIV_KINDS, **SU_KINDS}[name]
    facts, keys, _, _, _ = _launch(tmp_path, kind, root_sh=root_sh)
    if facts is None:
        return

    assert facts["marker"] == "present", (facts, sorted(keys))
    # Root's own variable, and the drop's helpers, never reach the agent.
    assert not keys & {
        "BF_ROOT_ONLY",
        "BENCHFLOW_SU_PATH",
        "BENCHFLOW_SU_NAMES",
        "bf_su_shell",
        "MAIL",
    }, keys
    assert "SU_LOG" in keys  # root's exported environment does, as with setpriv


@pytest.mark.parametrize("root_sh", _root_shells())
@pytest.mark.parametrize("name", sorted({**SETPRIV_KINDS, **SU_KINDS}))
def test_bash_env_runs_only_after_the_switch(tmp_path, name, root_sh):
    """Nothing of the agent's (BASH_ENV here) runs before the switch: every
    su switches users before it starts the shell that reads it."""
    kind = {**SETPRIV_KINDS, **SU_KINDS}[name]
    facts, _, calls, _, _ = _launch(tmp_path, kind, root_sh=root_sh)
    if facts is None:
        return

    assert calls.count("bash-env") == 1, calls
    switch = next(
        i
        for i, c in enumerate(calls)
        if c.startswith("su switched") or c.startswith("setpriv --reuid")
    )
    assert calls.index("bash-env") > switch, calls


@pytest.mark.parametrize("name", sorted(SU_KINDS))
def test_the_su_call_keeps_the_environment_and_no_login(tmp_path, name):
    facts, _, calls, _, _ = _launch(tmp_path, SU_KINDS[name], root_sh="/bin/bash")
    if facts is None:
        return

    (su,) = [c for c in calls if c.startswith("su ") and "switched" not in c]
    assert su.startswith("su -m -s /bin/bash -c "), su
    assert su.endswith(f" {USER}")
    assert " -l " not in su and not su.startswith("su - ")


def test_busyboxs_setpriv_still_sets_no_new_privs_before_su(tmp_path):
    """Guards S2-CONF-9 at 85b87bc9: nnp alone cannot admit BusyBox."""
    facts, _, calls, _, _ = _launch(
        tmp_path, SU_KINDS["busybox-setpriv-nnp-drop"], root_sh="/bin/bash"
    )

    assert facts is None and not calls


def test_util_linux_setpriv_is_unchanged(tmp_path):
    """The setpriv path never goes through su and its line is as before."""
    _, _, calls, _, _ = _launch(
        tmp_path, SETPRIV_KINDS["util-linux-setpriv-drop"], root_sh="/bin/bash"
    )

    assert not any(c.startswith("su ") for c in calls), calls
    (call,) = [c for c in calls if c.startswith("setpriv ")]
    # Root's launch starts /bin/bash by path with no startup file (C8).
    assert call.startswith(
        "setpriv --reuid=agent --regid=agent --init-groups --no-new-privs"
        " --bounding-set=-net_admin,-net_raw --inh-caps=-all --ambient-caps=-all"
        " -- /bin/bash --noprofile --norc -c "
    )
    plain = build_priv_drop_cmd("my-agent", USER)
    assert (
        "exec setpriv --reuid=agent --regid=agent --init-groups --no-new-privs"
        " --bounding-set=-net_admin,-net_raw --inh-caps=-all --ambient-caps=-all"
        " -- /bin/bash --noprofile --norc -c 'export HOME=/home/agent; " in plain
    )
    assert "exec /bin/bash -c my-agent" in plain


def test_without_bash_su_starts_sh(tmp_path):
    """The shell su starts is bash where the image has it, else sh; never
    root's $SHELL, which su -m would pick."""
    cmd = build_priv_drop_cmd("my-agent", USER)
    pick = cmd.split("-q bounding-set; then", 1)[1].split("BENCHFLOW_SU_NAMES=", 1)[0]
    assert pick.strip() == (
        "if [ -x /bin/bash ]; then bf_su_shell=/bin/bash; else bf_su_shell=/bin/sh; fi;"
    )
    assert '-s "$bf_su_shell"' in cmd
    subprocess.run(["bash", "-n", "-c", cmd], check=True)
    subprocess.run(
        ["bash", "-n", "-c", build_priv_drop_cmd("x", USER, drop_net_admin=True)],
        check=True,
    )
