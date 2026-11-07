"""The files BenchFlow writes for and gives to the agent user, in a real
container: the published final reply and trajectory (#1182) and the agent
tool's temp folder (#1183).

The container is ``python:3.12-slim`` (as in ``test_sealed_channel_container``)
with an ``agent`` user and /logs laid out as setup leaves it: /logs root's,
/logs/agent the agent's. Skipped when no Docker daemon is available.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.rollout._setup import (
    _prepare_agent_temp_dirs,
    _publish_trajectory_for_verifier,
)
from tests.reachable_fixture import directory

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"

TRAJECTORY = [
    {"type": "user_message", "text": "Reply with Answer: X."},
    {"type": "agent_message", "text": "Answer: C"},
]


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str, user: str = "root") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", user, name, "sh", "-c", command],
        capture_output=True,
        text=True,
    )


class ContainerEnv:
    """The sandbox calls publishing and the temp-folder step make, on Docker."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def exec(self, command: str, user: str | None = None, timeout_sec=None):
        done = _dx(self.name, command, user or "root")
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    async def upload_file(self, source: Path | str, target: str, **_kwargs) -> None:
        subprocess.run(
            ["docker", "cp", str(source), f"{self.name}:{target}"],
            check=True,
            capture_output=True,
        )


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "bf-v6-test-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            "--label",
            "benchflow.target=FF-verifier",
            IMAGE,
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start container: {run.stderr[:200]}")
    setup = _dx(
        name,
        "useradd -m agent && mkdir -p /logs/agent /logs/verifier && "
        "chown root:root /logs && chmod 755 /logs && chown agent:agent /logs/agent && "
        "printf 'original\\n' > /usr/local/bin/victim-a && chmod 755 /usr/local/bin/victim-a",
    )
    assert setup.returncode == 0, setup.stderr
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _read(name: str, path: str) -> str:
    done = _dx(name, f"cat {path}")
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.asyncio
async def test_final_reply_reaches_logs_agent_as_the_agent_user(box, tmp_path):
    """Guards #1182: the verifier finds the reply where MMMLU reads it."""
    await _publish_trajectory_for_verifier(
        ContainerEnv(box),
        TRAJECTORY,
        directory(tmp_path / "agent"),
        sandbox_user="agent",
    )

    assert _read(box, "/logs/agent/output.txt") == "Answer: C\n"
    assert '"Answer: C"' in _read(box, "/logs/agent/acp_trajectory.jsonl")
    owners = _dx(
        box, "stat -c %U /logs/agent/output.txt /logs/agent/acp_trajectory.jsonl"
    )
    assert owners.stdout.split() == ["agent", "agent"]
    # Nothing is left in /logs besides the two folders.
    assert _dx(box, "ls -A /logs").stdout.split() == ["agent", "verifier"]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["mkdir -m", "chmod 644", "rc=0;"])
async def test_publication_recovers_when_exec_ack_is_lost_after_completion(
    box, tmp_path, step
):
    """Guards FF-VERIFIER-CONFIRM-3 against 2c153ddc after real exec effects."""

    class LostAckEnv(ContainerEnv):
        lost = False

        async def exec(self, command, **kwargs):
            result = await super().exec(command, **kwargs)
            # ``in``, not ``startswith``: the stage mkdir follows the defensive
            # ``mkdir -p /logs/agent;`` in the same exec (PY8 U12).
            if step in command and not self.lost:
                assert result.return_code == 0, result.stderr
                self.lost = True
                raise ConnectionError("synthetic lost exec acknowledgement")
            return result

    env = LostAckEnv(box)
    await _publish_trajectory_for_verifier(
        env, TRAJECTORY, directory(tmp_path / "agent"), sandbox_user="agent"
    )
    assert env.lost
    assert _read(box, "/logs/agent/output.txt") == "Answer: C\n"
    assert '"Answer: C"' in _read(box, "/logs/agent/acp_trajectory.jsonl")
    assert _dx(box, "ls -A /logs").stdout.split() == ["agent", "verifier"]


@pytest.mark.asyncio
async def test_swapped_temp_root_cannot_give_away_a_protected_file(box):
    """Swap at the first ownership probe."""
    from benchflow.rollout._setup import _agent_temp_dirs_cmd

    setup = _dx(
        box,
        r"""
mkdir -p /tmp/opencode /tmp/agent-stash /tmp/probe-bin
chown agent:agent /tmp/opencode /tmp/agent-stash
printf 'locked sentinel' > /tmp/agent-stash/locked
chmod 600 /tmp/agent-stash/locked
cat > /tmp/probe-bin/find <<'SH'
#!/bin/sh
case "$*" in
  *'-prune -user agent'*)
    if [ ! -e /tmp/swapped ]; then
      touch /tmp/swapped
      su -s /bin/sh agent -c 'mv /tmp/opencode /tmp/previous; mv /tmp/agent-stash/locked /tmp/opencode'
    fi;;
esac
exec /usr/bin/find "$@"
SH
chmod 755 /tmp/probe-bin/find
""",
    )
    assert setup.returncode == 0, setup.stderr
    done = _dx(
        box, "PATH=/tmp/probe-bin:$PATH; " + _agent_temp_dirs_cmd(["opencode"], "agent")
    )
    assert _dx(box, "stat -c %U /tmp/opencode").stdout.strip() == "root", done.stderr
    assert _dx(box, "cat /tmp/opencode", "agent").returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "locked",
    [
        "/tmp/opencode",
        "/tmp",
        "/tmp/opencode/grader",
        "/",
    ],
)
async def test_temp_folder_preparation_never_unlocks_effective_policy(box, locked):
    """Guards the fix from a9fd1ee1, for ancestors and children too."""
    from benchflow.rollout._setup import _agent_temp_dirs_cmd

    made = _dx(
        box,
        "mkdir -p /tmp/opencode && printf private > /tmp/opencode/grader && chmod 700 /tmp/opencode && chmod 600 /tmp/opencode/grader",
    )
    assert made.returncode == 0, made.stderr
    # Use a recording adapter to support the new optional policy at the base
    # without failing first on a missing keyword instead of the behavior.
    import inspect

    kwargs = (
        {"locked_paths": [locked]}
        if "locked_paths" in inspect.signature(_agent_temp_dirs_cmd).parameters
        else {}
    )
    if locked == "/":
        # The shared lockdown validator rejects locking the filesystem root.
        with pytest.raises(ValueError, match="Invalid locked path"):
            _agent_temp_dirs_cmd(["opencode"], "agent", **kwargs)
        return
    done = _dx(box, _agent_temp_dirs_cmd(["opencode"], "agent", **kwargs))
    assert done.returncode != 0
    assert _dx(box, "stat -c '%U %a' /tmp/opencode/grader").stdout.strip() == "root 600"


@pytest.mark.asyncio
@pytest.mark.parametrize("locked", ["/tmp/open*", "/tmp/opencode/*"])
async def test_locked_policy_globs_have_the_same_scope_as_lockdown(box, locked):
    """Do not treat a locked glob as a literal path."""
    await test_temp_folder_preparation_never_unlocks_effective_policy(box, locked)


@pytest.mark.asyncio
@pytest.mark.parametrize("shadow", [False, True])
async def test_a_missing_temp_folder_is_not_created_inside_a_locked_glob(box, shadow):
    """Guards the fix before mkdir too, regardless of other glob matches."""
    import inspect

    from benchflow.rollout._setup import _agent_temp_dirs_cmd

    if shadow:
        assert _dx(box, "mkdir /tmp/open-other").returncode == 0
    kwargs = (
        {"locked_paths": ["/tmp/open*"]}
        if "locked_paths" in inspect.signature(_agent_temp_dirs_cmd).parameters
        else {}
    )
    done = _dx(box, _agent_temp_dirs_cmd(["opencode"], "agent", **kwargs))
    assert done.returncode != 0
    assert _dx(box, "test ! -e /tmp/opencode").returncode == 0


@pytest.mark.asyncio
async def test_glob_components_do_not_match_unrelated_siblings(box):
    """A policy /tmp/*/private must not block the sibling /tmp/app/cache."""
    from benchflow.rollout._setup import _agent_temp_dirs_cmd

    assert _dx(box, "mkdir -p /tmp/app/private").returncode == 0
    done = _dx(
        box,
        "TMPDIR=/tmp/app; "
        + _agent_temp_dirs_cmd(["cache"], "agent", locked_paths=["/tmp/*/private"]),
    )
    assert done.returncode == 0, done.stderr
    assert _dx(box, "touch /tmp/app/cache/ok", "agent").returncode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backing", "locked", "nested", "refused"),
    [
        ("/grading", "/grading", "", True),
        ("/grading/sub", "/grading", "", True),
        ("/grading", "/grading/sub", "", True),
        ("/grading", "/grad*", "", True),
        ("/grading", "/grading/missing", "", True),
        ("/grading/sub", "/grading/su?", "", True),
        ("/grading/sub", "/grading/s*", "", True),
        ("/", "/grading", "", True),
        (
            "/unrelated",
            "/grading",
            "3 1 0:1 /grading/sub /tmp/opencode/sub rw - tmpfs tmpfs rw\n",
            True,
        ),
        ("/unrelated", "/grading", "", False),
        ("/grading-other", "/grading", "", False),
        ("/grading/public", "/grading/private", "", False),
        ("/grading/sub", "/grading/private*", "", False),
    ],
)
async def test_mount_aliases_cannot_unlock_protected_trees(
    box, backing, locked, nested, refused
):
    """Guards FF-VERIFIER-CONFIRM-2 at 24cdccea with modeled kernel mount records.

    The test runs real ownership commands but substitutes mountinfo input;
    it needs no host mounts, Docker socket or mount capability.
    """
    from benchflow.rollout._setup import _agent_temp_dirs_cmd

    setup = _dx(
        box,
        r"""
mkdir -p /grading/sub /tmp/opencode/sub /tmp/probe-bin
printf 'private dummy marker' > /tmp/opencode/sub/grader
chmod 700 /grading /tmp/opencode
chmod 600 /tmp/opencode/sub/grader
cat > /tmp/probe-bin/awk <<'SH'
#!/bin/sh
exec /usr/bin/awk "$@" < /tmp/mountinfo
SH
chmod 755 /tmp/probe-bin/awk
""",
    )
    assert setup.returncode == 0, setup.stderr
    records = (
        "1 0 0:1 / / rw - tmpfs tmpfs rw\n"
        f"2 1 0:1 {backing} /tmp/opencode rw - tmpfs tmpfs rw\n" + nested
    )
    assert _dx(box, "cat > /tmp/mountinfo <<'EOF'\n" + records + "EOF").returncode == 0
    done = _dx(
        box,
        "PATH=/tmp/probe-bin:$PATH; "
        + _agent_temp_dirs_cmd(["opencode"], "agent", locked_paths=[locked]),
    )
    expected = "root 600" if refused else "agent 600"
    assert (
        _dx(box, "stat -c '%U %a' /tmp/opencode/sub/grader").stdout.strip() == expected
    )
    assert done.returncode == (6 if refused else 0), done.stderr
    assert (
        _dx(box, "cat /tmp/opencode/sub/grader", "agent").returncode != 0
    ) == refused


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["acp_trajectory.jsonl", "output.txt"])
async def test_a_link_the_agent_left_is_never_written_through_as_root(
    box, tmp_path, name
):
    """A link at a published name used to make root write the trajectory into
    whatever it points at (here a root-owned program a verifier might run)."""
    planted = _dx(box, f"ln -s /usr/local/bin/victim-a /logs/agent/{name}", "agent")
    assert planted.returncode == 0, planted.stderr

    await _publish_trajectory_for_verifier(
        ContainerEnv(box),
        TRAJECTORY,
        directory(tmp_path / "agent"),
        sandbox_user="agent",
    )

    assert _read(box, "/usr/local/bin/victim-a") == "original\n"
    assert (
        _dx(
            box, f"test -f /logs/agent/{name} && test ! -L /logs/agent/{name}"
        ).returncode
        == 0
    )


@pytest.mark.asyncio
async def test_a_folder_at_one_name_does_not_hold_back_the_other(box, tmp_path):
    made = _dx(box, "mkdir /logs/agent/acp_trajectory.jsonl", "agent")
    assert made.returncode == 0, made.stderr

    # Other files are still placed, but a missing required file now refuses
    # verification rather than silently changing the grade.
    with pytest.raises(RuntimeError, match="publication"):
        await _publish_trajectory_for_verifier(
            ContainerEnv(box),
            TRAJECTORY,
            directory(tmp_path / "agent"),
            sandbox_user="agent",
        )

    assert _read(box, "/logs/agent/output.txt") == "Answer: C\n"


# --- #1183 ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_owned_opencode_temp_folder_becomes_the_agents(box):
    """Guards #1183: OpenCode's install leaves /tmp/opencode root's 755."""
    made = _dx(
        box,
        "mkdir -p /tmp/opencode/sub && touch /tmp/opencode/sub/state && "
        "chmod 555 /tmp/opencode/sub && "
        "printf 'secret\\n' > /etc/victim-b && chmod 600 /etc/victim-b && "
        "ln /etc/victim-b /tmp/opencode/hard && ln -s /etc/victim-c /tmp/opencode/soft && "
        "printf 'x\\n' > /etc/victim-c && chmod 600 /etc/victim-c",
    )
    assert made.returncode == 0, made.stderr
    assert _dx(box, "touch /tmp/opencode/t", "agent").returncode != 0

    await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    assert (
        _dx(box, "touch /tmp/opencode/t /tmp/opencode/sub/u", "agent").returncode == 0
    )
    owners = _dx(
        box,
        "stat -c '%n %U' /tmp/opencode /tmp/opencode/sub /tmp/opencode/sub/state "
        "/etc/victim-b /etc/victim-c && stat -c '%n %U' -L /tmp/opencode/soft",
    ).stdout.splitlines()
    assert owners == [
        "/tmp/opencode agent",
        "/tmp/opencode/sub agent",
        "/tmp/opencode/sub/state agent",
        # A hard-linked file and a link's target stay root's.
        "/etc/victim-b root",
        "/etc/victim-c root",
        "/tmp/opencode/soft root",
    ]


@pytest.mark.asyncio
async def test_missing_opencode_temp_folder_is_made_the_agents(box):
    await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    assert _dx(box, "stat -c %U /tmp/opencode").stdout.strip() == "agent"


@pytest.mark.asyncio
async def test_a_link_at_the_temp_folder_refuses_the_trial(box):
    assert (
        _dx(box, "mkdir /etc/victim-d && ln -s /etc/victim-d /tmp/opencode").returncode
        == 0
    )

    with pytest.raises(RuntimeError, match=r"sandbox startup refused.*/tmp/opencode"):
        await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    assert _dx(box, "stat -c %U /etc/victim-d").stdout.strip() == "root"


@pytest.mark.asyncio
async def test_a_root_file_renamed_into_an_agent_made_temp_folder_stays_roots(box):
    """In a scene an earlier role runs as the agent user before a later role's
    OpenCode install: it can make /tmp/opencode itself and rename into it a
    locked file (root's, 600) from a folder it owns. The step must not give
    that file to the agent."""
    made = _dx(
        box,
        "mkdir /app && chown agent:agent /app && printf 'locked\\n' > /app/answer && "
        "chmod 600 /app/answer",
    )
    assert made.returncode == 0, made.stderr
    planted = _dx(box, "mkdir /tmp/opencode && mv /app/answer /tmp/opencode/", "agent")
    assert planted.returncode == 0, planted.stderr

    await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    assert _dx(box, "stat -c '%U %a' /tmp/opencode/answer").stdout.strip() == (
        "root 600"
    )
    assert _dx(box, "cat /tmp/opencode/answer", "agent").returncode != 0
    assert _dx(box, "touch /tmp/opencode/t", "agent").returncode == 0


@pytest.mark.asyncio
async def test_a_leftover_agent_process_cannot_swap_a_link_into_the_walk(box):
    """An earlier role's background process, still alive, swaps entries of
    /tmp/opencode for links to a root-only file as soon as it can write the
    folder. Root's re-own must never follow one."""
    made = _dx(
        box,
        "mkdir /tmp/opencode && for i in $(seq 1 100); do : > /tmp/opencode/f$i; done "
        "&& printf 'x\\n' > /tmp/victim && chmod 600 /tmp/victim",
    )
    assert made.returncode == 0, made.stderr
    # A tight rename loop (the image has Python): every entry turns into a
    # link to the victim, then back into a plain file, round after round, so
    # a walk that lstats an entry and then chowns it by name follows a link.
    attacker = (
        "import os\n"
        "d = '/tmp/opencode'\n"
        "open('/tmp/attacker-ready', 'w').close()\n"
        "while not os.access(d, os.W_OK):\n"
        "    pass\n"
        "link = True\n"
        "while True:\n"
        "    for i in range(1, 101):\n"
        "        p = d + '/f' + str(i)\n"
        "        try:\n"
        "            if link:\n"
        "                os.symlink('/tmp/victim', p + '.n')\n"
        "            else:\n"
        "                open(p + '.n', 'w').close()\n"
        "            os.replace(p + '.n', p)\n"
        "        except OSError:\n"
        "            pass\n"
        "    link = not link\n"
    )
    subprocess.run(
        ["docker", "exec", "-d", "-u", "agent", box, "python3", "-I", "-c", attacker],
        check=True,
        capture_output=True,
    )
    ready = "for i in $(seq 1 100); do [ -e /tmp/attacker-ready ] && exit 0; sleep 0.1; done"
    assert _dx(box, ready + "; exit 1").returncode == 0

    # A refused trial is fine; a re-owned victim is not.
    with contextlib.suppress(RuntimeError):
        await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    assert _dx(box, "stat -c %U /tmp/victim").stdout.strip() == "root"


@pytest.mark.asyncio
async def test_a_temp_folder_another_user_could_fill_is_refused(box):
    """A root-owned /tmp/opencode with a folder in it the agent can write may
    hold files the agent renamed in: the trial is refused, nothing in it
    changes owner."""
    made = _dx(
        box,
        "mkdir -p /tmp/opencode/open && chmod 777 /tmp/opencode/open && "
        "printf 'x\\n' > /tmp/opencode/open/locked && chmod 600 /tmp/opencode/open/locked",
    )
    assert made.returncode == 0, made.stderr

    with pytest.raises(RuntimeError, match=r"/tmp/opencode could hold files"):
        await _prepare_agent_temp_dirs(ContainerEnv(box), AGENTS["opencode"], "agent")

    owners = _dx(
        box, "stat -c %U /tmp/opencode /tmp/opencode/open /tmp/opencode/open/locked"
    ).stdout.split()
    assert owners == ["root", "root", "root"]
