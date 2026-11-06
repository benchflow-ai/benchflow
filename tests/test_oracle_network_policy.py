"""The task's network policy covers the oracle's processes during its turn only.

REPORT B10: with ``network_mode = "allowlist"`` the oracle
ran as root outside the agent's uid firewall and proxy, reached any host, and
so passed tasks whose reference solution needs a host the policy forbids.
The policy is now armed right before solve.sh starts and removed right after
it ends, before the verifier, and it matches the oracle's processes only:
never root as a whole, so BenchFlow's own installs and the verifier keep
their hosts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout._setup import ORACLE_IMAGE_PATH
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.egress_denylist import EGRESS_PORT

PROXY = f"http://127.0.0.1:{EGRESS_PORT}"
# benchflow.sandbox.oracle_egress.ORACLE_EGRESS_GID, written out: the
# supplementary group a root oracle joins while the policy covers it.
ORACLE_EGRESS_GID = 61628


class FakeSandbox:
    """Records every command; answers each with ``answers(cmd)`` (rc 0 by default)."""

    def __init__(self, answers=None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._answers = answers or (lambda cmd: None)

    def _answer(self, cmd: str):
        answer = self._answers(cmd)
        if answer is None and cmd.startswith("id -u"):
            # The image's default user is root unless a test says otherwise.
            return ExecResult(stdout="0\n", stderr="", return_code=0)
        if answer is None and _is_join_probe(cmd):
            # Root joins the policy's group unless a test says otherwise.
            return ExecResult(
                stdout=f"0 {ORACLE_EGRESS_GID}\n", stderr="", return_code=0
            )
        return answer

    async def exec(self, cmd: str, **kwargs: Any) -> ExecResult:
        self.calls.append(("exec", cmd, kwargs))
        answer = self._answer(cmd)
        if isinstance(answer, BaseException):
            raise answer
        return answer or ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, local: Any, remote: str, **kwargs: Any) -> None:
        self.calls.append(("upload", remote, kwargs))

    async def upload_dir(self, local: Any, remote: str, **kwargs: Any) -> None:
        self.calls.append(("upload", remote, kwargs))

    async def download_file(self, remote: str, local: Any) -> None:
        self.calls.append(("download", remote, {}))
        Path(local).write_text("")

    def index(self, predicate) -> int:
        for i, (_kind, cmd, _kwargs) in enumerate(self.calls):
            if predicate(cmd):
                return i
        raise AssertionError(f"no matching call in {[c[1][:80] for c in self.calls]}")


def _is_solve(cmd: str) -> bool:
    return "solve.sh" in cmd and "/logs/agent/oracle.txt" in cmd and "tail" not in cmd


def _is_join_probe(cmd: str) -> bool:
    return "setgroups" in cmd and "id -G" in cmd and "solve.sh" not in cmd


def _is_arm(cmd: str) -> bool:
    return "-I OUTPUT 1" in cmd and ("--suppl-groups" in cmd or "--uid-owner" in cmd)


def _is_disarm(cmd: str) -> bool:
    return "-D OUTPUT" in cmd


def _is_proxy_start(cmd: str) -> bool:
    return "proxy.py" in cmd and "nohup" in cmd


def _is_proxy_stop(cmd: str) -> bool:
    return "rm -rf /opt/runtime-egress" in cmd


def _task(root: Path, network: str = "", extra: str = "") -> Path:
    (root / "solution").mkdir(parents=True)
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho done\n")
    (root / "instruction.md").write_text("Do it.\n")
    (root / "environment").mkdir()
    (root / "task.toml").write_text(
        'version = "1.0"\n' + extra + "[environment]\n" + network
    )
    return root


ALLOWLIST = 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org", "10.1.0.0/16"]\n'
DENYLIST = 'network_mode = "denylist"\nblocked_hosts = ["example.com"]\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("network", [ALLOWLIST, DENYLIST], ids=["allow", "deny"])
async def test_the_policy_is_armed_around_solve_sh_only(tmp_path, network):
    from benchflow.rollout._setup import _run_oracle

    env = FakeSandbox()
    await _run_oracle(env, _task(tmp_path, network), timeout=30)

    start = env.index(_is_proxy_start)
    arm = env.index(_is_arm)
    solve = env.index(_is_solve)
    disarm = env.index(_is_disarm)
    stop = env.index(_is_proxy_stop)
    assert start < arm < solve < disarm < stop
    # Nothing after the stop re-arms anything (the verifier comes next).
    assert not any(_is_arm(cmd) for _, cmd, _ in env.calls[stop:])

    _, solve_cmd, solve_kwargs = env.calls[solve]
    # The oracle's processes take the proxy's path and trust its CA ...
    assert solve_kwargs["env"]["HTTPS_PROXY"] == PROXY
    assert solve_kwargs["env"]["https_proxy"] == PROXY
    assert solve_kwargs["env"]["SSL_CERT_FILE"].endswith("ca-bundle.crt")
    assert not any(key.startswith("BENCHFLOW_EGRESS") for key in solve_kwargs["env"])
    # ... and the root oracle joins the policy's group just before solve.sh.
    assert f" {ORACLE_EGRESS_GID} bash /solution/solve.sh" in solve_cmd
    assert "setgroups" in solve_cmd


@pytest.mark.asyncio
async def test_what_the_oracle_was_refused_lands_in_the_trial_folder(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    env = FakeSandbox()
    trial = tmp_path / "trial"
    await _run_oracle(
        env, _task(tmp_path / "task", ALLOWLIST), timeout=30, rollout_dir=trial
    )

    assert (trial / "trajectory" / "egress_denylist.jsonl").is_file()


@pytest.mark.asyncio
async def test_an_open_task_arms_nothing(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    env = FakeSandbox()
    await _run_oracle(env, _task(tmp_path), timeout=30)

    assert not any(_is_arm(c) or _is_proxy_start(c) for _, c, _ in env.calls)
    _, solve_cmd, solve_kwargs = env.calls[env.index(_is_solve)]
    assert solve_cmd.startswith(ORACLE_IMAGE_PATH + "bash /solution/solve.sh")
    assert "HTTPS_PROXY" not in solve_kwargs["env"]


@pytest.mark.asyncio
async def test_an_agent_allowlist_override_covers_the_oracle_too(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    env = FakeSandbox()
    task = _task(
        tmp_path,
        extra='[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    )
    await _run_oracle(env, task, timeout=30)

    assert env.index(_is_arm) < env.index(_is_solve) < env.index(_is_disarm)


@pytest.mark.asyncio
async def test_a_non_root_task_user_is_matched_by_its_uid(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if cmd.startswith("id -u "):
            return ExecResult(stdout="1001\n", stderr="", return_code=0)
        return None

    env = FakeSandbox(answers)
    task = _task(tmp_path, ALLOWLIST, extra='[agent]\nuser = "builder"\n')
    await _run_oracle(env, task, timeout=30)

    arm = env.calls[env.index(_is_arm)][1]
    assert "--uid-owner 1001" in arm
    assert "--gid-owner" not in arm
    solve_cmd = env.calls[env.index(_is_solve)][1]
    assert solve_cmd.startswith("su -s /bin/bash builder -c ")
    assert "setgroups" not in solve_cmd
    assert "--uid-owner 1001" in env.calls[env.index(_is_disarm)][1]


@pytest.mark.asyncio
async def test_an_image_whose_user_is_not_root_is_matched_by_its_uid(tmp_path):
    """The oracle's exec names no user, so it runs as the image's USER; a
    non-root process cannot join a group, so its uid is matched."""
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if cmd == "id -u":
            return ExecResult(stdout="1000\n", stderr="", return_code=0)
        return None

    env = FakeSandbox(answers)
    await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)

    uid_call = next(c for c in env.calls if c[1] == "id -u")
    assert "user" not in uid_call[2]
    assert "--uid-owner 1000" in env.calls[env.index(_is_arm)][1]
    solve_cmd = env.calls[env.index(_is_solve)][1]
    assert solve_cmd.startswith(ORACLE_IMAGE_PATH + "bash /solution/solve.sh")
    assert "setgroups" not in solve_cmd


@pytest.mark.asyncio
async def test_a_root_task_user_joins_the_group_inside_su(tmp_path):
    """su resets supplementary groups, so the join happens inside su's shell."""
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if cmd.startswith("id -u "):
            return ExecResult(stdout="0\n", stderr="", return_code=0)
        return None

    env = FakeSandbox(answers)
    task = _task(tmp_path, ALLOWLIST, extra='[agent]\nuser = "root"\n')
    await _run_oracle(env, task, timeout=30)

    solve_cmd = env.calls[env.index(_is_solve)][1]
    assert solve_cmd.startswith("su -s /bin/bash root -c ")
    inner = solve_cmd.split(" -c ", 1)[1]
    assert "setgroups" in inner and str(ORACLE_EGRESS_GID) in inner


@pytest.mark.asyncio
async def test_a_policy_that_cannot_be_armed_never_runs_the_oracle(tmp_path):
    """Fail closed: no iptables, no nat table, a taken group id."""
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if _is_arm(cmd):
            return ExecResult(stdout="", stderr="no iptables", return_code=86)
        return None

    env = FakeSandbox(answers)
    with pytest.raises(RuntimeError, match="could not be armed"):
        await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)

    assert not any(_is_solve(c) for _, c, _ in env.calls)
    # Whatever was added is taken away again, and the proxy stopped.
    assert env.index(_is_disarm) < env.index(_is_proxy_stop)


@pytest.mark.asyncio
async def test_the_policy_is_removed_when_solve_sh_times_out(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if _is_solve(cmd):
            return TimeoutError("oracle timed out")
        return None

    env = FakeSandbox(answers)
    with pytest.raises(TimeoutError):
        await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)

    assert env.index(_is_solve) < env.index(_is_disarm) < env.index(_is_proxy_stop)


@pytest.mark.asyncio
async def test_rules_that_stay_armed_stop_the_trial_before_the_verifier(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if _is_disarm(cmd):
            return ExecResult(stdout="", stderr="", return_code=4)
        return None

    env = FakeSandbox(answers)
    with pytest.raises(RuntimeError, match="could not be removed"):
        await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    [
        # setgroups without CAP_SETGID (a compose with cap_drop: [ALL]; the
        # net-admin overlay adds NET_ADMIN back, so the rules still arm).
        ExecResult(
            stdout="",
            stderr="PermissionError: [Errno 1] Operation not permitted",
            return_code=1,
        ),
        # The join ran but the group is not there.
        ExecResult(stdout="0\n", stderr="", return_code=0),
    ],
    ids=["setgroups-refused", "group-missing"],
)
async def test_a_root_oracle_that_cannot_join_the_group_errors_the_trial(
    tmp_path, probe
):
    """R3-R1-A-2: the join used to fail inside solve.sh's own command, so
    solve.sh never ran, only rc=1 was logged, and the verifier scored the
    untouched workspace 0 instead of the trial erroring."""
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        return probe if _is_join_probe(cmd) else None

    env = FakeSandbox(answers)
    with pytest.raises(RuntimeError, match=f"cannot join group {ORACLE_EGRESS_GID}"):
        await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)

    assert not any(_is_solve(c) for _, c, _ in env.calls)
    # The probe ran after the rules were armed, as the oracle's own exec would
    # (the sandbox's default user), and what was armed is taken away again.
    probe_at = env.index(_is_join_probe)
    assert env.index(_is_arm) < probe_at < env.index(_is_disarm)
    assert "user" not in env.calls[probe_at][2]
    assert env.index(_is_disarm) < env.index(_is_proxy_stop)


@pytest.mark.asyncio
async def test_a_root_task_user_is_probed_inside_su(tmp_path):
    """su resets supplementary groups, so the probe joins inside su's shell,
    the way the oracle's command does."""
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if cmd.startswith("id -u "):
            return ExecResult(stdout="0\n", stderr="", return_code=0)
        return None

    env = FakeSandbox(answers)
    task = _task(tmp_path, ALLOWLIST, extra='[agent]\nuser = "root"\n')
    await _run_oracle(env, task, timeout=30)

    probe = env.calls[env.index(_is_join_probe)][1]
    assert probe.startswith("su -s /bin/bash root -c ")
    assert env.index(_is_join_probe) < env.index(_is_solve)


@pytest.mark.asyncio
async def test_an_oracle_matched_by_its_uid_joins_no_group(tmp_path):
    from benchflow.rollout._setup import _run_oracle

    def answers(cmd):
        if cmd == "id -u":
            return ExecResult(stdout="1000\n", stderr="", return_code=0)
        return None

    env = FakeSandbox(answers)
    await _run_oracle(env, _task(tmp_path, ALLOWLIST), timeout=30)

    assert not any(_is_join_probe(c) for _, c, _ in env.calls)
