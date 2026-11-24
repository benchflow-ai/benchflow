"""A workspace the sandbox user's grant made root's and sticky 1777, graded by a root test.sh (#1249 (b), (c)), and
the verifier's folder against an agent whose uid is the host's (#1247), as whole trials on real Docker.

The grant makes a workspace root's and 1777 when it holds a provider daemon's folder (``/root/.daytona``, as on a
Daytona sandbox) or a folder BenchFlow reserves (``/opt`` holds ``/opt/runtime``), and gives every entry in it to
the sandbox user. Under ``fs.protected_regular`` (Daytona and stock Ubuntu run 2) root can't ``O_CREAT``-open a file
another user owns in such a folder, so a root test.sh that rewrote an agent-owned file in place got EACCES and the
trial scored 0 with no error, in integrity audit and off (0.10.0 as tagged; strict ran test.sh in a separate
verifier). The rows are skipped where the kernel's ``fs.protected_regular`` is 0.

Harness: the grading-integrity matrix's (``tests/test_grading_integrity_matrix.py``): ``_task``, ``_run``, the
``bfprobe`` model-agent stand-in that runs the task's payload as the sandbox user, and its readers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The matrix's own autouse fixtures come too: direct Docker as a run has it, and never skipped without Docker.
from tests.test_grading_integrity_matrix import (  # noqa: F401
    _LOGS,
    _PY,
    AGENT_PROBE,
    R,
    _category,
    _direct_docker_as_a_run,
    _docker_or_fail,
    _payload_log,
    _precondition,
    _provenance,
    _result,
    _reward,
    _run,
    _seen,
    _task,
    _trials,
    _why,
)

pytestmark = [
    pytest.mark.deterministic,
    pytest.mark.verifier_runtime,
    pytest.mark.grading_matrix,
    pytest.mark.bfx_refusal,
]

DOCKER_RUN_DEFAULTS = True


def _protected_regular() -> int:
    try:
        return int(Path("/proc/sys/fs/protected_regular").read_text().strip())
    except (OSError, ValueError):
        return 0


_NEEDS_PROTECTED = pytest.mark.skipif(
    _protected_regular() < 1, reason="the kernel's fs.protected_regular is 0"
)

#: ``/root`` holds a daemon folder, as on a Daytona sandbox; the input comes with the image.
IMAGE_ROOT = (
    f"FROM {_PY}\nWORKDIR /root\n{_LOGS}"
    "RUN mkdir -p /root/.daytona/sessions && chmod 700 /root && echo old > /root/input.txt\n"
)
#: ``/opt`` holds BenchFlow's reserved ``/opt/runtime``: no daemon folder, plain Docker.
IMAGE_OPT = f"FROM {_PY}\nWORKDIR /opt\n{_LOGS}RUN chmod 755 /opt && echo old > /opt/input.txt\n"
#: ``/usr/local`` holds the loader's ``/usr/local/lib*``, reserved the same way.
IMAGE_USR_LOCAL = (
    f"FROM {_PY}\nWORKDIR /usr/local\n{_LOGS}"
    "RUN chmod 755 /usr/local && echo old > /usr/local/input.txt\n"
)
_WORKSPACES = {
    "root": ("/root", IMAGE_ROOT),
    "opt": ("/opt", IMAGE_OPT),
    "usrlocal": ("/usr/local", IMAGE_USR_LOCAL),
}


def _rewrite_test_sh(workspace: str) -> str:
    """Root's test.sh: record what it sees, then rewrite the agent-owned input in place (``>``: O_CREAT)."""
    return (
        "#!/bin/sh\n"
        f"{{ stat -c 'ws %U %a' {workspace}; stat -c 'file %U' {workspace}/input.txt; echo \"id $(id -u)\"; }} "
        "> /logs/verifier/seen.txt 2>&1\n"
        f"if printf 'checked\\n' > {workspace}/input.txt; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    )


def _graded_as_granted(trial: Path, workspace: str) -> None:
    """The row's precondition: test.sh ran as root on an input the grant gave the agent user."""
    seen = _seen(trial)
    assert "file agent" in seen and "id 0" in seen, (seen, _why(trial))


# == A root test.sh in audit and off =============================================================================


@_NEEDS_PROTECTED
@pytest.mark.parametrize("mode", ["audit", "off"])
@pytest.mark.parametrize("agent", ["nop", "oracle", AGENT_PROBE])
@pytest.mark.parametrize("where", ["root", "opt", "usrlocal"])
async def test_a_root_test_sh_rewrites_an_agent_owned_file_in_a_sticky_workspace(
    tmp_path, mode, agent, where
):
    """Audit and off, in ``/root``, ``/opt`` and ``/usr/local``. The workspace is root's and 1777 from the grant (a
    daemon folder in ``/root``, the reserved ``/opt/runtime`` in ``/opt``), the input is the agent user's. Before a
    root test.sh in audit and off the workspace loses group and other write (1755, sticky kept): root's in-place
    rewrite works, and the trial scores test.sh's 1.

    Mutants: skip the step in audit; skip it in off; record the mode only for a daemon folder (``/opt``,
    ``/usr/local``)."""
    workspace, image = _WORKSPACES[where]
    task = _task(
        tmp_path / "tasks",
        f"b-{where}-{mode}-{agent}",
        image=image,
        test_sh=_rewrite_test_sh(workspace),
    )
    jobs = await _run([task], tmp_path / "jobs", agent=agent, user="agent", mode=mode)
    trial = _trials(jobs)[task.name]
    _graded_as_granted(trial, workspace)
    assert _reward(trial) == 1.0, _why(trial)
    assert "ws root 1755" in _seen(trial), _seen(trial)
    if mode == "audit":
        # The step's record beside the close's (the reward's provenance).
        assert _provenance(trial).get("workspace_modes_before_test") == {
            workspace: "1755"
        }, _why(trial)


# == Strict =======================================================================================================


@_NEEDS_PROTECTED
async def test_strict_still_grades_a_root_workspace_in_the_image_mode(tmp_path):
    """Guard. Strict is unchanged: test.sh runs in a separate verifier, where ``/root`` has the image's
    mode (700, root's) and the transferred input is root's; it scores 1."""
    task = _task(
        tmp_path / "tasks",
        "b3-strict",
        image=IMAGE_ROOT,
        test_sh=_rewrite_test_sh("/root"),
    )
    jobs = await _run(
        [task], tmp_path / "jobs", agent="nop", user="agent", mode="strict"
    )
    trial = _trials(jobs)[task.name]
    assert _reward(trial) == 1.0, _why(trial)
    assert "ws root 700" in _seen(trial), _seen(trial)


async def test_strict_still_refuses_an_opt_workspace_by_name(tmp_path):
    """Guard, real kernel. Strict's separate verifier owns ``/opt/runtime``: a workspace of ``/opt`` is
    refused with that reason, unscored, never a quiet 0."""
    task = _task(
        tmp_path / "tasks",
        "c2-strict-opt",
        image=IMAGE_OPT,
        test_sh=_rewrite_test_sh("/opt"),
    )
    jobs = await _run(
        [task], tmp_path / "jobs", agent="nop", user="agent", mode="strict"
    )
    trial = _trials(jobs)[task.name]
    result = _result(trial)
    assert _reward(trial) is None, _why(trial)
    text = f"{result.get('error')} {result.get('verifier_error')}"
    assert "/opt overlaps verifier-owned /opt/runtime" in text, _why(trial)


# == A kept service as the agent user ============================================================================

_KEPT_APPENDER = (
    "echo start > /root/svc.log\n"
    "cat > /tmp/svc.py <<'EOF'\n"
    "import http.server\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        try:\n"
    "            with open('/root/svc.log', 'a') as f:\n"
    "                f.write('graded\\n')\n"
    "            body, code = open('/root/svc.log', 'rb').read(), 200\n"
    "        except Exception as e:\n"
    "            body, code = repr(e).encode(), 500\n"
    "        self.send_response(code)\n"
    "        self.end_headers()\n"
    "        self.wfile.write(body)\n"
    "    def log_message(self, *a):\n"
    "        pass\n"
    "http.server.HTTPServer(('127.0.0.1', 5000), H).serve_forever()\n"
    "EOF\n"
    "setsid nohup python3 /tmp/svc.py </dev/null >/dev/null 2>&1 &\n"
    "sleep 1\n"
    "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:5000/\", timeout=5)'"
    " && echo SERVING\n"
)


@_NEEDS_PROTECTED
async def test_a_kept_service_reads_and_appends_its_own_file_in_the_workspace_while_graded(
    tmp_path,
):
    """A service the agent left running (``x-keep-services``) as the agent user reads and appends to its
    own file directly in ``/root`` during audit grading: ``go-w`` keeps read and traverse for it, where the image's
    700 would not. test.sh also rewrites the agent-owned input, as root.

    Mutant: restore the image's mode (700) instead of ``go-w``."""
    test_sh = (
        "#!/bin/sh\n"
        "ok=0\n"
        "python3 - <<'EOF' && ok=1\n"
        "import sys, urllib.request\n"
        "r = urllib.request.urlopen('http://127.0.0.1:5000/', timeout=10)\n"
        "sys.exit(0 if r.status == 200 and b'start' in r.read() else 1)\n"
        "EOF\n"
        'echo "service $ok" > /logs/verifier/seen.txt\n'
        f"if [ $ok = 1 ] && printf 'checked\\n' > /root/input.txt; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    )
    task = _task(
        tmp_path / "tasks",
        "b4-kept",
        image=IMAGE_ROOT,
        test_sh=test_sh,
        payload=_KEPT_APPENDER,
        keep_services=[5000],
    )
    jobs = await _run([task], tmp_path / "jobs", agent=AGENT_PROBE, user="agent")
    trial = _trials(jobs)[task.name]
    _precondition(trial, "SERVING")
    assert "service 1" in _seen(trial), _why(trial)
    assert _reward(trial) == 1.0, _why(trial)


# == A verify, then another agent turn ===========================================================================

_STEPS = """
multi_step_reward_strategy = "final"

[[steps]]
name = "one"
min_reward = 1.0

[[steps]]
name = "two"
"""

_STEP_PAYLOAD = (
    "n=$(cat /tmp/bf-step 2>/dev/null || echo 0); n=$((n+1)); echo $n > /tmp/bf-step\n"
    'if touch /root/made-$n; then echo "MADE-$n"; else echo "NOTMADE-$n"; fi >> /logs/agent/b5-steps.txt\n'
)


@_NEEDS_PROTECTED
async def test_the_next_steps_turn_can_create_a_file_in_the_workspace_after_a_verify(
    tmp_path,
):
    """A multi-step task: step one's root test.sh (a soft verify) rewrites the agent-owned input, so the
    workspace is ``go-w`` while it runs; step two's agent turn then creates a file directly in ``/root`` again (the
    grant's 1777 is back before the turn), and step two's test.sh finds it.

    Mutant: no re-open after the verify."""
    two = (
        "#!/bin/sh\n"
        "ls -la /root > /logs/verifier/seen.txt 2>&1\n"
        f"if [ -f /root/made-2 ] && printf 'two\\n' > /root/input.txt; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    )
    task = _task(
        tmp_path / "tasks",
        "b5-steps",
        image=IMAGE_ROOT,
        test_sh=two,
        payload=_STEP_PAYLOAD,
    )
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace('version = "1.0"\n', 'version = "1.0"\n' + _STEPS, 1)
    )
    for name, test_sh in (("one", _rewrite_test_sh("/root")), ("two", two)):
        step = task / "steps" / name
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text(f"Step {name}.\n")
        (step / "tests" / "test.sh").write_text(test_sh)
        (step / "tests" / "test.sh").chmod(0o755)
    jobs = await _run([task], tmp_path / "jobs", agent=AGENT_PROBE, user="agent")
    trial = _trials(jobs)[task.name]
    steps = (trial / "agent" / "b5-steps.txt").read_text().split()
    assert steps == ["MADE-1", "MADE-2"], (steps, _why(trial))
    assert _reward(trial) == 1.0, _why(trial)


# == A verifier user of the task's own ===========================================================================


@_NEEDS_PROTECTED
async def test_a_non_root_verifier_keeps_the_grant_and_writes_its_reward(tmp_path):
    """Real kernel, with guards. test.sh runs as the task's ``[verifier] user``: the workspace stays as the
    grant made it (1777), so that user can still create a file directly in it, and ``/logs/verifier`` is its own
    for ``reward.txt``.

    Mutants: ``go-w`` for every verifier user; no chown of the folder before test.sh."""
    test_sh = (
        "#!/bin/sh\n"
        f"if touch /root/by-checker; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    )
    task = _task(
        tmp_path / "tasks",
        "b6-checker",
        image=IMAGE_ROOT + "RUN useradd -m -u 4321 checker\n",
        test_sh=test_sh,
        verifier_user="checker",
    )
    jobs = await _run([task], tmp_path / "jobs", agent="nop", user="agent")
    trial = _trials(jobs)[task.name]
    assert _reward(trial) == 1.0, _why(trial)


# == An agent with the host's uid ================================================================================

_PLANTER = (
    'id -u > /tmp/my-uid; echo "UID $(id -u)"\n'
    "echo mine > /app/made-by-agent && echo WROTE-APP\n"
    "cat > /tmp/plant.sh <<'EOF'\n"
    "i=0; while [ $i -lt 6000 ]; do : > /logs/verifier/planted-$i 2>/dev/null; i=$((i+1)); sleep 0.05; done\n"
    "EOF\n"
    "setsid nohup sh /tmp/plant.sh </dev/null >/dev/null 2>&1 &\n"
)
_D5_TEST_SH = (
    "#!/bin/sh\n"
    f'if [ "$(cat /app/made-by-agent)" = mine ]; then echo 1 > {R}; else echo 0 > {R}; fi\n'
    "sleep 2\n"
)


def _host_uid() -> int:
    return os.getuid()


@pytest.mark.skipif(
    _host_uid() == 0, reason="BenchFlow runs as root: no uid to collide with"
)
async def test_an_images_own_agent_user_with_the_host_uid_is_refused(tmp_path):
    """Refused. The image's own account, named as the sandbox user, has the uid BenchFlow runs as on the
    host, and ``/logs/verifier`` is the host's bind-mounted folder: once test.sh exits the folder goes back to that
    uid, which the agent's still-running processes share. The trial is refused at setup with the reason named,
    before the agent's turn: unscored, never a quiet 0.

    Mutant: drop the check."""
    task = _task(
        tmp_path / "tasks",
        "d5-own-user",
        image=f"FROM {_PY}\nWORKDIR /app\n{_LOGS}RUN useradd -m -u {_host_uid()} worker\n",
        test_sh=_D5_TEST_SH,
        payload=_PLANTER,
    )
    jobs = await _run([task], tmp_path / "jobs", agent=AGENT_PROBE, user="worker")
    trial = _trials(jobs)[task.name]
    assert _reward(trial) is None, _why(trial)
    assert _category(trial) == "integrity_unavailable", _why(trial)
    assert str(_host_uid()) in (_result(trial).get("error") or ""), _why(trial)
    assert "UID" not in _payload_log(trial), "the agent's turn ran"


@pytest.mark.skipif(
    _host_uid() < 1000,
    reason="the host's uid is below the range BenchFlow gives its sandbox user",
)
async def test_the_default_agent_user_never_gets_the_host_uid(tmp_path):
    """Remapped. BenchFlow's own ``agent`` user takes the first uid from 1000 nothing in the image names;
    here the image names every uid below the host's, so that would be the host's own (a test VM: ``ubuntu:24.04`` and
    host uid 1001; a host whose user is 1000 and ``python:3.12-slim``). The user is never refused: it gets another
    uid, still owns its workspace (the grant goes by name), and nothing it leaves running can add a name to the
    bind-mounted ``/logs/verifier`` after test.sh. (The uid inventory already skips the host-owned log folders'
    uid here, so "no remap" doesn't turn this row red; the kernel remap row in
    ``test_verifier_dir_closed_to_agent.py`` does.)

    Mutants: refuse the default user; check every user's uid."""
    taken = "".join(
        f"RUN groupadd -g {u} u{u} && useradd -M -u {u} -g {u} u{u}\n"
        for u in range(1000, _host_uid())
    )
    task = _task(
        tmp_path / "tasks",
        "d5-default-user",
        image=f"FROM {_PY}\nWORKDIR /app\n{_LOGS}{taken}",
        test_sh=_D5_TEST_SH,
        payload=_PLANTER,
    )
    jobs = await _run([task], tmp_path / "jobs", agent=AGENT_PROBE, user="agent")
    trial = _trials(jobs)[task.name]
    _precondition(trial, "WROTE-APP")
    uid = next(
        line.split()[1]
        for line in _payload_log(trial).splitlines()
        if line.startswith("UID ")
    )
    assert uid != str(_host_uid()), f"the agent ran as the host's uid {uid}"
    # The trial records the uid the sandbox user got (result.json's sandbox_account).
    account = _result(trial).get("sandbox_account") or {}
    assert (
        account.get("uid") == int(uid) and account.get("existed_before_setup") is False
    ), account
    assert _reward(trial) == 1.0, _why(trial)
    planted = sorted(p.name for p in (trial / "verifier").glob("planted-*"))
    assert not planted, planted[:5]


# == The oracle ======================================================================================================

_P2_SOLVE = (
    "#!/bin/sh\n"
    "runuser -u agent -- sh -c 'echo appended >> /root/input.txt' && echo AGENT-APPEND-OK > /logs/agent/p2.txt "
    "|| echo AGENT-APPEND-FAILED > /logs/agent/p2.txt\n"
    "if printf 'new\\n' >> /root/input.txt; then echo ROOT-WRITE-OK >> /logs/agent/p2.txt; "
    "else echo ROOT-WRITE-FAILED >> /logs/agent/p2.txt; fi\n"
)


def _agent_file(trial: Path, name: str) -> list[str]:
    path = Path(trial) / "agent" / name
    return path.read_text().split() if path.is_file() else []


@_NEEDS_PROTECTED
async def test_an_oracle_with_a_verifier_user_of_its_own_keeps_the_grant(tmp_path):
    """An oracle run of the verifier-user task above: test.sh runs as the task's ``[verifier] user`` and creates
    a file directly in ``/root``. The oracle's step before its root solve.sh takes only group and other write and
    gives the grant's 1777 back after, so that verifier meets the workspace as in an agent trial (1777).

    Mutant: give the image's mode (700) back before solve.sh (9cacfab5's restore)."""
    test_sh = (
        "#!/bin/sh\n"
        "stat -c 'ws %U %a' /root > /logs/verifier/seen.txt 2>&1\n"
        f"if touch /root/by-checker; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    )
    task = _task(
        tmp_path / "tasks",
        "p1-oracle-checker",
        image=IMAGE_ROOT + "RUN useradd -m -u 4321 checker\n",
        test_sh=test_sh,
        verifier_user="checker",
    )
    jobs = await _run([task], tmp_path / "jobs", agent="oracle", user="agent")
    trial = _trials(jobs)[task.name]
    assert "ws root 1777" in _seen(trial), _why(trial)
    assert _reward(trial) == 1.0, _why(trial)


@_NEEDS_PROTECTED
async def test_a_root_oracle_that_runs_a_step_as_the_sandbox_user(tmp_path):
    """A root solve.sh drops to the sandbox user to append to that user's file in ``/root`` (the
    matrix's own ``_SOLVE`` pattern), then appends to it as root. With ``go-w`` (1755) both work: the agent user can
    traverse and append, root can open the file.

    Mutant: give the image's mode (700) back before solve.sh."""
    test_sh = (
        "#!/bin/sh\n"
        f"if grep -q appended /root/input.txt && grep -q new /root/input.txt; then echo 1 > {R}; "
        f"else echo 0 > {R}; fi\n"
    )
    task = _task(
        tmp_path / "tasks", "p2-oracle-runuser", image=IMAGE_ROOT, test_sh=test_sh
    )
    (task / "solution" / "solve.sh").write_text(_P2_SOLVE)
    jobs = await _run([task], tmp_path / "jobs", agent="oracle", user="agent")
    trial = _trials(jobs)[task.name]
    assert _agent_file(trial, "p2.txt") == ["AGENT-APPEND-OK", "ROOT-WRITE-OK"], _why(
        trial
    )
    assert _reward(trial) == 1.0, _why(trial)


@_NEEDS_PROTECTED
async def test_a_root_oracles_sandbox_user_step_cannot_create_directly_in_the_workspace(
    tmp_path,
):
    """Recorded, not blocked (11-20). The one oracle shape ``go-w`` doesn't allow: a root solve.sh whose
    step as the sandbox user creates a *new* name directly in ``/root`` (with the grant's 1777 it could; 9cacfab5:
    700, it couldn't). The trial scores test.sh's 0, with no error; a solve.sh that writes as root, or into a
    subfolder, is unaffected. This row pins the outcome so a change to it is seen."""
    solve = (
        "#!/bin/sh\n"
        "runuser -u agent -- touch /root/new-by-agent && echo AGENT-CREATE-OK > /logs/agent/p2b.txt "
        "|| echo AGENT-CREATE-FAILED > /logs/agent/p2b.txt\n"
    )
    test_sh = f"#!/bin/sh\nif [ -f /root/new-by-agent ]; then echo 1 > {R}; else echo 0 > {R}; fi\n"
    task = _task(
        tmp_path / "tasks",
        "p2b-oracle-runuser-create",
        image=IMAGE_ROOT,
        test_sh=test_sh,
    )
    (task / "solution" / "solve.sh").write_text(solve)
    jobs = await _run([task], tmp_path / "jobs", agent="oracle", user="agent")
    trial = _trials(jobs)[task.name]
    assert _agent_file(trial, "p2b.txt") == ["AGENT-CREATE-FAILED"], _why(trial)
    assert _reward(trial) == 0.0, _why(trial)


_KEPT_CREATOR = (
    "cat > /tmp/svc.py <<'EOF'\n"
    "import http.server, sqlite3\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        try:\n"
    "            db = sqlite3.connect('/root/app.db')\n"
    "            db.execute('create table if not exists t (x)')\n"
    "            db.execute('insert into t values (1)')\n"
    "            db.commit()\n"
    "            n = db.execute('select count(*) from t').fetchone()[0]\n"
    "            db.close()\n"
    "            body, code = f'rows {n}'.encode(), 200\n"
    "        except Exception as e:\n"
    "            body, code = repr(e).encode(), 500\n"
    "        self.send_response(code)\n"
    "        self.end_headers()\n"
    "        self.wfile.write(body)\n"
    "    def log_message(self, *a):\n"
    "        pass\n"
    "http.server.HTTPServer(('127.0.0.1', 5000), H).serve_forever()\n"
    "EOF\n"
    "setsid nohup python3 /tmp/svc.py </dev/null >/dev/null 2>&1 &\n"
    "sleep 1\n"
    "python3 -c 'import urllib.request; print(urllib.request.urlopen(\"http://127.0.0.1:5000/\", timeout=5).read())'"
    " && echo SERVING\n"
)


@_NEEDS_PROTECTED
async def test_a_kept_service_can_no_longer_create_names_directly_in_the_workspace_while_graded(
    tmp_path,
):
    """The accepted behaviour change (11-20; CHANGELOG "Changed"). A service the agent left
    running, as the agent user, that creates new names directly in ``/root`` while graded (an sqlite database's
    journal): with ``go-w`` before a root test.sh it can't, and test.sh's check of it fails (with 1777 it scored
    1.0). The kept-service row keeps read and append. This guard pins the new outcome.

    Mutant: skip the step (the service creates its journal again)."""
    test_sh = (
        "#!/bin/sh\n"
        "stat -c 'ws %U %a' /root > /logs/verifier/seen.txt 2>&1\n"
        "python3 - >> /logs/verifier/seen.txt 2>&1 <<'EOF' && ok=1 || ok=0\n"
        "import sys, urllib.request, urllib.error\n"
        "try:\n"
        "    r = urllib.request.urlopen('http://127.0.0.1:5000/', timeout=10)\n"
        "    b = r.read(); print('svc', r.status, b)\n"
        "    sys.exit(0 if r.status == 200 and b.startswith(b'rows') else 1)\n"
        "except urllib.error.HTTPError as e:\n"
        "    print('svc', e.code, e.read()); sys.exit(1)\n"
        "EOF\n"
        f"echo $ok > {R}\n"
    )
    task = _task(
        tmp_path / "tasks",
        "p3-kept-creator",
        image=IMAGE_ROOT,
        test_sh=test_sh,
        payload=_KEPT_CREATOR,
        keep_services=[5000],
    )
    jobs = await _run([task], tmp_path / "jobs", agent=AGENT_PROBE, user="agent")
    trial = _trials(jobs)[task.name]
    _precondition(trial, "SERVING")
    seen = _seen(trial)
    assert "ws root 1755" in seen and "svc 500" in seen, _why(trial)
    assert _reward(trial) == 0.0, _why(trial)


_ORACLE_STEP_SOLVE = (
    "#!/bin/sh\n"
    "n=$(cat /tmp/bf-step 2>/dev/null || echo 0); n=$((n+1)); echo $n > /tmp/bf-step\n"
    'if touch /root/made-$n; then echo "MADE-$n"; else echo "NOTMADE-$n"; fi >> /logs/agent/steps.txt\n'
    # The agent user's input, rewritten in place: as root this needs the step before solve.sh.
    'if printf "step $n\\n" > /root/input.txt; then echo "WROTE-$n"; else echo "NOTWROTE-$n"; fi'
    " >> /logs/agent/steps.txt\n"
)


@_NEEDS_PROTECTED
@pytest.mark.parametrize("oracle_user", ["agent", "root"])
async def test_a_multi_step_oracle_creates_in_the_workspace_after_each_verify(
    tmp_path, oracle_user
):
    """A multi-step task run by the oracle (``Rollout._run_scripted_steps`` → ``_solve_step`` →
    ``_run_oracle``, not the user loop of the verify-then-turn row): each step's solve.sh creates a file directly in
    ``/root``, then the step's root test.sh (a soft verify, which takes group and other write) checks it and rewrites
    the input as root.
    As the task's ``[agent].user``, step two's solve.sh gets the grant's 1777 back first; as root, each solve.sh meets
    the workspace without group or other write and the grant's mode comes back after it.

    Mutants: no re-open for an [agent].user oracle; no step before a root solve.sh."""

    def test_sh(n: int) -> str:
        return (
            "#!/bin/sh\n"
            f"if [ -f /root/made-{n} ] && printf 'checked\\n' > /root/input.txt; then echo 1 > {R}; "
            f"else echo 0 > {R}; fi\n"
        )

    task = _task(
        tmp_path / "tasks",
        f"s4-steps-oracle-{oracle_user}",
        image=IMAGE_ROOT,
        test_sh=test_sh(2),
    )
    toml = task / "task.toml"
    text = toml.read_text().replace(
        'version = "1.0"\n', 'version = "1.0"\n' + _STEPS, 1
    )
    if oracle_user == "agent":
        text = text.replace("[agent]\n", '[agent]\nuser = "agent"\n', 1)
    toml.write_text(text)
    (task / "solution" / "solve.sh").write_text(_ORACLE_STEP_SOLVE)
    for n, name in ((1, "one"), (2, "two")):
        step = task / "steps" / name
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text(f"Step {name}.\n")
        (step / "tests" / "test.sh").write_text(test_sh(n))
        (step / "tests" / "test.sh").chmod(0o755)
    jobs = await _run([task], tmp_path / "jobs", agent="oracle", user="agent")
    trial = _trials(jobs)[task.name]
    assert _agent_file(trial, "steps.txt") == [
        "MADE-1",
        "WROTE-1",
        "MADE-2",
        "WROTE-2",
    ], _why(trial)
    assert _reward(trial) == 1.0, _why(trial)
