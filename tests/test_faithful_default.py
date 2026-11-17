"""The faithful default (the lead's 09:58Z and 10:01Z rulings).

Integrity ``audit`` is the default, and in ``off`` and ``audit`` the verifier
runs the task's test.sh as its authors wrote it: no conftest sweep, no
PYTEST_ADDOPTS, no plugin guard, no PATH judge, no BASH_ENV, pip, git, Python
or loader pins, no workspace freeze, no /tmp purge, no cache reclaim. What
every mode keeps (the agent-process kill, the network re-proof, the delegate
restore, the metadata cover, the output clear, the bounded launch and the
verifier network) is unchanged. ``strict`` keeps all of it.

The detector diffs the sandbox after the turn against the image manifest
recorded before it and writes ``integrity/verifier-inputs.json``; a flag never
changes the reward. Real Docker trials, skipped without a daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._verifier_env import pinned_shell_prefix
from benchflow.integrity.constants import normalize_integrity_mode
from tests.exec_diet_fakes import PYTHON, docker_available

# -- the mode and its default --------------------------------------------------


def test_audit_is_the_default_and_off_is_an_explicit_opt_out():
    from benchflow.rollout import RolloutConfig

    assert normalize_integrity_mode(None) == "audit"
    assert normalize_integrity_mode(False) == "off"
    assert normalize_integrity_mode("strict") == "strict"
    with pytest.raises(ValueError):
        normalize_integrity_mode("shared")  # the verifier sandbox's word, not a mode
    assert RolloutConfig.__dataclass_fields__["integrity"].default == "audit"


def test_test_shs_command_gets_no_pins_when_it_runs_as_written():
    assert pinned_shell_prefix(None, faithful=True) == ""
    pinned = pinned_shell_prefix(None)
    assert "export PIP_CONFIG_FILE=/dev/null" in pinned and "unset " in pinned


def test_the_detector_has_its_timing_name():
    from benchflow._utils import timing_spans

    assert "detect_verifier_inputs" in timing_spans.HARDEN_STEPS


# -- real trials ---------------------------------------------------------------

# The real trials are marked deterministic, as the repository's other Docker
# trials are: run them with ``-m deterministic``.

# The image: a repository whose tests use a fixture from tests/conftest.py, an
# image library path, an image PATH folder anyone may write, and root's caches.
DOCKERFILE = f"""\
FROM {PYTHON}
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
RUN pip install --no-cache-dir -q pytest==8.4.1
RUN mkdir -p /app/tests /opt/img-lib /root/.cache/pip/http && mkdir -m 777 /opt/anybin \\
    && touch /root/.cache/pip/http/kept && printf '%s\\n' \\
    'import pytest' '' '' '@pytest.fixture' 'def app():' '    return "repo"' \\
    > /app/tests/conftest.py && printf '%s\\n' \\
    'from pathlib import Path' '' '' 'def test_fix(app):' \\
    '    assert app == "repo"' \\
    '    assert Path("/app/fixed.txt").read_text().strip() == "fixed"' \\
    > /app/tests/test_repo.py
ENV LD_LIBRARY_PATH=/opt/img-lib PATH=/opt/anybin:$PATH
"""

SOLVE = "echo fixed > /app/fixed.txt\n"

# What the agent leaves behind for test.sh, as the agent user.
LEFT_BEHIND = (
    "mkdir -p /app/sub /app/pkg/__pycache__ && echo 'X = 1' > /app/sub/conftest.py"
    " && ln -s fixed.txt /app/link && echo bytes > /app/pkg/__pycache__/m.pyc"
    " && echo 'x = 1' > /tmp/planted.py && echo mine > /app/mine.txt"
    " && chown agent:agent /app/mine.txt\n"
)

# test.sh records its environment and the files, then runs the tests.
RECORDING_TEST = """\
#!/bin/bash
cd /app
env | sort > /logs/verifier/env.txt
{
  for p in /app/sub/conftest.py /app/link /app/pkg/__pycache__/m.pyc /tmp/planted.py /root/.cache/pip/http/kept; do
    if [ -e "$p" ] || [ -L "$p" ]; then echo "present $p"; else echo "gone $p"; fi
  done
  echo "owner $(stat -c %U /app/mine.txt)"
} > /logs/verifier/files.txt
python3 -m pytest tests/test_repo.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

# A solution that edits the repository's conftest.py so that the test passes
# without the fix.
EDITED_CONFTEST = (
    "printf '%s\\n' 'import pathlib' 'import pytest' '' '' '@pytest.fixture' "
    "'def app():' '    pathlib.Path(\"/app/fixed.txt\").write_text(\"fixed\")' "
    "'    return \"repo\"' > /app/tests/conftest.py\n"
)


def _task(tmp_path: Path, *, solve: str, test: str, name: str) -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n'
        # Strict judges PATH and, with a folder left out, needs the contract.
        "required_tools = []\n\n"
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Write /app/fixed.txt.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + solve)
    (task / "tests" / "test.sh").write_text(test)
    return task


async def _oracle(task: Path, **isolation):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=task.parent / "jobs",
            job_name=f"{task.name}-oracle",
            rollout_name=f"{task.name}__oracle",
            **isolation,
        )
    ).run()


def _trial_file(task: Path, name: str) -> Path:
    """The trial's own copy (a separate verifier keeps one under
    verifier-sandbox/ too)."""
    found = [
        p
        for p in (task.parent / "jobs").rglob(name)
        if "verifier-sandbox" not in p.parts
    ]
    (path,) = found
    return path


def _inputs(task: Path) -> dict:
    return json.loads(_trial_file(task, "integrity/verifier-inputs.json").read_text())


@pytest.fixture(scope="module")
def trials(tmp_path_factory):
    """One trial per mode with the recording test.sh, the agent's leftovers
    in place."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    import asyncio

    out = {}
    for mode in ("audit", "strict"):
        task = _task(
            tmp_path_factory.mktemp(mode),
            solve=SOLVE + LEFT_BEHIND,
            test=RECORDING_TEST,
            name=f"faithful-{mode}",
        )
        kwargs = {} if mode == "audit" else {"integrity": "strict"}
        result = asyncio.run(_oracle(task, **kwargs))
        env = dict(
            line.split("=", 1)
            for line in _trial_file(task, "verifier/env.txt").read_text().splitlines()
            if "=" in line
        )
        files = _trial_file(task, "verifier/files.txt").read_text()
        out[mode] = (result, env, files, task)
    return out


# Each removed step: what test.sh sees by default, and that strict still
# does it (the control, so the check can fail).
ENV_STEPS = [
    # (step, variable, default value or None for unset, strict check)
    # Strict always passes -c, the image's own file or /dev/null, and
    # --rootdir (the lead's 13:57Z ruling).
    (
        "pytest options (-c, --rootdir, --confcutdir)",
        "PYTEST_ADDOPTS",
        None,
        lambda v: v.startswith("-c ") and " --rootdir=" in v and "--confcutdir=" in v,
    ),
    ("plugin autoload off", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", None, lambda v: v == "1"),
    ("bytecode pins", "PYTHONPYCACHEPREFIX", None, lambda v: bool(v)),
    ("git pins", "GIT_CONFIG_COUNT", None, lambda v: int(v) > 0),
    ("git global config", "GIT_CONFIG_GLOBAL", None, lambda v: bool(v)),
    ("pip config", "PIP_CONFIG_FILE", None, lambda v: v == "/dev/null"),
    ("loader pins", "LD_LIBRARY_PATH", "/opt/img-lib", lambda v: v == ""),
    # strict moves the verifier to a separate sandbox with no agent user, so
    # no folder there is the agent's to drop: no control.
    ("PATH judge", "PATH", "starts:/opt/anybin:", lambda v: True),
]


@pytest.mark.deterministic
@pytest.mark.parametrize(
    ("step", "name", "default", "strict"), ENV_STEPS, ids=[s[0] for s in ENV_STEPS]
)
def test_default_mode_leaves_test_shs_environment_as_the_image_set_it(
    trials, step, name, default, strict
):
    _, audit_env, _, _ = trials["audit"]
    _, strict_env, _, _ = trials["strict"]
    if default is None:
        assert name not in audit_env, (step, audit_env.get(name))
    elif default.startswith("starts:"):
        assert audit_env[name].startswith(default[len("starts:") :]), (
            step,
            audit_env[name],
        )
    else:
        assert audit_env[name] == default, (step, audit_env.get(name))
    assert strict(strict_env.get(name, "")), (step, strict_env.get(name))


FILE_STEPS = [
    # (step, path, gone in strict too: the control)
    ("conftest sweep", "/app/sub/conftest.py", True),
    ("workspace symlink purge", "/app/link", False),
    ("__pycache__ purge", "/app/pkg/__pycache__/m.pyc", True),
    ("/tmp .py purge", "/tmp/planted.py", True),
]


@pytest.mark.deterministic
@pytest.mark.parametrize(
    ("step", "path", "strict_gone"), FILE_STEPS, ids=[s[0] for s in FILE_STEPS]
)
def test_default_mode_leaves_the_agents_files_where_they_are(
    trials, step, path, strict_gone
):
    assert f"present {path}" in trials["audit"][2], (step, trials["audit"][2])
    if strict_gone:
        assert f"gone {path}" in trials["strict"][2], (step, trials["strict"][2])


@pytest.mark.deterministic
def test_default_mode_leaves_owners_and_caches(trials):
    files = trials["audit"][2]
    assert "owner agent" in files  # no chown of the workspace to root
    assert "present /root/.cache/pip/http/kept" in files  # no cache reclaim
    assert "owner root" in trials["strict"][2]


@pytest.mark.deterministic
def test_both_modes_score_the_oracle_and_record_the_detector(trials):
    for mode in ("audit", "strict"):
        result, _, _, task = trials[mode]
        assert result.error is None and result.verifier_error is None, mode
        assert result.rewards == {"reward": 1.0}, (mode, result.rewards)
        inputs = _inputs(task)
        assert inputs["mode"] == mode
        assert inputs["status"] == "flagged"  # the planted conftest.py
        flagged = {(f["path"], f["kind"]) for f in inputs["flags"]}
        assert ("/app/sub/conftest.py", "planted") in flagged
        assert inputs["reward_changed"] is False


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
async def test_a_flag_never_changes_the_reward(tmp_path):
    """The edited conftest is flagged with its path, and the trial scores
    what test.sh gives: 1.0 by default. Strict removes the edit (the image
    manifest's keep rule) and scores 0.0, the flag recorded the same."""
    plain = """\
#!/bin/bash
cd /app
python3 -m pytest tests/test_repo.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""
    for mode, reward in (("audit", 1.0), ("strict", 0.0)):
        task = _task(
            tmp_path / mode, solve=EDITED_CONFTEST, test=plain, name=f"edited-{mode}"
        )
        kwargs = {} if mode == "audit" else {"integrity": "strict"}
        result = await _oracle(task, **kwargs)
        assert result.error is None and result.verifier_error is None, (
            mode,
            result.error,
        )
        assert result.rewards == {"reward": reward}, (mode, result.rewards)
        inputs = _inputs(task)
        assert [(f["path"], f["kind"], f["category"]) for f in inputs["flags"]] == [
            ("/app/tests/conftest.py", "edited", "conftest")
        ], (mode, inputs["flags"])
        assert inputs["reward_changed"] is False


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
async def test_off_runs_no_detector_and_says_so(tmp_path):
    task = _task(tmp_path, solve=SOLVE, test=RECORDING_TEST, name="faithful-off")

    result = await _oracle(task, integrity="off")

    assert result.rewards == {"reward": 1.0}, result.rewards
    inputs = _inputs(task)
    assert inputs["status"] == "off" and inputs["flags"] == []


# CIR's honest probe (11:00Z): the image's pytest.ini sets ``pythonpath = .``;
# with -c /dev/null strict dropped it and the honest answer scored 0. Strict
# passes BenchFlow's copy of it in /app (the lead's 13:57Z ruling).
PYTHONPATH_INI = DOCKERFILE.replace(
    "ENV LD_LIBRARY_PATH",
    "RUN printf '[pytest]\\npythonpath = .\\n' > /app/pytest.ini"
    " && printf 'def answer():\\n    return 42\\n' > /app/mod.py\n"
    "ENV LD_LIBRARY_PATH",
)
IMPORTS_MOD = """\
#!/bin/bash
cd /app
printf 'import mod\\n\\n\\ndef test_mod():\\n    assert mod.answer() == 42\\n' > /app/tests/test_mod.py
python3 -m pytest tests/test_mod.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
@pytest.mark.parametrize("mode", ["audit", "strict"])
async def test_the_images_pytest_ini_is_read_in_every_mode(tmp_path, mode):
    task = _task(tmp_path, solve=SOLVE, test=IMPORTS_MOD, name=f"ini-{mode}")
    (task / "environment" / "Dockerfile").write_text(PYTHONPATH_INI)
    kwargs = {} if mode == "audit" else {"integrity": "strict"}

    result = await _oracle(task, **kwargs)

    assert result.error is None and result.verifier_error is None, result.error
    assert result.rewards == {"reward": 1.0}, (mode, result.rewards)
    stdout = _trial_file(task, "verifier/test-stdout.txt").read_text()
    from benchflow.sandbox.lockdown import _PYTEST_CONFIG_DIR

    configfile = (
        f"configfile: ../{_PYTEST_CONFIG_DIR.lstrip('/')}/"
        if mode == "strict"
        else "configfile: pytest.ini"
    )
    assert configfile in stdout, stdout[-2000:]


# -- S2: the sandbox user's processes live through grading in off and audit ----


def test_only_strict_stops_turn_processes_at_the_turns_end():
    """Before another connection they stop in every mode; at the end of the
    turn only in strict (the lead's 11:25Z ruling)."""
    from types import SimpleNamespace

    from benchflow.integrity.constants import hardens_verifier
    from benchflow.rollout import Rollout

    def stops(integrity, reconnecting=False):
        rollout = SimpleNamespace(
            _config=SimpleNamespace(integrity=integrity), _reconnecting=reconnecting
        )
        return Rollout._stops_turn_processes(rollout)

    assert not stops(None) and not stops("audit") and not stops("off")
    assert stops("strict")
    assert stops("audit", reconnecting=True)
    assert hardens_verifier("strict") and not hardens_verifier(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", [None, "strict"])
async def test_the_kill_runs_before_grading_only_in_strict(integrity):
    from benchflow.sandbox.lockdown import harden_before_verify
    from tests.test_sandbox_hardening import _make_env, _make_task

    env = _make_env()
    await harden_before_verify(
        env, _make_task(), "agent", workspace="/app", integrity=integrity
    )

    kills = [c.args[0] for c in env.exec.call_args_list if "each KILL" in c.args[0]]
    assert bool(kills) is (integrity == "strict")


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", [None, "strict"])
async def test_off_and_audit_stop_the_sandbox_users_processes_after_grading(
    integrity, tmp_path
):
    """The kill audit deferred runs once the verifier is done."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from benchflow.sdk import SDK
    from tests.test_sandbox_hardening import _make_env, _make_task

    env = _make_env()
    task = _make_task()
    task.config.verifier.service = "main"
    paths = MagicMock()
    paths.verifier_dir = tmp_path / "verifier"
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
    order: list[str] = []
    verifier.verify.side_effect = lambda *a, **k: (
        order.append("verify") or MagicMock(rewards={"reward": 1.0})
    )
    with patch("benchflow.task.Verifier", return_value=verifier):
        await SDK()._verify(
            env,
            task,
            paths,
            {},
            sandbox_user="agent",
            workspace="/app",
            integrity=integrity,
        )

    cmds = [c.args[0] for c in env.exec.call_args_list]
    kills = [i for i, c in enumerate(cmds) if "each KILL" in c]
    assert kills, cmds
    clear = next(i for i, c in enumerate(cmds) if "find /logs/verifier" in c)
    if integrity == "strict":
        assert kills[0] < clear  # before grading
    else:
        assert kills[0] > clear  # after it


# A service the solution starts as the agent's user, which test.sh uses.
SERVICE = (
    "CMD='nohup python3 -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 &'\n"
    'runuser -u agent -- sh -c "$CMD" 2>/dev/null || sh -c "$CMD"\n'
    "sleep 2\n"
)
USES_SERVICE = """\
#!/bin/bash
cd /app
if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765', timeout=5)"; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
@pytest.mark.parametrize(("mode", "reward"), [("audit", 1.0), ("strict", 0.0)])
async def test_a_service_the_solution_started_serves_test_sh_by_default(
    tmp_path, mode, reward
):
    """FAITHFUL's uvicorn/mlflow/daemon class: by default the service is still
    there when test.sh runs (as under other harnesses); strict grades in a
    separate sandbox, where it never ran."""
    task = _task(tmp_path, solve=SERVICE, test=USES_SERVICE, name=f"service-{mode}")
    kwargs = {} if mode == "audit" else {"integrity": "strict"}

    result = await _oracle(task, **kwargs)

    assert result.error is None and result.verifier_error is None, result.error
    assert result.rewards == {"reward": reward}, (mode, result.rewards)


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", ["audit", "strict"])
async def test_soft_verify_changes_the_sandbox_only_in_strict(tmp_path, integrity):
    """Between rounds as at the end: in audit nothing soft verify does may
    change what the scored verify later sees (no hook purge, no /app, no
    BASH_ENV policy, no pins)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from benchflow.sandbox.user import PassthroughUser
    from tests.test_user import _make_user_trial

    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path / "trial")
    trial._config.integrity = integrity
    trial._config.sandbox_user = None
    planes = MagicMock()
    for name in (
        "clear_verifier_output_dir",
        "ensure_legacy_app_dir",
        "cleanup_verifier_python_hooks",
    ):
        setattr(planes, name, AsyncMock())
    planes.soft_verifier_bash_env = AsyncMock(return_value="/etc/judged.bashrc")
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0}))
    planes.verifier = MagicMock(return_value=verifier)
    trial._planes = planes
    with patch("benchflow.rollout._publish_trajectory_for_verifier", AsyncMock()):
        _rewards, _output, error = await trial.soft_verify()

    assert error is None, error
    hardened = integrity == "strict"
    assert planes.cleanup_verifier_python_hooks.await_count == int(hardened)
    assert planes.ensure_legacy_app_dir.await_count == int(hardened)
    assert verifier.bash_env == ("/etc/judged.bashrc" if hardened else "")
    assert planes.clear_verifier_output_dir.await_count == 1  # in every mode
