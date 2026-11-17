"""In ``--integrity strict`` a run whose pytest left no record of
the plugin guard is never test.sh's pass (the lead, 2026-11-16 13:57Z).

The evidence that a pytest the guard reaches ran is the host's copy of
test.sh (sandbox/_guard_reach.py), never the sandbox's outputs: CIR's
``si-exit`` puts ``addopts = -p bf_evil`` in the workspace's pytest.ini, and
``bf_evil`` calls ``os._exit(0)`` on import, before the guard loads and
before pytest prints a word. Such a run is a recorded 0 with the reason, or,
where the host re-judges it on the pristine image, a 0 when the guard gives a
verdict there and unscored when it gives none. A test.sh that starts no
pytest, or one the guard does not reach on the task's own terms (tox, a reset
environment, a replaced PYTEST_ADDOPTS), scores as before.

The verifier runs a real pytest subprocess with the environment it passes to
test.sh and the real armed guard; only the container transport is stubbed
(tests/test_verifier_guard_load_failure.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import PLUGIN_GUARD_REJECTED, PLUGIN_GUARD_UNSCORED
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _guard_rejudge
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._guard_reach import PytestReach, script_reach, task_reach
from benchflow.sandbox.lockdown import (
    pytest_plugin_guard_key,
    pytest_plugin_guard_key_file,
    pytest_plugin_guard_markers_dir,
)
from benchflow.task import RolloutPaths, Verifier
from tests import test_verifier_guard_load_failure as load_failure
from tests.test_guard_verdicts import PREFIX

PASSING = load_failure.PASSING
FAILING = load_failure.FAILING
SI_EXIT = "#!/bin/bash\nmkdir -p /logs/verifier\ncd /app\npython -m pytest tests -rA\n"
EVIL = "import os\nos._exit(0)\n"
EVIL_INI = "[pytest]\naddopts = -p bf_evil\n"


@pytest.fixture(autouse=True)
def _guard_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        load_failure.lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path)
    )
    _guard_rejudge._CACHE.clear()


# -- what the host's copy of test.sh says -------------------------------------


@pytest.mark.parametrize(
    ("text", "expects"),
    [
        (SI_EXIT, True),
        ("pytest tests -rA > /logs/verifier/out.txt 2>&1\n", True),
        ("timeout -k 5 600 python3 -u -m pytest -q 2>&1 | tee /logs/out.txt\n", True),
        (
            "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n"
            "source $HOME/.local/bin/env\n"
            "uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest "
            "--ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA\n",
            True,
        ),
        ("uv run --with pytest -- pytest -q\n", True),
        ("poetry run pytest\n", True),
        ("coverage run -m pytest\n", True),
        ('out=$(pytest -q 2>&1)\necho "$out"\n', True),
        ('bash -c "cd /app && pytest"\n', True),
        ("xvfb-run -a pytest\n", True),
        ("$PYTHON -m pytest\n", True),
        ("if ! pytest -x; then echo failed; fi\n", True),
        (
            'python - <<EOF\nimport pytest\nraise SystemExit(pytest.main(["-q"]))\nEOF\n',
            True,
        ),
        ('export PYTEST_ADDOPTS="$PYTEST_ADDOPTS -x"\npytest\n', True),
        (
            "#!/usr/bin/env python3\nimport subprocess\nsubprocess.run(['pytest'])\n",
            True,
        ),
        # No pytest at all, or only words that name it.
        ("node test.js\n", False),
        ('echo "run pytest now"\npip install pytest\nwhich pytest\n', False),
        ("command -v pytest\n# pytest -q\n", False),
        ("python -m unittest discover\n", False),
        ('cat > /tmp/x <<"EOF"\npytest -q\nEOF\n', False),
        ("make test\n", False),
        # pytest the guard does not reach on the task's own terms.
        ("tox --current-env -epy39 -v -- tests/test_x.py\n", False),
        ("python -m tox -e py\n", False),
        ("nox -s tests\n", False),
        ("sudo -u agent pytest -q\n", False),
        ('su - agent -c "cd /app; python -m pytest"\n', False),
        ("env -i PATH=/usr/bin pytest\n", False),
        ("docker exec app pytest\n", False),
        ("ssh box pytest\n", False),
        ('PYTEST_ADDOPTS="-x" pytest\n', False),
        ("unset PYTEST_ADDOPTS\npytest\n", False),
        ("env -u PYTEST_ADDOPTS pytest\n", False),
        ("pytest -q\ntox -e py\n", False),
    ],
)
def test_the_host_reads_whether_test_sh_starts_a_pytest_the_guard_reaches(
    text, expects
):
    assert script_reach(text).expects_record is expects


def test_scripts_of_the_verifier_folder_are_followed(tmp_path):
    tests = tmp_path / "tests"
    (tests / "lib").mkdir(parents=True)
    (tests / "run.sh").write_text("cd /app\npytest -p myplug -p no:cacheprovider\n")
    (tests / "lib" / "run_tests.py").write_text("import pytest\npytest.main([])\n")
    (tests / "shell.sh").write_text("node test.js\n")
    (tmp_path / "outside.sh").write_text("pytest\n")
    for text in (
        "bash /tests/run.sh\n",
        "bash $TEST_DIR/run.sh\n",
        "source ${TEST_DIR}/run.sh\n",
        "cd /tests && ./run.sh\n",
        "python3 /tests/lib/run_tests.py\n",
    ):
        assert script_reach(text, tests).expects_record, text
    reach = script_reach("bash /tests/run.sh\n", tests)
    assert reach.plugins == ("myplug",) and reach.disabled == ("cacheprovider",)
    for text in ("bash /tests/shell.sh\n", "bash /tests/../outside.sh\n"):
        assert not script_reach(text, tests).expects_record, text
    # Nothing outside the verifier folder is read.
    assert not script_reach("bash /app/run.sh\n", tests).direct


def test_an_unreadable_test_sh_says_nothing(tmp_path):
    task = SimpleNamespace(
        paths=SimpleNamespace(tests_dir=tmp_path, test_path=tmp_path / "missing.sh")
    )
    assert task_reach(task) == PytestReach()
    assert task_reach(SimpleNamespace()) == PytestReach()
    assert task_reach(task, "python -m pytest -q").expects_record


# -- the verifier, with a real pytest and the real guard ----------------------


class _NoPytest(load_failure.PytestSandbox):
    """test.sh checks without pytest and writes a 1."""

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return await super().exec(command, env=env, **kwargs)
        self._paths.reward_text_path.write_text("1")
        return ExecResult(stdout="", stderr="", return_code=0)


class _Pristine(load_failure.PytestSandbox):
    """A sandbox that can start a pristine container: a fake judge that
    trusts what it is asked, or exits with ``exit_code``."""

    _benchflow_guard_source = "# the guard hardening installed\n"
    _benchflow_guard_pythons = ("/usr/bin/python3",)
    exit_code: ClassVar[int] = 0
    runs: ClassVar[list] = []

    async def pristine_image(self):
        return "sha256:" + "ab" * 32

    async def run_pristine(self, image, python, script, stdin, timeout_sec):
        request = json.loads(stdin)["request"]
        type(self).runs.append(request)
        rows = [
            {"module": m, "kind": "trusted", "why": "", "python": "3.12.0"}
            for m in request["modules"]
        ]
        return ExecResult(
            stdout=json.dumps(rows) if type(self).exit_code == 0 else "",
            stderr="Traceback\nSyntaxError: invalid syntax",
            return_code=type(self).exit_code,
        )


async def _run(
    tmp_path,
    test_sh,
    test_source,
    *,
    sandbox_class=load_failure.PytestSandbox,
    files=None,
    integrity="strict",
):
    """A strict verify of a task whose host test.sh is *test_sh*; the sandbox
    runs ``python -m pytest -q <the workspace's test>`` (unless it is
    :class:`_NoPytest`) with the verifier's env and the real armed guard."""
    guard = PREFIX + uuid.uuid4().hex
    guard_dir = tmp_path / guard
    guard_dir.mkdir()
    Path(pytest_plugin_guard_markers_dir(guard)).mkdir(mode=0o700)
    Path(pytest_plugin_guard_key_file(guard)).write_text(pytest_plugin_guard_key(guard))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_outputs.py").write_text(test_source)
    for name, text in (files or {}).items():
        (workspace / name).write_text(text)
    task = load_failure._task(
        tmp_path,
        {
            "PYTEST_ADDOPTS": f"-p no:cacheprovider -p {guard}",
            "PYTHONPATH": str(guard_dir),
        },
    )
    task.paths.test_path.write_text(test_sh)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    (guard_dir / f"{guard}.py").write_text(
        load_failure.armed()(guard, paths.verifier_dir, workspace)
    )
    sandbox = sandbox_class(
        paths,
        workspace / "test_outputs.py",
        isolated=False,
        pythonpath=None,
        banner="",
        log=None,
        plugins=(),
    )
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(
        cast(Any, sandbox), task, paths, {}, planes, integrity=integrity
    )
    return guard, paths, rewards, error


def _markers(guard: str) -> list[str]:
    return sorted(os.listdir(pytest_plugin_guard_markers_dir(guard)))


@pytest.mark.asyncio
async def test_a_pytest_that_left_no_record_is_not_test_shs_pass(tmp_path, caplog):
    """CIR's si-exit: the agent's ini loads a module that exits 0 before the
    guard loads. pytest ran, the guard recorded nothing, test.sh wrote 1."""
    guard, paths, rewards, error = await _run(
        tmp_path,
        SI_EXIT,
        FAILING,
        files={"pytest.ini": EVIL_INI, "bf_evil.py": EVIL},
    )
    assert paths.reward_text_path.read_text() == "1"  # test.sh's forged pass
    assert _markers(guard) == []
    assert rewards == {"reward": 0.0} and error is None
    assert (
        "The guard's record: the guard left no record, though test.sh starts "
        "pytest (`python -m pytest tests -rA`)" in caplog.text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "rewards", "starts", "said"),
    [
        (0, {"reward": 0.0}, PLUGIN_GUARD_REJECTED, "trusts what it judged"),
        (1, None, PLUGIN_GUARD_UNSCORED, "gives no verdict either"),
        (125, {"reward": 0.0}, None, ""),
    ],
    ids=["pristine-verdict", "no-verdict-there-either", "docker-failed"],
)
async def test_the_pristine_image_decides_a_run_with_no_record(
    tmp_path, monkeypatch, exit_code, rewards, starts, said, caplog
):
    """A verdict on the pristine image: the trial's own state kept the guard
    out, a 0. No verdict there either (its Python ran it and failed): the
    image's doing, unscored. Docker's own failure: the record's 0 stands."""
    monkeypatch.setattr(_Pristine, "exit_code", exit_code)
    monkeypatch.setattr(_Pristine, "runs", [])
    _, _, got, error = await _run(
        tmp_path,
        SI_EXIT,
        FAILING,
        sandbox_class=_Pristine,
        files={"pytest.ini": EVIL_INI, "bf_evil.py": EVIL},
    )
    assert got == rewards
    if starts is None:
        assert error is None and "the guard left no record" in caplog.text
    else:
        assert error.startswith(starts) and said in error
        assert "the guard left no record, though test.sh starts pytest" in error
    assert len(_Pristine.runs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_sh",
    [
        "#!/bin/bash\ncd /app\n"
        "[ \"$(python -c 'import calc; print(calc.add(2, 3))')\" = 5 ] "
        "&& echo 1 > /logs/verifier/reward.txt\n",
        "#!/bin/bash\ncd /app\ntox -e py\n",
        '#!/bin/bash\ncd /app\nPYTEST_ADDOPTS="-x" python -m pytest tests\n',
        "#!/bin/bash\nsudo -u agent python -m pytest tests\n",
    ],
    ids=["no-pytest", "tox", "own-addopts", "sudo"],
)
async def test_a_run_the_guard_does_not_reach_scores_as_before(tmp_path, test_sh):
    """Nothing reached the guard and test.sh starts no pytest it reaches:
    test.sh's reward, as written, in strict too."""
    guard, _, rewards, error = await _run(
        tmp_path, test_sh, FAILING, sandbox_class=_NoPytest
    )
    assert _markers(guard) == []
    assert rewards == {"reward": 1.0} and error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["pass", "fail"]
)
async def test_an_honest_pytest_run_is_scored(tmp_path, source, reward):
    """The guard reached pytest and registered: test.sh's reward stands."""
    _, _, rewards, error = await _run(tmp_path, SI_EXIT, source)
    assert rewards == {"reward": reward} and error is None


@pytest.mark.asyncio
async def test_the_rule_is_the_guards_alone(tmp_path):
    """With no guard in the verifier's env (off and audit arm none), nothing
    is read for it: test.sh's reward stands, as written."""
    task = load_failure._task(tmp_path, {})
    task.paths.test_path.write_text(SI_EXIT)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_outputs.py").write_text(FAILING)
    (workspace / "pytest.ini").write_text(EVIL_INI)
    (workspace / "bf_evil.py").write_text(EVIL)
    sandbox = load_failure.PytestSandbox(
        paths,
        workspace / "test_outputs.py",
        isolated=False,
        pythonpath=None,
        banner="",
        log=None,
        plugins=(),
    )
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(
        cast(Any, sandbox), task, paths, {}, planes, integrity="audit"
    )
    assert rewards == {"reward": 1.0} and error is None


def test_the_fixture_forges_a_pass_outside_benchflow(tmp_path):
    """si-exit's facts with plain pytest: it exits 0, prints nothing and runs
    no test, so test.sh's 1 is forged."""
    (tmp_path / "test_outputs.py").write_text(FAILING)
    (tmp_path / "pytest.ini").write_text(EVIL_INI)
    (tmp_path / "bf_evil.py").write_text(EVIL)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "PYTHON"))
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_outputs.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""


# -- the guard's end-of-session record (the lead's 15:41Z ruling) -------

PYTEST_SH = "#!/bin/bash\ncd /app\npython -m pytest -q test_outputs.py\n"
# A test module that ends pytest during collection, with exit status 0.
EXITS_IN_COLLECTION = "import os\nos._exit(0)\n" + FAILING
# A conftest that turns pytest's failed run into exit status 0.
STATUS_ZERO = "def pytest_sessionfinish(session):\n    session.exitstatus = 0\n"


def _records(paths, guard: str) -> list[dict]:
    """The end-of-session records the verifier copied to verifier/."""
    return [
        json.loads(p.read_text())
        for p in sorted(paths.verifier_dir.glob(f"{guard}.*.finished"))
    ]


@pytest.mark.asyncio
async def test_a_pass_stands_on_the_guards_end_of_session_record(tmp_path):
    """An honest pytest: its record (exit 0, one test collected and passed)
    is copied to verifier/, its registered marker is gone, and test.sh's 1
    stands."""
    guard, paths, rewards, error = await _run(tmp_path, PYTEST_SH, PASSING)
    assert rewards == {"reward": 1.0} and error is None
    assert not [m for m in _markers(guard) if m.endswith(".registered")]
    assert _records(paths, guard) == [
        {"session": True, "passed": 1, "exitstatus": 0, "collected": 1, "failed": 0}
    ]


@pytest.mark.asyncio
async def test_a_pytest_that_ends_during_collection_is_not_test_shs_pass(
    tmp_path, caplog
):
    """The guard registered, then a test module ended pytest with exit status
    0 before its session finished: no record, and test.sh's 1 is a 0."""
    guard, paths, rewards, error = await _run(tmp_path, PYTEST_SH, EXITS_IN_COLLECTION)
    assert paths.reward_text_path.read_text() == "1"
    assert [m.rpartition(".")[2] for m in _markers(guard)] == ["registered"]
    assert rewards == {"reward": 0.0} and error is None
    assert "ended before pytest's own cleanup and left no end-of-session record" in (
        caplog.text
    )


@pytest.mark.asyncio
async def test_a_record_that_disagrees_with_pytests_exit_status_is_not_a_pass(
    tmp_path, caplog
):
    """A conftest sets exit status 0 on a failed run: the record counts the
    failure beside the 0, which is not pytest's own status, so no pass."""
    guard, paths, rewards, error = await _run(
        tmp_path, PYTEST_SH, FAILING, files={"conftest.py": STATUS_ZERO}
    )
    assert paths.reward_text_path.read_text() == "1"
    assert _records(paths, guard) == [
        {"session": True, "passed": 0, "exitstatus": 0, "collected": 1, "failed": 1}
    ]
    assert rewards == {"reward": 0.0} and error is None
    assert "gives exit status 0 with 1 failed of 1 collected" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "rewards", "starts"),
    [(0, {"reward": 0.0}, PLUGIN_GUARD_REJECTED), (1, None, PLUGIN_GUARD_UNSCORED)],
    ids=["pristine-verdict", "no-verdict-there-either"],
)
async def test_the_pristine_image_decides_a_run_that_ended_early(
    tmp_path, monkeypatch, exit_code, rewards, starts
):
    """No whole record: a 0 where the guard gives a verdict on the pristine
    image, unscored where it gives none there either."""
    monkeypatch.setattr(_Pristine, "exit_code", exit_code)
    monkeypatch.setattr(_Pristine, "runs", [])
    _, _, got, error = await _run(
        tmp_path, PYTEST_SH, EXITS_IN_COLLECTION, sandbox_class=_Pristine
    )
    assert got == rewards
    assert error.startswith(starts)
    assert "ended before pytest's own cleanup" in error
    assert len(_Pristine.runs) == 1


@pytest.mark.asyncio
async def test_a_failing_run_is_not_judged_on_its_record(tmp_path):
    """test.sh's 0 is a 0 whatever the record: only a pass needs one."""
    _, _, rewards, error = await _run(tmp_path, PYTEST_SH, FAILING)
    assert rewards == {"reward": 0.0} and error is None


@pytest.mark.parametrize(
    ("args", "record"),
    [
        (["--version"], None),
        (
            ["-q", "test_outputs.py"],
            {
                "session": True,
                "passed": 1,
                "exitstatus": 0,
                "collected": 1,
                "failed": 0,
            },
        ),
        (
            ["-q", "-p", "xdist", "-n", "2", "test_outputs.py"],
            {
                "session": True,
                "passed": 1,
                "exitstatus": 0,
                "collected": 1,
                "failed": 0,
            },
        ),
    ],
    ids=["version", "session", "xdist"],
)
def test_every_pytest_the_guard_loads_in_leaves_its_record(tmp_path, args, record):
    """The real armed guard in a real pytest: each process (an xdist
    worker's included) writes its record at its cleanup and takes its
    registered marker away. ``--version`` answers before any plugin loads
    and leaves no marker at all. The venv here is not root's, so the armed
    copy skips the ownership rule for xdist to load."""
    if "xdist" in args:
        pytest.importorskip("xdist")
    guard = PREFIX + uuid.uuid4().hex
    guard_dir = tmp_path / guard
    guard_dir.mkdir()
    markers = Path(pytest_plugin_guard_markers_dir(guard))
    markers.mkdir(mode=0o700)
    Path(pytest_plugin_guard_key_file(guard)).write_text(pytest_plugin_guard_key(guard))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_outputs.py").write_text(PASSING)
    (guard_dir / f"{guard}.py").write_text(
        load_failure.armed("\n_BENCHFLOW_OWNERSHIP = False\n")(
            guard, tmp_path / "verifier", workspace
        )
    )
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "PYTHON"))
    }
    env.update(
        PYTEST_ADDOPTS=f"-p no:cacheprovider -p {guard}",
        PYTHONPATH=str(guard_dir),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    kinds = sorted(name.rpartition(".")[2] for name in os.listdir(markers))
    if record is None:
        assert kinds == []
        return
    processes = 3 if "xdist" in args else 1  # the controller and two workers
    assert kinds == ["finished"] * processes
    records = [
        json.loads(json.loads(p.read_text().split("\t", 1)[1]))
        for p in markers.iterdir()
    ]
    if "xdist" in args:
        print(records)
        assert {r["exitstatus"] for r in records} == {0}
        assert max(r["passed"] for r in records) == 1
    else:
        assert records == [record]


def test_a_pytest_a_test_runs_leaves_no_record_of_its_own(tmp_path):
    """A test that starts pytest and kills it: the inner pytest's end is
    that test's to judge, so it leaves no registered marker behind, and the
    outer pytest's record stands alone (its rule for nested pytests)."""
    guard = PREFIX + uuid.uuid4().hex
    guard_dir = tmp_path / guard
    guard_dir.mkdir()
    markers = Path(pytest_plugin_guard_markers_dir(guard))
    markers.mkdir(mode=0o700)
    Path(pytest_plugin_guard_key_file(guard)).write_text(pytest_plugin_guard_key(guard))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inner").mkdir()
    (workspace / "inner" / "test_inner.py").write_text("import os\nos._exit(0)\n")
    (workspace / "test_outer.py").write_text(
        "import subprocess, sys\n\n"
        "def test_runs_pytest():\n"
        "    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'inner'], check=True)\n"
    )
    (guard_dir / f"{guard}.py").write_text(
        load_failure.armed()(guard, tmp_path / "verifier", workspace)
    )
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "PYTHON"))
    }
    env.update(
        PYTEST_ADDOPTS=f"-p no:cacheprovider -p {guard}",
        PYTHONPATH=str(guard_dir),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_outer.py"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    kinds = sorted(name.rpartition(".")[2] for name in os.listdir(markers))
    assert kinds == ["finished"]
