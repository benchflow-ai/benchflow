"""(g2) R: the host re-judges a plugin guard stop on the pristine image
(the lead, 2026-11-16 08:20Z).

test.sh's user can forge or delete the trial's record, so nothing of the
trial chooses what is judged. In a fresh container from the trial's image,
the guard judges what a pytest there would load, with the host's own facts.
If it rejects or fails there too, the trial is unscored; if it trusts all of
it, the trial scores 0. The pristine container here is a fake that answers as
the guard's ``--judge`` mode would. The guard that writes the trial's record
is the real one, in this process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import (
    PLUGIN_GUARD_REJECTED,
    PLUGIN_GUARD_UNSCORED,
    VERIFIER_FAILED,
    classify_result,
    classify_verifier_error,
)
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _guard_rejudge
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._guard_reach import script_reach
from benchflow.sandbox._guard_rejudge import (
    BOOTSTRAP,
    Judged,
    Nomination,
    decide,
    nominate,
    parse_judgment,
)
from benchflow.sandbox._image_manifest import ImageManifest
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import pytest_plugin_guard_judge_source
from benchflow.task import RolloutPaths, Verifier
from tests import test_pytest_plugin_guard_manifest as manifest_tests
from tests import test_verifier_guard_load_failure as load_failure
from tests.test_guard_verdicts import _InternalError, _refuse, armed_here, repo

__all__ = ["armed_here", "repo"]  # fixtures

IMAGE = "sha256:" + "ab" * 32
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def _fresh_cache():
    _guard_rejudge._CACHE.clear()


def test_the_host_names_what_is_judged(tmp_path):
    """The task's own tests folder (the host's copy) and the Pythons hardening
    gave a guard copy: never the trial's record."""
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["sphinx.testing.fixtures", "pkg.plugin"]\n'
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/conftest.py").write_text(
        'pytest_plugins = "pkg.plugin"\nOTHER = ["x"]\npytest_plugins += [name()]\n'
    )
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/conftest.py").write_text("pytest_plugins = [\n")
    found = nominate(tmp_path, ["/a/python", "/b/python", "/a/python", "python3"])
    assert found == Nomination(
        (), ("sphinx.testing.fixtures", "pkg.plugin"), ("/a/python", "/b/python")
    )
    assert nominate(None, []) == Nomination()
    # The task's own -p plugins (its [verifier] env), not the guard's, not no:.
    addopts = (
        "-p no:cacheprovider -p _runtime_guard_ab -p xdist -ptimeout --strict-markers"
    )
    assert nominate(None, [], addopts, "_runtime_guard_ab").modules == (
        "xdist",
        "timeout",
    )
    assert nominate(None, [], "-p 'unclosed", "g").modules == ()


def test_only_a_whole_answer_counts():
    asked = Nomination(("/testbed/a.py",), ("pkg.mod",), (PYTHON,))
    rows = [
        {"path": "/testbed/a.py", "kind": "trusted", "why": "", "python": "3.6.13"},
        {"module": "pkg.mod", "kind": "absent", "why": "", "python": "3.6.13"},
    ]
    judged = parse_judgment("noise\n" + json.dumps(rows), asked)
    assert judged is not None and [j.kind for j in judged] == ["trusted", "absent"]
    for bad in (
        json.dumps(rows[:1]),  # a nominated module left out
        json.dumps([{**rows[0], "kind": "fine"}, rows[1]]),
        json.dumps({"rows": rows}),
        "Traceback (most recent call last):",
        "",
    ):
        assert parse_judgment(bad, asked) is None, bad


def test_the_pristine_image_decides():
    rejected = Judged("/testbed/x/conftest.py", "rejected", "owner chain", "3.6.13")
    failed = Judged("pkg.mod", "guard_error", "TypeError: zip()", "3.6.13")
    trusted = Judged("/testbed/x.pyc", "absent")
    unscored = decide([trusted, rejected], "the trial's stop")
    assert unscored.outcome == "unscored"
    assert (
        "rejects /testbed/x/conftest.py too (owner chain, Python 3.6.13)"
        in unscored.reason
    )
    assert "fails while judging pkg.mod too" in decide([failed], "s").reason
    zero = decide([trusted], "the trial's stop")
    assert zero.outcome == "zero" and zero.reason.startswith("the trial's stop; ")


class _Pristine(_InternalError):
    """A sandbox that can start a pristine container: a fake answering as the
    guard's ``--judge`` mode would, per path or module."""

    _benchflow_guard_source = "# the guard hardening installed\n"
    _benchflow_guard_pythons = ("/usr/bin/python3",)
    answers: ClassVar[dict] = {}
    default = "trusted"
    exit_code = 0
    runs: ClassVar[list] = []

    async def pristine_image(self):
        return IMAGE

    async def run_pristine(self, image, python, script, stdin, timeout_sec):
        sent = json.loads(stdin)
        assert image == IMAGE and script == BOOTSTRAP
        assert sent["guard"] == pytest_plugin_guard_judge_source(
            self._benchflow_guard_source
        )
        request = sent["request"]
        type(self).runs.append((python, request))
        # The discover() set, and what the host named.
        asked = ["/usr/lib/python3/dist-packages/pkg/conftest.py", *request["modules"]]
        rows = [
            {
                ("module" if "/" not in what else "path"): what,
                "kind": type(self).answers.get(what, type(self).default),
                "why": "judged",
                "python": "3.6.13",
            }
            for what in asked
        ]
        return ExecResult(
            stdout=json.dumps(rows), stderr="", return_code=type(self).exit_code
        )


@pytest.fixture
def pristine(monkeypatch):
    monkeypatch.setattr(_Pristine, "answers", {})
    monkeypatch.setattr(_Pristine, "default", "trusted")
    monkeypatch.setattr(_Pristine, "exit_code", 0)
    monkeypatch.setattr(_Pristine, "runs", [])
    return _Pristine


async def _rollout(
    tmp_path,
    name,
    sandbox_class,
    conftest=None,
    integrity="strict",
    test_sh=None,
    addopts="",
):
    task = load_failure._task(
        tmp_path, {"PYTEST_ADDOPTS": f"-p no:cacheprovider {addopts}-p {name}"}
    )
    if conftest is not None:
        (task.paths.tests_dir / "conftest.py").write_text(conftest)
    if test_sh is not None:
        task.paths.test_path.write_text(test_sh)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = sandbox_class(
        paths,
        tmp_path / "test_outputs.py",
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
    return rewards, error


def _planted_cache(repo):
    _, pyc = manifest_tests._stale_cache(repo)
    manifest_tests._record(repo.manifest, sorted(repo.workspace.rglob("*.py")))
    guard._MANIFEST.clear()
    _refuse(["pkg.testing.fixtures"])
    return str(pyc)


def _forced_identical(repo, monkeypatch):
    real = guard._untrusted_reason
    fixtures = str(repo.workspace / "pkg/testing/fixtures.py")
    monkeypatch.setattr(
        guard,
        "_untrusted_reason",
        lambda path, blocked: "forced" if path == fixtures else real(path, blocked),
    )
    _refuse(["pkg.testing.fixtures"])
    return fixtures


async def test_the_agents_planted_cache_stays_a_0(repo, tmp_path, armed_here, pristine):
    _planted_cache(repo)
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
    ((python, request),) = pristine.runs
    assert python == "/usr/bin/python3" and request["paths"] == []


async def test_the_record_cannot_choose_what_is_judged(
    repo, tmp_path, armed_here, pristine
):
    """A record naming an image file the guard rejects (a hacked one, or a
    real one) is never put to the pristine guard: the 0 stands."""
    pyc = _planted_cache(repo)
    pristine.answers = {pyc: "rejected"}
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
    ((_, request),) = pristine.runs
    assert pyc not in json.dumps(request)


async def test_a_guard_that_rejects_the_pristine_image_too_is_unscored(
    repo, tmp_path, armed_here, pristine
):
    _planted_cache(repo)
    pristine.default = "rejected"
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    assert rewards is None and error.startswith(PLUGIN_GUARD_UNSCORED)
    assert "rejects /usr/lib/python3/dist-packages/pkg/conftest.py too (judged" in error


async def test_the_tests_folders_own_plugins_are_judged(tmp_path, armed_here, pristine):
    guard._mark("checking")
    pristine.answers = {"sphinx.testing.fixtures": "guard_error"}
    rewards, error = await _rollout(
        tmp_path,
        armed_here,
        pristine,
        conftest='pytest_plugins = ["sphinx.testing.fixtures"]\n',
    )
    ((_, request),) = pristine.runs
    assert request["modules"] == ["sphinx.testing.fixtures"]
    assert (
        rewards is None and "fails while judging sphinx.testing.fixtures too" in error
    )


@pytest.mark.parametrize(
    ("pristine_says", "unscored"), [("trusted", False), ("rejected", True)]
)
async def test_an_image_identical_refusal_is_what_the_pristine_image_says(
    repo, tmp_path, armed_here, pristine, monkeypatch, pristine_says, unscored
):
    """Forged or a real false refusal, the record reads the same: only the
    pristine image tells them apart."""
    _forced_identical(repo, monkeypatch)
    pristine.default = pristine_says
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    if unscored:
        assert rewards is None and error.startswith(PLUGIN_GUARD_UNSCORED)
    else:
        assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
        assert (
            "byte for byte the image's own" in error
            and "trusts what it judged" in error
        )


@pytest.mark.parametrize(
    ("pristine_says", "unscored"), [("trusted", False), ("guard_error", True)]
)
async def test_a_guard_error_is_what_the_pristine_image_says(
    tmp_path, armed_here, pristine, pristine_says, unscored
):
    """test.sh's user can sign a guard_error of its own (it reads the key)."""
    guard._mark(
        "verdicts",
        json.dumps(
            [
                {
                    "v": 1,
                    "guard": armed_here,
                    "kind": "guard_error",
                    "path": "/testbed/astropy/conftest.py",
                    "sha256": "",
                    "why": "TypeError: zip() takes no keyword arguments",
                    "verdict": "",
                    "python": "3.6.13",
                    "executable": PYTHON,
                }
            ]
        ),
    )
    guard._mark("crashed", "Traceback\nGuardError: guard_error: TypeError: zip()")
    pristine.default = pristine_says
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    if unscored:
        assert rewards is None and error.startswith(PLUGIN_GUARD_UNSCORED)
        assert (
            "fails while judging /usr/lib/python3/dist-packages/pkg/conftest.py too"
            in error
        )
    else:
        assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)


@pytest.mark.parametrize(
    ("pristine_says", "unscored"), [("trusted", False), ("rejected", True)]
)
async def test_a_stop_with_no_verdict_is_judged_the_same_way(
    tmp_path, armed_here, pristine, pristine_says, unscored
):
    """Deleted records: the guard stopped while checking and left no verdict."""
    guard._mark("checking")
    pristine.default = pristine_says
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    ((python, request),) = pristine.runs
    assert python == "/usr/bin/python3" and request["paths"] == request["modules"] == []
    assert (rewards is None) == unscored
    assert error.startswith(
        PLUGIN_GUARD_UNSCORED if unscored else PLUGIN_GUARD_REJECTED
    )


async def test_without_a_pristine_judgment_the_records_0_stands(
    repo, tmp_path, armed_here, pristine, monkeypatch, caplog
):
    _forced_identical(repo, monkeypatch)
    pristine.exit_code = 1  # the judgment could not run
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    assert rewards == {"reward": 0.0} and error is None
    assert "byte for byte the image's own" in caplog.text
    # A sandbox with no pristine container (Daytona, Modal): L alone.
    rewards, error = await _rollout(tmp_path / "l", armed_here, _InternalError)
    assert rewards == {"reward": 0.0} and error is None


async def test_the_host_manifest_reaches_the_pristine_guard(
    tmp_path, armed_here, pristine
):
    """R334 F3: the judgment reads the host's kept image manifest, never
    none, and a judgment under one manifest is not reused under another
    (N1)."""
    guard._mark("checking")
    taken = []
    for run, digest in (("a", "1" * 64), ("b", "2" * 64)):
        manifest = ImageManifest(
            "2026-11-16T15:00:00Z", {"/testbed/conftest.py": {"sha256": digest}}
        )
        pristine._benchflow_image_manifest = manifest
        taken.append(manifest.to_json())
        await _rollout(tmp_path / run, armed_here, pristine)
    del pristine._benchflow_image_manifest
    assert [request["manifest"] for _, request in pristine.runs] == taken
    await _rollout(tmp_path / "none", armed_here, pristine)
    assert pristine.runs[-1][1]["manifest"] is None


def test_only_what_the_trials_pytest_loads_is_judged():
    """R334 F1: an entry point counts when the trial's armed guard loads it
    by name (asked for, by a conftest pytest collects, not turned off)."""
    rows = [
        {"module": "pytest_timeout", "kind": "rejected", "from": "entry_point:timeout"},
        {"path": "/s/pt.dist-info", "kind": "rejected", "from": "entry_point:timeout"},
        {"module": "myplug", "kind": "rejected", "from": "entry_point:myplug"},
        {"module": "pkg.plug", "kind": "trusted", "from": "conftest:/testbed/c.py"},
        {"module": "pkg.plug", "kind": "rejected", "from": "entry_point:other"},
        {"module": "asked", "kind": "trusted", "from": "request"},
    ]
    stdout = json.dumps([{**r, "why": "", "python": "3.12.0"} for r in rows])
    nomination = Nomination((), ("asked",), (PYTHON,))

    def judged(loads, disabled=()):
        found = parse_judgment(
            stdout,
            Nomination(
                (), ("asked",), (PYTHON,), disabled=disabled, anchors=("/testbed/t",)
            ),
            loads,
        )
        assert found is not None
        return sorted({(j.what, j.kind) for j in found})

    every = parse_judgment(stdout, nomination)
    assert every is not None and len(every) == len(rows)
    # Nothing asks for timeout or myplug: only the conftest's module counts,
    # by its own row and by the entry point whose module it is.
    assert judged(()) == [
        ("asked", "trusted"),
        ("pkg.plug", "rejected"),
        ("pkg.plug", "trusted"),
    ]
    assert ("pytest_timeout", "rejected") in judged({"timeout"})
    assert ("/s/pt.dist-info", "rejected") in judged({"pytest_timeout"})
    assert ("pytest_timeout", "rejected") not in judged({"timeout"}, ("timeout",))
    assert _guard_rejudge.armed_loads(
        "_BENCHFLOW_REQUESTED = []  # type: list[str]\n"
        "_BENCHFLOW_OPTIONAL = []  # type: list[str]\n"
        "_BENCHFLOW_REQUESTED = ['ctrf']\n_BENCHFLOW_OPTIONAL = ['timeout']\n"
    ) == {"ctrf", "timeout"}
    assert nominate(
        None, (PYTHON,), "-p no:myplug -p kept", "", ("timeout", "myplug"), ("x",)
    ) == Nomination((), ("kept", "timeout"), (PYTHON,), disabled=("myplug", "x"))


async def test_the_pristine_judgment_is_cached(
    repo, tmp_path, armed_here, pristine, monkeypatch
):
    _forced_identical(repo, monkeypatch)
    for run in ("a", "b"):
        await _rollout(tmp_path / run, armed_here, pristine)
    assert len(pristine.runs) == 1


async def test_trials_of_one_task_share_a_judgment():
    """Each trial's guard has a name of its own; nothing else differs."""
    from benchflow.sandbox.lockdown import _pytest_plugin_guard_source

    runs = []

    class Env:
        async def pristine_image(self):
            return IMAGE

        async def run_pristine(self, image, python, script, stdin, timeout_sec):
            runs.append(python)
            return ExecResult(stdout="[]", stderr="", return_code=0)

    nomination = Nomination((), (), ("/usr/bin/python3",))
    for name in ("_runtime_guard_" + "1" * 32, "_runtime_guard_" + "2" * 32):
        env = Env()
        env._benchflow_guard_source = _pytest_plugin_guard_source(
            name, ("/testbed",), []
        )
        assert (
            await _guard_rejudge.rejudge(env, name, nomination, "s")
        ).outcome == "zero"
    assert runs == ["/usr/bin/python3"]


@pytest.mark.parametrize("integrity", ["off", "audit"])
async def test_only_a_strict_trial_is_rejudged(
    repo, tmp_path, armed_here, pristine, monkeypatch, integrity
):
    """R is --integrity strict's alone (the lead, 09:58Z): elsewhere no
    pristine container starts and the record's 0 stands."""
    _forced_identical(repo, monkeypatch)
    pristine.default = "rejected"
    rewards, error = await _rollout(tmp_path, armed_here, pristine, integrity=integrity)
    assert rewards == {"reward": 0.0} and error is None and pristine.runs == []


async def test_a_plugin_the_verifier_installed_is_not_rejudged(
    tmp_path, armed_here, pristine
):
    guard._mark("refused", "Verifier plugin trust rejected: x")
    guard._mark("installed", "pytest-x 1.0\n/venv/x.py\tnot the image's")
    rewards, error = await _rollout(tmp_path, armed_here, pristine)
    assert rewards == {"reward": 0.0} and error is None and pristine.runs == []


def test_the_verdicts_are_deterministic():
    unscored = f"{PLUGIN_GUARD_UNSCORED}on the pristine image the pytest plugin guard rejects /x too"
    assert classify_verifier_error(unscored) == VERIFIER_FAILED
    assert (
        classify_result(reward=None, error=None, verifier_error=unscored)
        == "verifier_errored"
    )


def test_the_bootstrap_runs_the_guard_as_a_judge(tmp_path):
    """BOOTSTRAP, in a plain Python: the guard's source runs as __main__ with
    --judge, the request as its stdin."""
    source = (
        "import json, sys\n"
        "if __name__ == '__main__' and sys.argv[1:] == ['--judge']:\n"
        "    request = json.load(sys.stdin)\n"
        "    print(json.dumps([{'path': p, 'kind': 'trusted', 'why': sys.argv[0],"
        " 'python': ''} for p in request['paths']]))\n"
    )
    sent = _guard_rejudge.payload(
        "_runtime_guard_x", source, Nomination(("/a.py",), (), (PYTHON,)), "{}"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", BOOTSTRAP],
        input=sent,
        capture_output=True,
        check=True,
    ).stdout.decode()
    ((row),) = json.loads(out)
    assert row["path"] == "/a.py" and row["why"].endswith("/_runtime_guard_x.py")


async def test_docker_starts_a_fresh_network_less_root_container():
    sandbox = object.__new__(DockerSandbox)
    sandbox._recovery_baseline = None
    seen = {}

    async def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return ExecResult(stdout="[]", stderr="", return_code=0)

    sandbox._run_docker_process = run
    sandbox._docker_client_env = lambda: {}
    await sandbox.run_pristine(IMAGE, "/usr/bin/python3.6", BOOTSTRAP, b"{}", 180)
    command = seen["command"]
    run_args = command[command.index("run") + 1 :]
    assert run_args[: run_args.index("--entrypoint")] == [
        "--rm",
        "-i",
        "--pull",
        "never",
        "--network",
        "none",
        "--user",
        "0:0",
        "--label",
        "benchflow.pristine-guard-judgment=1",
    ]
    assert run_args[run_args.index("--entrypoint") :] == [
        "--entrypoint",
        "/usr/bin/python3.6",
        IMAGE,
        "-c",
        BOOTSTRAP,
    ]
    assert seen["stdin_data"] == b"{}" and seen["timeout_sec"] == 180
    assert "-v" not in run_args and "--mount" not in run_args


@pytest.mark.parametrize(
    ("baseline", "inspected", "image"),
    [
        (None, IMAGE + "\n", IMAGE),
        (object(), IMAGE, None),
        (None, "busybox:latest", None),
    ],
    ids=["started-image", "recovery-baseline", "not-an-id"],
)
async def test_the_pristine_image_is_mains_own_by_id(baseline, inspected, image):
    sandbox = object.__new__(DockerSandbox)
    sandbox._recovery_baseline = baseline

    async def main_id():
        return "c0ffee"

    async def cli(args, check=True, **kwargs):
        assert args == ["inspect", "--format", "{{.Image}}", "c0ffee"]
        return ExecResult(stdout=inspected, stderr="", return_code=0)

    sandbox._main_container_id = main_id
    sandbox._docker_cli = cli
    assert await sandbox.pristine_image() == image


def test_a_conftest_counts_only_where_pytest_collects_it():
    """R334's delta: pytest loads an image conftest.py in a folder that holds
    a path it is given, or one below a folder it collects into (not one its
    default norecursedirs skips), never one above --confcutdir."""
    collected = _guard_rejudge.collected
    tests = ("/app/tests",)
    assert collected("/app/conftest.py", tests)
    assert collected("/app/tests/conftest.py", tests)
    assert collected("/app/tests/unit/conftest.py", tests)
    assert not collected("/app/extra/conftest.py", tests)
    assert not collected("/app/tests/build/conftest.py", tests)
    assert not collected("/app/tests/.hidden/conftest.py", tests)
    assert collected("/app/tests/conftest.py", ("/app/tests/test_x.py",))
    assert not collected("/app/tests/sub/conftest.py", ("/app/tests/test_x.py",))
    assert collected("/conftest.py", ("/tests/test_outputs.py",))
    assert not collected("/conftest.py", ("/tests/test_outputs.py",), "/tests")
    assert collected("/tests/conftest.py", ("/tests/test_outputs.py",), "/tests")
    assert not collected("/app/conftest.py", ())


def test_only_what_the_trials_pytest_reads_counts():
    """R334's delta (latD, latE): an image conftest pytest never collects and
    an image ini strict's -c never gives pytest count for nothing, rows and
    names alike; the -c ini's -p names come in the nomination, and count."""
    rows = [
        {
            "module": "bad_d",
            "kind": "rejected",
            "from": "conftest:/app/extra/conftest.py",
        },
        {"module": "bad_d", "kind": "rejected", "from": "entry_point:bad_d"},
        {"module": "bad_e", "kind": "rejected", "from": "ini:/app/sub/tox.ini"},
        {"module": "bad_e", "kind": "rejected", "from": "entry_point:bad_e"},
        {
            "path": "/app/sub/setup.cfg",
            "kind": "guard_error",
            "from": "ini:/app/sub/setup.cfg",
        },
        {"module": "ok", "kind": "trusted", "from": "conftest:/app/conftest.py"},
        {"module": "ok", "kind": "trusted", "from": "entry_point:ok"},
        {"module": "from_c", "kind": "rejected", "from": "request"},
    ]
    stdout = json.dumps([{**r, "why": "", "python": "3.12.0"} for r in rows])
    nomination = nominate(
        None,
        (PYTHON,),
        "",
        "",
        ("from_c",),
        anchors=("/app/tests",),
        confcutdir="/tests",
    )
    found = parse_judgment(stdout, nomination, set())
    assert found is not None
    assert sorted((j.what, j.kind) for j in found) == [
        ("from_c", "rejected"),
        ("ok", "trusted"),
        ("ok", "trusted"),
    ]
    assert _guard_rejudge.decide(found, "stop").outcome == "unscored"
    nomination = nominate(None, (PYTHON,), "", "", (), anchors=("/app/tests",))
    rows = [r for r in rows if r["from"] != "request"]
    stdout = json.dumps([{**r, "why": "", "python": "3.12.0"} for r in rows])
    found = parse_judgment(stdout, nomination, set())
    assert found is not None and _guard_rejudge.decide(found, "stop").outcome == "zero"


@pytest.mark.parametrize(
    ("text", "paths", "bare"),
    [
        ("#!/bin/bash\ncd /app\npython -m pytest tests -rA\n", ("/app/tests",), ()),
        ("pytest /tests/test_outputs.py -rA\n", ("/tests/test_outputs.py",), ()),
        ("pytest $TEST_DIR/test_outputs.py\n", ("/tests/test_outputs.py",), ()),
        (
            "uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest "
            "--ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA\n",
            ("/tests/test_outputs.py",),
            (),
        ),
        ("cd /app && pytest -k smoke\n", (), ("/app",)),
        ("pytest tests\n", (), ()),
        ("pytest\n", (), ("",)),
        (
            "cd /testbed\npytest -p img_plugin tests/a.py::test_x ../b\n",
            ("/testbed/tests/a.py", "/b"),
            (),
        ),
        ("cd $REPO && pytest tests\n", (), ()),
    ],
    ids=[
        "cd-relative",
        "absolute",
        "test-dir",
        "uvx",
        "bare-cd",
        "unknown-cwd",
        "bare",
        "node-id",
        "variable-cd",
    ],
)
def test_the_host_reads_where_test_sh_points_pytest(text, paths, bare):
    reach = script_reach(text)
    assert reach.paths == paths and reach.bare == bare


def test_pytest_collects_from_test_shs_paths_or_the_c_inis_testpaths():
    from benchflow.task.verifier_core import _pytest_collects_from

    reach = script_reach("cd /app\npytest\npytest /tests/test_x.py\n")
    ini = SimpleNamespace(plugins=("pytester",), testpaths=("/app/testing",))
    addopts = "-p no:cacheprovider -c /x.ini --confcutdir=/tests --rootdir=/app"
    assert _pytest_collects_from(reach, ini, addopts) == (
        ("/tests/test_x.py", "/app/testing"),
        "/tests",
    )
    assert _pytest_collects_from(reach, None, addopts) == (
        ("/tests/test_x.py", "/app"),
        "/tests",
    )
    assert _pytest_collects_from(None, ini, addopts) == ((), "")


def test_the_verifiers_own_folder_starts_the_reading():
    """A verify run in a folder the verifier names reads test.sh's relative
    paths from there."""
    assert script_reach("pytest tests\n", cwd="/app").paths == ("/app/tests",)
    assert script_reach("pytest\n", cwd="/app").bare == ("/app",)
    assert script_reach("pytest tests\n", cwd="app").paths == ()


def test_the_nomination_is_what_the_trials_pytest_loads():
    """R334's MS3-MS5: the pristine judge gets test.sh's ``-p`` names and the
    ``-c`` ini's, less what test.sh turns off with ``-p no:``."""
    from benchflow.task.verifier_core import _guard_nomination

    reach = script_reach("cd /app\npytest -p tsplug -p no:offplug -c /ci.ini tests\n")
    sandbox = SimpleNamespace(
        _benchflow_guard_pythons=("/usr/bin/python3",),
        _benchflow_pytest_ini=SimpleNamespace(
            plugins=("iniplug", "offplug"), testpaths=()
        ),
    )
    task = SimpleNamespace(paths=SimpleNamespace(tests_dir=None), config=None)
    nomination = _guard_nomination(task, sandbox, "_runtime_guard_ab", reach, "")
    assert "tsplug" in nomination.modules, nomination
    assert "iniplug" in nomination.modules, nomination
    assert "offplug" not in nomination.modules, nomination
    assert "offplug" in nomination.disabled, nomination
    assert nomination.pythons == ("/usr/bin/python3",), nomination
