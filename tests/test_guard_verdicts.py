"""The plugin guard's signed ``verdicts`` record decides a trial's 0,
whatever pytest's exit (the lead, 2026-11-16 06:57Z and 08:20Z).

A rejection of a file the image does not have as it is (``changed``, ``new``,
``no-manifest``) scores 0, with the file named in the trial's verifier error.
The guard's own error, a rejection of the image's own file, and a stop with no
verdict are a recorded 0 with the reason, until the host re-judges the files
on the pristine image. The record is test.sh's user's to forge, so it may only
ever lower that trial's own score. Paths and digests here are made up.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import (
    PLUGIN_GUARD_REJECTED,
    VERIFIER_FAILED,
    classify_result,
    classify_verifier_error,
)
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._guard_verdicts import (
    agents_rejection,
    read_verdicts,
    unconfirmed,
)
from benchflow.task import RolloutPaths, Verifier, verifier_core
from benchflow.task.verifier_errors import (
    PluginGuardLoadError,
    PluginGuardRejectedError,
)
from tests import test_pytest_plugin_guard_manifest as manifest_tests
from tests import test_verifier_guard_load_failure as load_failure

GUARD = "_runtime_guard_0123abcd"
CONFTEST = "/testbed/astropy/conftest.py"
DIGEST = hashlib.sha256(b"bytes the guard judged").hexdigest()

repo = manifest_tests.repo  # the image-manifest workspace fixture


def entry(kind="rejected", path=CONFTEST, verdict="changed", why="untrusted", **extra):
    return {
        "v": 1,
        "guard": GUARD,
        "kind": kind,
        "path": path,
        "sha256": DIGEST,
        "why": why,
        "verdict": verdict,
        "python": "3.6.13",
        **extra,
    }


def signed(kind, detail, token="12-0123abcd", key=None):
    """One line of the marker read, signed as the guard signs it (``_mark``)."""
    name = f"{GUARD}.{token}.{kind}"
    body = json.dumps(detail)
    key = (key or lockdown.pytest_plugin_guard_key(GUARD)).encode()
    mac = hmac.new(key, f"{name}\n{body}".encode(), hashlib.sha256).hexdigest()
    return f"{name}\t{mac}\t{body}"


def verdicts(*entries, token="12-0123abcd"):
    return signed("verdicts", json.dumps(list(entries)), token=token)


def markers(*lines):
    by_kind: dict[str, list] = {}
    for marker in lockdown.parse_pytest_plugin_guard_markers("\n".join(lines), GUARD):
        by_kind.setdefault(marker.kind, []).append(marker)
    return by_kind


def test_the_record_reads_and_a_bad_entry_is_a_problem():
    found, problem = read_verdicts(
        markers(verdicts(entry())).get("verdicts", []), GUARD
    )
    assert problem is None and [(e.kind, e.path, e.verdict) for e in found] == [
        ("rejected", CONFTEST, "changed")
    ]
    for bad in (
        entry(v=2),
        entry(guard="_runtime_guard_other"),
        entry(kind="accepted"),
        entry(path=""),  # a rejection names a file or a module
        entry(verdict="maybe"),
        entry(why=["not", "text"]),
        "not an object",
    ):
        found, problem = read_verdicts(
            markers(verdicts(entry(path="/testbed/ok.py"), bad)).get("verdicts", []),
            GUARD,
        )
        assert [e.path for e in found] == ["/testbed/ok.py"] and problem, bad
    found, problem = read_verdicts(
        markers(signed("verdicts", "{not json")).get("verdicts", []), GUARD
    )
    assert found == [] and "not a list" in problem


@pytest.mark.parametrize(
    ("verdict", "said"),
    [
        ("changed", "changed since the image"),
        ("new", "new since the image"),
        ("no-manifest", "in a tree the image manifest leaves out"),
    ],
)
def test_a_file_the_image_lacks_is_the_agents(verdict, said):
    record = [
        entry(path="/testbed/astropy", verdict="identical"),
        entry(verdict=verdict, why="planted"),
        entry(path="/testbed/b.pyc", verdict="new"),
    ]
    found, _ = read_verdicts(markers(verdicts(*record))["verdicts"], GUARD)
    reason = agents_rejection(found)
    assert reason is not None
    assert f"rejected {CONFTEST} ({said}: planted) and 1 more file" in reason


def test_the_images_own_file_or_the_guards_error_is_not_the_agents():
    identical = entry(verdict="identical", why="owner chain")
    error = entry(
        kind="guard_error",
        path="astropy.conftest",  # a module name when no file was reached
        verdict="",
        why="TypeError: zip() takes no keyword arguments",
    )
    for record in ([identical], [error], [identical, error]):
        found, problem = read_verdicts(markers(verdicts(*record))["verdicts"], GUARD)
        assert agents_rejection(found) is None and problem is None
    found, _ = read_verdicts(markers(verdicts(error))["verdicts"], GUARD)
    assert (
        "the guard failed while judging astropy.conftest (TypeError: zip() takes "
        "no keyword arguments, Python 3.6.13)" in unconfirmed(found, None)
    )
    found, _ = read_verdicts(markers(verdicts(identical))["verdicts"], GUARD)
    assert f"{CONFTEST}, byte for byte the image's own (owner chain)" in unconfirmed(
        found, None
    )
    assert unconfirmed([], None) == "no verdict of the guard names a file"


def _verifier(tmp_path, read):
    verifier = object.__new__(verifier_core.Verifier)
    verifier._rollout_paths = SimpleNamespace(verifier_dir=tmp_path)
    verifier._sandbox = SimpleNamespace()
    verifier._task = SimpleNamespace(paths=SimpleNamespace(task_dir=None))
    verifier._logger = logging.getLogger(__name__)

    async def nothing(*args, **kwargs):
        return None

    async def guard_markers(name, service):
        return markers(*read)

    verifier._bring_in_outputs = nothing
    verifier._check_launch_record = nothing
    verifier._plugin_guard_markers = guard_markers
    return verifier


async def _after(tmp_path, *read):
    """test.sh exited 3, pytest's INTERNALERROR, and wrote a 0."""
    await _verifier(tmp_path, read)._after_test_command(
        3,
        sandbox_paths=None,
        service="main",
        guard=GUARD,
        verifier_outputs_are_mounted=True,
        parse_rewards=False,
        aggregate_policy=None,
    )


REGISTERED = signed("registered", "", token="11-0123abcd")
REFUSED = signed("refused", "Verifier plugin trust rejected: astropy.conftest")


@pytest.mark.parametrize(
    "refusal",
    [[REFUSED], [], [signed("checking", "")]],
    ids=["refused", "no-refused-marker", "checking-left"],
)
async def test_an_internalerror_with_the_agents_file_is_a_0_that_names_it(
    tmp_path, refusal
):
    """INTERNALERROR can stop pytest before the guard's refused marker, and
    leave its checking marker: the verdicts record decides either way."""
    with pytest.raises(PluginGuardRejectedError) as raised:
        await _after(
            tmp_path,
            REGISTERED,
            *refusal,
            verdicts(
                entry(
                    path="/testbed/astropy/__pycache__/conftest.cpython-36.pyc",
                    verdict="new",
                )
            ),
        )
    assert "conftest.cpython-36.pyc (new since the image" in str(raised.value)


@pytest.mark.parametrize(
    ("read", "said"),
    [
        (
            [REFUSED, verdicts(entry(verdict="identical"))],
            "byte for byte the image's own",
        ),
        ([REFUSED], "no verdict of the guard names a file"),
        (
            [
                signed(
                    "crashed", "Traceback\nGuardError: guard_error: TypeError: zip()"
                ),
                verdicts(entry(kind="guard_error", verdict="", why="TypeError: zip()")),
            ],
            "crashed inside pytest (GuardError: guard_error: TypeError: zip())",
        ),
        (
            # A forged record (not the guard's key) is dropped: the refusal
            # stands with no verdict.
            [REFUSED, signed("verdicts", json.dumps([entry(verdict="new")]), key="00")],
            "no verdict of the guard names a file",
        ),
    ],
    ids=["image-identical", "no-verdict", "guard-error", "forged-record"],
)
async def test_what_is_not_the_agents_on_the_records_word_is_a_recorded_0(
    tmp_path, read, said
):
    with pytest.raises(PluginGuardLoadError) as raised:
        await _after(tmp_path, REGISTERED, *read)
    assert said in str(raised.value)


async def test_a_plugin_the_verifier_installed_stays_the_tasks_problem(tmp_path):
    """The verifier's own install after the agent stopped is new since the
    image too: the installed marker keeps it from being the agent's."""
    installed = signed("installed", "pytest-agent 1.0\n/venv/x.py\tnot the image's")
    with pytest.raises(PluginGuardLoadError) as raised:
        await _after(
            tmp_path, REGISTERED, REFUSED, installed, verdicts(entry(verdict="new"))
        )
    assert "was installed after the agent stopped" in str(raised.value)


async def test_a_clean_run_is_untouched(tmp_path):
    assert (
        await _verifier(tmp_path, [REGISTERED])._after_test_command(
            0,
            sandbox_paths=None,
            service="main",
            guard=GUARD,
            verifier_outputs_are_mounted=True,
            parse_rewards=False,
            aggregate_policy=None,
        )
    ).rewards is None


def test_a_rejection_is_a_deterministic_failure():
    error = f"{PLUGIN_GUARD_REJECTED}the pytest plugin guard rejected /testbed/x.pyc"
    assert classify_verifier_error(error) == VERIFIER_FAILED
    assert classify_result(reward=0.0, error=None, verifier_error=error) == "failed"


PREFIX = next(
    prefix
    for prefix in ("_runtime_guard_", "_benchflow_guard_")
    if lockdown.pytest_plugin_guard_name(f"-p {prefix}0123abcd")
)


class _InternalError(load_failure.PytestSandbox):
    """test.sh: pytest stopped with an INTERNALERROR (exit 3) and test.sh
    wrote a 0. The markers are what the real guard left."""

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return await super().exec(command, env=env, **kwargs)
        self._paths.reward_text_path.write_text("0")
        return ExecResult(stdout="", stderr="", return_code=3)


@pytest.fixture
def armed_here(tmp_path, monkeypatch):
    """The real guard, in this process, signing its markers where the verifier
    reads them back; returns the guard's name."""
    name = PREFIX + "0123abcd" * 4
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    folder = Path(lockdown.pytest_plugin_guard_markers_dir(name))
    folder.mkdir(parents=True, mode=0o700)
    key = Path(lockdown.pytest_plugin_guard_key_file(name))
    key.write_text(lockdown.pytest_plugin_guard_key(name))
    monkeypatch.setattr(guard, "_BENCHFLOW_MARKERS", str(folder / name))
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    # In the sandbox the guard is imported under its own name, which its
    # records carry.
    monkeypatch.setattr(guard, "__name__", name)
    guard._mark("registered")
    return name


async def _rollout(tmp_path, name):
    task = load_failure._task(
        tmp_path, {"PYTEST_ADDOPTS": f"-p no:cacheprovider -p {name}"}
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = _InternalError(
        paths,
        tmp_path / "test_outputs.py",
        isolated=False,
        pythonpath=None,
        banner="",
        log=None,
        plugins=(),
    )
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(sandbox, task, paths, {}, planes)
    return rewards, error


def _refuse(names):
    with pytest.raises(guard.Rejected) as raised:
        guard._validate(names, entry_points=False)
    guard._stopping(raised.value)  # as the guard's hook does


async def test_the_real_guards_record_of_a_planted_cache_is_a_named_0(
    repo, tmp_path, armed_here
):
    """The image's fixtures.py with a byte-code cache the image never had: the
    real guard refuses it, pytest stops with an INTERNALERROR, and the trial
    is a 0 whose verifier error names the cache."""
    _, pyc = manifest_tests._stale_cache(repo)
    manifest_tests._record(repo.manifest, sorted(repo.workspace.rglob("*.py")))
    guard._MANIFEST.clear()
    _refuse(["pkg.testing.fixtures"])
    rewards, error = await _rollout(tmp_path, armed_here)
    assert rewards == {"reward": 0.0}
    assert error.startswith(PLUGIN_GUARD_REJECTED)
    assert f"rejected {pyc} (new since the image" in error


async def test_the_real_guard_refusing_the_images_own_file_is_a_recorded_0(
    repo, tmp_path, armed_here, monkeypatch, caplog
):
    """A false refusal, forced, of a file byte for byte the image's: a 0 with
    the reason in the trial's warning, not a rejection of the agent's file."""
    real = guard._untrusted_reason
    fixtures = str(repo.workspace / "pkg/testing/fixtures.py")
    monkeypatch.setattr(
        guard,
        "_untrusted_reason",
        lambda path, blocked: "forced" if path == fixtures else real(path, blocked),
    )
    _refuse(["pkg.testing.fixtures"])
    rewards, error = await _rollout(tmp_path, armed_here)
    assert rewards == {"reward": 0.0} and error is None
    assert "The guard's record: the guard refused " in caplog.text
    assert (
        "byte for byte the image's own (Verifier plugin trust rejected: "
        "pkg.testing.fixtures (pkg.testing.fixtures: forced))" in caplog.text
    )
    assert f"refused {repo.workspace}/pkg/__init__.py and " in caplog.text


@pytest.mark.parametrize(
    ("extra", "exit_code"),
    [
        (
            "\ndef _untrusted_reason(path, blocked):\n"
            "    raise TypeError('zip() takes no keyword arguments')\n",
            1,
        ),
        (
            # Raised inside a session hook, as astropy-7166's was: pytest's
            # INTERNALERROR.
            "\ndef pytest_collection_modifyitems(items):\n"
            "    error = _guard_error(\n"
            "        TypeError('zip() takes no keyword arguments'), str(items[0].path)\n"
            "    )\n"
            "    _stopping(error)\n"
            "    raise error\n",
            3,
        ),
    ],
    ids=["loading-plugins", "internalerror"],
)
async def test_the_real_guards_own_error_is_a_recorded_0_with_its_reason(
    tmp_path, caplog, monkeypatch, extra, exit_code
):
    """The guard's own code raising while it judges a file (astropy-7166:
    ``zip(strict=True)`` under Python 3.6) stops a real pytest: a 0 whose
    trial warning carries the exception, never a rejection."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    exits = []

    class Recording(load_failure.PytestSandbox):
        async def exec(self, command, env=None, **kwargs):
            result = await super().exec(command, env=env, **kwargs)
            if "test-stdout.txt" in command:
                exits.append(result.return_code)
            return result

    monkeypatch.setattr(load_failure, "PytestSandbox", Recording)
    name, paths, rewards, error = await load_failure._run(
        tmp_path, load_failure.PASSING, guard_source=load_failure.armed(extra)
    )
    assert exits == [exit_code]
    assert rewards == {"reward": 0.0} and error is None
    assert "The guard's record: the guard failed while judging" in caplog.text
    assert "zip() takes no keyword arguments" in caplog.text
    # The 0's warning never repeats the error's "not scored" (FIX-49).
    assert "not scored" not in caplog.text
    assert list(paths.verifier_dir.glob(name + ".*.crashed"))
    (record,) = paths.verifier_dir.glob(name + ".*.verdicts")
    assert '"guard_error"' in record.read_text()
