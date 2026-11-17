"""(g2) R in real containers: the pristine re-judgment runs the guard's
``--judge`` mode in a fresh, network-less root container from a real image,
through ``DockerSandbox.run_pristine`` (the lead, 2026-11-16 08:20Z).

What a trial's record says cannot make it unscored. A forged guard_error, a
forged refusal of an image file and deleted markers all score 0 when the
guard trusts the pristine image. Only a guard that rejects or fails on the
pristine image itself ends the trial unscored. The trial's markers are
written by the real guard in this process; the judgment is the real guard's,
in the image's own Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any, cast

import pytest

from benchflow._utils.scoring import PLUGIN_GUARD_REJECTED, PLUGIN_GUARD_UNSCORED
from benchflow.sandbox import _guard_rejudge, lockdown
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from tests.test_guard_rejudge import _rollout
from tests.test_guard_verdicts import _InternalError, _refuse, armed_here, repo

__all__ = ["armed_here", "repo"]  # fixtures


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=20, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _docker_ready(), reason="Docker daemon required")

PYTHON = "/usr/local/bin/python3"
# A task image with a pytest11 plugin. The "loose" one leaves that plugin
# writable by anyone, which the guard rightly refuses on the pristine image
# too. Every image also has a module no pytest loads, writable by anyone: one
# a forged record could name. "py36" is a Python 3.6 task image, as
# astropy-7166's.
SITE = "/usr/local/lib/python3.12/site-packages"
DOCKERFILE = """\
ARG BASE=python:3.12-slim
FROM ${BASE}
ARG PYTEST="pytest==8.3.3 pytest-timeout==2.3.1"
RUN pip install --no-cache-dir $PYTEST
RUN P=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \\
 && echo "VALUE = 1" > $P/unloaded_helper.py && chmod o+w $P/unloaded_helper.py
ARG LOOSE=
RUN if [ -n "$LOOSE" ]; then \\
      P=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \\
      chmod o+w $P/pytest_timeout.py; fi
"""


def _build(tag: str, *args: str) -> str:
    subprocess.run(
        ["docker", "build", "-q", "-t", tag, *args, "-"],
        input=DOCKERFILE.encode(),
        check=True,
        capture_output=True,
        timeout=600,
    )
    return subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def images():
    # A tag per xdist worker: one worker's teardown untags only its own, and
    # an image the others still tag stays.
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    tags = tuple(
        f"benchflow-test/guard-judge:en708-{kind}-{worker}"
        for kind in ("clean", "loose", "py36")
    )
    built = {
        "clean": _build(tags[0]),
        "loose": _build(tags[1], "--build-arg", "LOOSE=1"),
        "py36": _build(
            tags[2],
            "--build-arg",
            "BASE=python:3.6-slim",
            "--build-arg",
            "PYTEST=pytest==6.2.5 pytest-timeout==1.4.2",
        ),
    }
    yield built
    subprocess.run(["docker", "rmi", "-f", *tags], capture_output=True, check=False)


@pytest.fixture(autouse=True)
def _fresh_cache():
    _guard_rejudge._CACHE.clear()


def _pristine(image: str, source: str):
    """A trial sandbox whose pristine container is a real one, by ID."""

    class Pristine(_InternalError):
        _benchflow_guard_source = source
        _benchflow_guard_pythons = (PYTHON,)

        async def pristine_image(self):
            return image

        async def run_pristine(self, image, python, script, stdin, timeout_sec):
            docker = object.__new__(DockerSandbox)
            docker._recovery_baseline = None
            return await DockerSandbox.run_pristine(
                docker, image, python, script, stdin, timeout_sec
            )

    return Pristine


def _source(name: str, extra: str = "", optional=("timeout",)) -> str:
    """The guard hardening would install for *name*, as the host keeps it.
    Hardening's discovery makes every plugin it trusts in the image's Python
    optional: pytest-timeout on the clean and 3.6 images, none on the loose
    one, whose plugin it rejects."""
    return (
        lockdown._pytest_plugin_guard_source(
            name,
            ("/testbed",),
            [],
            ownership=True,
            agent_gids=(1000,),
            optional=list(optional),
        )
        + extra
    )


def _forge_guard_error(name: str) -> None:
    """test.sh's user signs a guard_error of its own: it can read the key."""
    guard._mark(
        "verdicts",
        json.dumps(
            [
                {
                    "v": 1,
                    "guard": name,
                    "kind": "guard_error",
                    "path": "/usr/local/lib/python3.12/site-packages/pytest_timeout.py",
                    "sha256": "",
                    "why": "TypeError: zip() takes no keyword arguments",
                    "verdict": "",
                    "python": "3.6.13",
                    "executable": "/usr/bin/python3.6",
                }
            ]
        ),
    )
    guard._mark("crashed", "Traceback\nGuardError: guard_error: TypeError: zip()")


async def test_the_pristine_container_is_fresh_root_and_offline(images, monkeypatch):
    """The bootstrap's container: root, no network, nothing of the host."""
    probe = (
        "import json, os, socket, sys\n"
        "if __name__ == '__main__' and sys.argv[1:] == ['--judge']:\n"
        "    request = json.load(sys.stdin)\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "        net = 'online'\n"
        "    except OSError:\n"
        "        net = 'offline'\n"
        "    print(json.dumps([{'module': m, 'kind': 'trusted', 'python': '',"
        " 'why': f'{os.getuid()} {net} {os.path.exists(\"/rollout\")}'}"
        " for m in request['modules']]))\n"
    )
    # The probe alone, without the guard's judge after it.
    monkeypatch.setattr(_guard_rejudge, "pytest_plugin_guard_judge_source", lambda s: s)
    nomination = _guard_rejudge.Nomination((), ("pkg.plugin",), (PYTHON,))
    sandbox = object.__new__(_pristine(images["clean"], probe))
    rejudged = await _guard_rejudge.rejudge(
        sandbox, "_runtime_guard_x", nomination, "s"
    )
    assert rejudged is not None and rejudged.outcome == "zero"
    (judged,) = rejudged.judged
    assert judged.why == "0 offline False"


@pytest.mark.parametrize(
    "forge",
    ["guard_error", "image_identical", "deleted_markers"],
)
async def test_what_the_record_says_cannot_unscore_a_trial(
    repo, tmp_path, armed_here, monkeypatch, images, forge
):
    if forge == "guard_error":
        _forge_guard_error(armed_here)
    elif forge == "image_identical":
        real = guard._untrusted_reason
        fixtures = str(repo.workspace / "pkg/testing/fixtures.py")
        monkeypatch.setattr(
            guard,
            "_untrusted_reason",
            lambda path, blocked: "forged" if path == fixtures else real(path, blocked),
        )
        _refuse(["pkg.testing.fixtures"])
    else:
        guard._mark("checking")  # the guard's verdicts deleted, a stop left
    sandbox = _pristine(images["clean"], _source(armed_here))
    rewards, error = await _rollout(tmp_path, armed_here, cast(Any, sandbox))
    assert rewards == {"reward": 0.0}
    assert error.startswith(PLUGIN_GUARD_REJECTED) and "trusts what it judged" in error


async def test_a_record_naming_a_rejectable_image_file_still_scores_0(
    tmp_path, armed_here, images
):
    """A forged record names an image file the guard would reject (writable by
    anyone) that no pytest of the task loads: it is never put to the pristine
    guard, so the agent's 0 stands."""
    helper = f"{SITE}/unloaded_helper.py"
    guard._mark(
        "verdicts",
        json.dumps(
            [
                {
                    "v": 1,
                    "guard": armed_here,
                    "kind": "rejected",
                    "path": helper,
                    "sha256": "",
                    "why": "writable by anyone",
                    "verdict": "no-manifest",
                    "python": "3.12.0",
                    "executable": PYTHON,
                }
            ]
        ),
    )
    guard._mark(
        "refused", f"Verifier plugin trust rejected: unloaded_helper ({helper})"
    )
    sandbox = _pristine(images["clean"], _source(armed_here))
    rewards, error = await _rollout(tmp_path, armed_here, cast(Any, sandbox))
    assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
    assert f"rejected {helper} (in a tree the image manifest leaves out" in error


@pytest.mark.parametrize(
    ("image", "extra", "optional", "addopts", "said"),
    [
        # The guard's own code failing on the pristine image (astropy-7166),
        # on the plugin hardening found there.
        (
            "clean",
            "\ndef _untrusted_reason(path, blocked):\n"
            "    raise TypeError('zip() takes no keyword arguments')\n",
            ("timeout",),
            "",
            "fails while judging",
        ),
        # The image's own plugin, writable by anyone, that the task loads by
        # name: refused there too.
        ("loose", "", (), "-p timeout ", "rejects"),
    ],
    ids=["guard-error", "image-refused"],
)
async def test_a_guard_the_pristine_image_defeats_too_is_unscored(
    tmp_path, armed_here, images, image, extra, optional, addopts, said
):
    _forge_guard_error(armed_here)
    sandbox = _pristine(images[image], _source(armed_here, extra, optional))
    rewards, error = await _rollout(
        tmp_path, armed_here, cast(Any, sandbox), addopts=addopts
    )
    assert rewards is None and error.startswith(PLUGIN_GUARD_UNSCORED)
    assert f"on the pristine image the pytest plugin guard {said}" in error


class _Honest(_InternalError):
    """test.sh: pytest passed and test.sh wrote a 1. The guard's cleanup left
    its end-of-session record, as a real pytest's does ((g2))."""

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return await super().exec(command, env=env, **kwargs)
        guard._mark(
            "finished",
            json.dumps(
                {
                    "session": True,
                    "passed": 1,
                    "exitstatus": 0,
                    "collected": 1,
                    "failed": 0,
                }
            ),
        )
        guard._unmark("registered")
        self._paths.reward_text_path.write_text("1")
        return ExecResult(stdout="", stderr="", return_code=0)


@pytest.mark.parametrize(
    "addopts", ["", "-p no:timeout "], ids=["never-named", "turned-off"]
)
async def test_an_entry_point_the_trial_never_loads_cannot_unscore_it(
    tmp_path, armed_here, images, addopts
):
    """R334 F1 (latC): the loose image's pytest-timeout is an entry point the
    guard rejects on the pristine image, but with autoload off the trial's
    pytest never loads it: nothing asks for it, or the task turns it off. An
    honest trial scores 1.0; a failing one whose test.sh forges the guard's
    own error stays a 0, never unscored."""
    source = _source(armed_here, optional=())
    honest = _pristine(images["loose"], source)

    class HonestPristine(_Honest, honest):
        pass

    rewards, error = await _rollout(
        tmp_path / "honest", armed_here, cast(Any, HonestPristine), addopts=addopts
    )
    assert rewards == {"reward": 1.0} and error is None
    _forge_guard_error(armed_here)
    sandbox = _pristine(images["loose"], source)
    rewards, error = await _rollout(
        tmp_path / "forged", armed_here, cast(Any, sandbox), addopts=addopts
    )
    assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
    assert (
        "trusts what it judged" in error
        and "pytest_timeout" not in error.split("; on the pristine image")[1]
    )


# The guard as it was for astropy-7166: zip(strict=True), which Python 3.6
# refuses, in its byte-code comparison.
_FIXED = re.compile(r"zip\(inner, expected\)")
_AS_IT_WAS = "zip(inner, expected, strict=True)"


@pytest.mark.parametrize(("as_it_was", "unscored"), [(True, True), (False, False)])
async def test_the_3_6_guard_error_is_judged_on_the_pristine_3_6_image(
    tmp_path, armed_here, images, as_it_was, unscored
):
    """astropy-7166: the guard's own error on a Python 3.6 image. The guard as
    it was fails there too, so the trial is unscored; the guard as shipped
    trusts the image, so the trial's record alone leaves a 0."""
    _forge_guard_error(armed_here)
    source = _source(armed_here)
    if as_it_was:
        if len(_FIXED.findall(source)) != 1:
            pytest.skip("this line's guard never compared byte code with zip()")
        source = _FIXED.sub(_AS_IT_WAS, source)
    sandbox = _pristine(images["py36"], source)
    rewards, error = await _rollout(tmp_path, armed_here, cast(Any, sandbox))
    if unscored:
        assert rewards is None and error.startswith(PLUGIN_GUARD_UNSCORED)
        assert "fails while judging" in error and "Python 3.6" in error
    else:
        assert rewards == {"reward": 0.0} and error.startswith(PLUGIN_GUARD_REJECTED)
