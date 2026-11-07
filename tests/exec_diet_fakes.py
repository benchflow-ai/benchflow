"""Test doubles for the exec-merging tests (``tests/test_exec_diet_*.py``, PY8 U12).

``BoxSandbox`` runs each exec in a Docker container the test owns, the way
``DockerSandbox.exec`` does: ``sh -c``, stdout and stderr in one stream, the
environment through the env file, the budget kept on the host. It records
every exec (command, user, budget, and the BenchFlow function that asked), so
a test can count them, and it runs on the base and on the branch alike, which
is what the equivalence goldens need.

``Faults`` makes chosen execs fail: an exec whose command contains a given
text raises or answers a given status instead of running.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from benchflow.sandbox._base import (
    BaseSandbox,
    ExecResult,
    with_system_path,
    wrap_command_with_env_file,
)
from benchflow.sandbox.lockdown import _keep_route_path
from tests._receipt_fakes import inner_command, receipt_answer

UBUNTU = (
    "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)
PYTHON = (
    "python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
)
LABEL = "benchflow.test=exec-diet"
UPDATE_GOLDEN_ENV = "BENCHFLOW_UPDATE_GOLDEN"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "exec_diet"
# The marker every merged exec carries (sandbox/_exec_merge.py); absent on the
# base, so a fault keyed on it does nothing there.
MERGED_MARK = "@@benchflow-merged-"


def docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def caller() -> str:
    """The BenchFlow function that asked for an exec: the innermost frame in
    the package outside ``sandbox/_base.py``, ``sandbox/_exec_merge.py``,
    ``sandbox/_exec_receipt.py`` and ``sandbox/_exec_output.py`` (``exec_raw``,
    which harness readers call through and which only marks the read raw)."""
    import benchflow

    package = str(Path(benchflow.__file__).parent) + os.sep
    skip = (
        os.path.join(package, "sandbox", "_base.py"),
        os.path.join(package, "sandbox", "_exec_merge.py"),
        os.path.join(package, "sandbox", "_exec_receipt.py"),
        os.path.join(package, "sandbox", "_exec_output.py"),
    )
    frames = [
        f
        for f in traceback.extract_stack()
        if f.filename.startswith(package) and f.filename not in skip
    ]
    return frames[-1].name if frames else "?"


@dataclass
class Exec:
    command: str
    user: str | int | None
    timeout_sec: int | None
    cwd: str | None
    caller: str
    return_code: int | None = None


@dataclass
class Faults:
    """``raise_on``: text -> exception raised by an exec whose command holds
    the text; ``status_on``: text -> (status, stdout) answered instead."""

    raise_on: dict[str, BaseException] = field(default_factory=dict)
    status_on: dict[str, tuple[int, str]] = field(default_factory=dict)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=300, check=check
    )


@pytest.fixture
def box_factory() -> Iterator[Any]:
    """Make containers (``box_factory(image, *run_args)``), removed after the
    test."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    made: list[str] = []

    def make(image: str, *run_args: str) -> str:
        name = f"exec-diet-{uuid.uuid4().hex[:12]}"
        # Listed first: a run that fails after creating the container still
        # leaves it behind.
        made.append(name)
        _run(
            "run",
            "-d",
            "--name",
            name,
            "--label",
            LABEL,
            "--network",
            "none",
            *run_args,
            image,
            "sleep",
            "infinity",
        )
        return name

    yield make
    for name in made:
        _run("rm", "-f", name, check=False)


def as_root(container: str, script: str) -> str:
    """Set up a container (not through a sandbox, so nothing is counted)."""
    return _run("exec", "-u", "0", container, "sh", "-c", script).stdout


class BoxSandbox(BaseSandbox):
    def __init__(
        self,
        container: str,
        *,
        persistent_env: dict[str, str] | None = None,
        faults: Faults | None = None,
    ) -> None:
        # Not BaseSandbox.__init__: it validates a task environment folder.
        self.container = container
        self.default_user = None
        self._persistent_env = persistent_env or {}
        self.faults = faults or Faults()
        self.execs: list[Exec] = []
        self.rollout_paths = None
        self.is_mounted_flag = False

    def _validate_definition(self) -> None:
        return None

    @classmethod
    def preflight(cls) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        return None

    async def stop(self, delete: bool) -> None:
        return None

    async def upload_file(
        self, source_path, target_path, *, mode=None, **_kwargs
    ) -> None:
        data = Path(source_path).read_bytes()
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-u",
            "0",
            self.container,
            "sh",
            "-c",
            'mkdir -p "$(dirname "$1")" && cat > "$1"',
            "sh",
            target_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await proc.communicate(data)

    async def upload_dir(
        self, source_dir, target_dir, service="main", **_kwargs
    ) -> None:
        for source in Path(source_dir).rglob("*"):
            if source.is_file():
                await self.upload_file(
                    source, f"{target_dir}/{source.relative_to(source_dir)}"
                )

    async def download_file(self, source_path, target_path) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        _run("cp", f"{self.container}:{source_path}", str(target_path))

    async def download_dir(self, source_dir, target_dir, service="main") -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        _run("cp", f"{self.container}:{source_dir}/.", str(target_dir))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        raw_output: bool = False,
    ) -> ExecResult:
        record = Exec(command, user, timeout_sec, cwd, caller())
        self.execs.append(record)
        for text, error in self.faults.raise_on.items():
            if text in command:
                raise error
        for text, (status, stdout) in self.faults.status_on.items():
            if text in command:
                record.return_code = status
                return ExecResult(stdout=stdout, return_code=status)
        user = self._resolve_user(user)
        merged_env = self._merge_env(env)
        if merged_env:
            command = wrap_command_with_env_file(
                merged_env, command, env_path_prefix="/tmp/.exec_diet_env_"
            )
        args = ["exec"]
        if cwd:
            args += ["-w", cwd]
        if user is not None:
            args += ["-u", str(user)]
        args += [self.container, "sh", "-c", command]
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            if timeout_sec:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout_sec)
            else:
                out, _ = await proc.communicate()
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds"
            ) from None
        record.return_code = proc.returncode or 0
        return ExecResult(
            stdout=out.decode(errors="replace") if out else None,
            return_code=proc.returncode or 0,
        )


_STEP = re.compile(
    r"'(@@benchflow-merged-[0-9a-f]+):(\d+):begin'\n(?:\(|__bf_out\d+=\$\()\n"
    r"(.*?)\n\)(?: 2>&1)?\n__bf_rc=\$\?\n(?:printf '%s' \"\$__bf_out\d+\"\n)?"
    r"printf [^\n]*(?:\n|$)(\[ \"\$__bf_rc\" -eq 0 \] \|\| exit)?",
    re.DOTALL,
)


def merged_steps(script: str) -> tuple[str, list[tuple[str, bool]]]:
    """A merged exec's tag and its steps: (command, gate) each."""
    steps = [(m.group(3), m.group(4) is not None) for m in _STEP.finditer(script)]
    match = _STEP.search(script)
    assert match is not None, script[:200]
    return match.group(1), steps


def merged_answer(tag: str, answers: list[tuple[int, str]]) -> str:
    """The stdout a merged exec prints when its steps answer *answers*."""
    return "".join(
        f"{tag}:{i}:begin\n{out}\n{tag}:{i}:end:{rc}\n"
        for i, (rc, out) in enumerate(answers)
    )


class ScriptedSandbox(BoxSandbox):
    """No container: ``handler(command, user)`` answers each command with
    (status, output) or raises. A merged exec is one recorded call whose
    steps are answered in order (a failed gate step stops it), so a test
    sees exactly the execs BenchFlow made. ``garble`` makes every merged
    exec answer unreadably."""

    def __init__(
        self,
        handler: Any,
        *,
        persistent_env: dict[str, str] | None = None,
        garble: bool = False,
    ) -> None:
        super().__init__("scripted", persistent_env=persistent_env)
        self.handler = handler
        self.garble = garble

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        raw_output: bool = False,
    ) -> ExecResult:
        # A read lockdown decides on comes as a receipt script: record and
        # answer the command it wraps, then print that answer's receipt.
        wrapped, command = (
            command,
            inner_command(command)
            .removeprefix(_keep_route_path(""))
            .removeprefix(with_system_path("")),
        )
        self.execs.append(Exec(command, user, timeout_sec, cwd, caller()))
        return receipt_answer(wrapped, self._answer(command, user))

    def _answer(self, command: str, user: str | int | None) -> ExecResult:
        if MERGED_MARK not in command:
            status, output = self.handler(command, user)
            return ExecResult(stdout=output, return_code=status)
        if self.garble:
            return ExecResult(stdout="no markers here", return_code=0)
        tag, steps = merged_steps(command)
        answers = []
        for step, gate in steps:
            status, output = self.handler(
                step.removeprefix(_keep_route_path("")).removeprefix(
                    with_system_path("")
                ),
                user,
            )
            answers.append((status, output))
            if gate and status != 0:
                break
        return ExecResult(stdout=merged_answer(tag, answers), return_code=0)


def check_golden(name: str, value: Any) -> None:
    """Compare *value* with ``fixtures/exec_diet/<name>.json``; with
    ``BENCHFLOW_UPDATE_GOLDEN=1`` write it instead (done once, on the base)."""
    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(value, indent=1, sort_keys=True) + "\n"
    if os.environ.get(UPDATE_GOLDEN_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no golden {path.name}; make it on the base"
    assert json.loads(text) == json.loads(path.read_text()), name
