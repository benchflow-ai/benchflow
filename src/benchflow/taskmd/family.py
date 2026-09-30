"""``family@1``: one instance of a task family, built by its generator from a seed.

A family is one task.md whose instruction has ``{{name}}`` placeholders, plus a
generator, ``[family] generator``, run as ``<generator> --seed <N> --out <DIR>``
(docs/runtime/episodes.md, ``family@1``). It writes ``instance.json`` (seed,
params, placeholders), ``agent/`` (files the agent sees, ``agent/`` standing
for ``/``), and ``verifier/`` (files for the verifier only).

BenchFlow runs the generator when it materializes a seed, in a fresh
container of the task's own image, as the spec requires: offline
(``--network none``), as the host's unprivileged uid with no-new-privileges,
with the package read-only at ``/package`` (its working folder) and an empty
writable ``/out``, no host environment, the task's ``[sandbox]`` CPU and
memory, and a 10-minute limit. The image is ``[sandbox] image``, or one
built from ``sandbox/Dockerfile`` and removed afterwards. This needs a Docker
daemon on the machine that loads the task.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.taskmd._util import table
from benchflow.taskmd._vendor import judgeprompt as jp
from benchflow.taskmd._vendor import taskmd as ref

GENERATOR_LIMIT_S = 600  # family@1: a runtime allows a generator 10 minutes
BUILD_LIMIT_S = 1800


class FamilyError(ValueError):
    """The instance could not be generated (``generator-failed``)."""


@dataclass(frozen=True)
class Instance:
    """One generated instance of a family."""

    seed: int
    params: dict[str, Any]
    placeholders: dict[str, str]
    split: str | None  # the family split whose rule selects the seed
    role: str | None  # that split's role: train, validation, or test
    root: Path  # the generator's output folder: instance.json, agent/, verifier/

    @property
    def params_jcs(self) -> str:
        return jp.jcs(self.params)


def fill_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """``{{name}}`` (spaces allowed inside the braces) replaced from ``placeholders``."""
    return ref.PLACEHOLDER.sub(lambda m: placeholders.get(m.group(1), m.group(0)), text)


def seed_split(config: dict[str, Any], seed: int) -> tuple[str | None, str | None]:
    """(split name, role) of the family split whose rule selects ``seed``, or (None, None)."""
    family = table(config.get("family"))
    variable = family.get("seed_param", "seed")
    variable = variable if isinstance(variable, str) else "seed"
    splits = table(family.get("splits"))
    for name, entry in splits.items():
        rule = (
            entry
            if isinstance(entry, str)
            else entry.get("rule")
            if isinstance(entry, dict)
            else None
        )
        role = (
            name
            if isinstance(entry, str)
            else entry.get("role", name)
            if isinstance(entry, dict)
            else None
        )
        try:
            if isinstance(rule, str) and ref.parse_rule(rule, variable)(seed):
                return str(name), str(role) if role is not None else None
        except ref.RuleError:
            continue
    return None, None


def _docker(
    args: list[str], *, timeout: float, what: str
) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        raise FamilyError(
            f"{what}: family@1 runs the generator in a fresh container of the task's image, "
            "and this machine has no docker command"
        )
    try:
        return subprocess.run(
            [docker, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FamilyError(f"{what}: timed out after {int(timeout)} s") from exc


def _tree_digest(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            h.update(path.relative_to(folder).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


def _interpreter(generator: Path) -> list[str]:
    """The generator's #! line as argv, as the kernel reads it: an absolute path and at most one argument."""
    try:
        with generator.open("rb") as handle:
            head = handle.readline(256).decode("utf-8", "replace").strip()
    except OSError as exc:
        raise FamilyError(f"{generator.name} cannot be read: {exc.strerror}") from exc
    if not head.startswith("#!"):
        raise FamilyError(f"{generator.name} has no #! line naming its interpreter")
    parts = head[2:].strip().split(None, 1)
    if not parts:
        raise FamilyError(f"{generator.name} has an empty #! line")
    return parts[:1] + ([parts[1].strip()] if len(parts) > 1 else [])


def generate(
    task_dir: Path,
    config: dict[str, Any],
    seed: int,
    out: Path,
    *,
    document: Any | None = None,
) -> Instance:
    """Run the family's generator for ``seed`` and return the checked instance.

    ``out`` must be an empty folder the caller owns. Raises :class:`FamilyError`
    when the generator cannot run, fails, or writes a malformed instance.
    """
    if not 0 <= seed < 2**53:
        raise FamilyError(f"seed {seed} is outside 0 to 2^53 - 1")
    family = table(config.get("family"))
    generator = family.get("generator")
    if not isinstance(generator, str) or not (task_dir / generator).is_file():
        raise FamilyError(
            f"[family] generator {generator!r} is not a file in the package"
        )
    argv = _interpreter(task_dir / generator)
    sandbox = table(config.get("sandbox"))
    declared_image = sandbox.get("image")
    image = declared_image if isinstance(declared_image, str) else None
    built: str | None = None
    if image is None:
        context = task_dir / "sandbox"
        if not (context / "Dockerfile").is_file():
            raise FamilyError(
                "the task has neither [sandbox] image nor sandbox/Dockerfile to run the generator in"
            )
        built = f"benchflow-taskmd-generator:{_tree_digest(context)}"
        result = _docker(
            ["build", "-q", "-t", built, str(context)],
            timeout=BUILD_LIMIT_S,
            what="building the task's image",
        )
        if result.returncode != 0:
            raise FamilyError(
                f"building the task's image failed: {(result.stderr or result.stdout).strip()[-500:]}"
            )
        image = built
    out.mkdir(parents=True, exist_ok=True)
    limits: list[str] = []
    cpus = sandbox.get("cpus")
    if isinstance(cpus, int) and not isinstance(cpus, bool):
        limits += ["--cpus", str(cpus)]
    from benchflow.taskmd.plan import megabytes

    memory = megabytes(sandbox.get("memory"))
    if memory is not None:
        limits += ["--memory", f"{memory}m"]
    run = [
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--security-opt",
        "no-new-privileges",
        *limits,
        "-v",
        f"{task_dir.resolve()}:/package:ro",
        "-v",
        f"{out.resolve()}:/out",
        "-w",
        "/package",
        "--entrypoint",
        argv[0],
        image,
        *argv[1:],
        f"/package/{generator}",
        "--seed",
        str(seed),
        "--out",
        "/out",
    ]
    try:
        result = _docker(
            run, timeout=GENERATOR_LIMIT_S, what=f"{generator} --seed {seed}"
        )
    finally:
        if built is not None:
            _docker(["rmi", built], timeout=120, what="removing the generator's image")
    if result.returncode != 0:
        raise FamilyError(
            f"{generator} --seed {seed} exited with status {result.returncode} (generator-failed): "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )
    return check_instance(
        task_dir, config, seed, out, generator=generator, document=document
    )


def check_instance(
    task_dir: Path,
    config: dict[str, Any],
    seed: int,
    out: Path,
    *,
    generator: str,
    document: Any | None = None,
) -> Instance:
    """family@1's rules for what a generator wrote, by the reference checker's own check."""
    doc = document if document is not None else ref.parse(task_dir)
    problems = [
        d.message
        for d in ref.check_instance(doc, out, seed, generator)
        if d.level == "error"
    ]
    if not (out / "instance.json").is_file():
        problems.append("no instance.json")
    if problems:
        raise FamilyError(
            f"{generator} wrote a malformed instance for seed {seed} (generator-failed): "
            + "; ".join(problems)
        )
    data = json.loads((out / "instance.json").read_text(encoding="utf-8"))
    split, role = seed_split(config, seed)
    return Instance(
        seed=seed,
        params=dict(data.get("params") or {}),
        placeholders={
            str(k): str(v) for k, v in (data.get("placeholders") or {}).items()
        },
        split=split,
        role=role,
        root=out,
    )


def scratch_dir() -> Path:
    """An empty folder for one generator run (the caller removes it)."""
    return Path(tempfile.mkdtemp(prefix="benchflow-taskmd-family-"))
