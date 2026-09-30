"""The ``taskmd`` task format: task.md draft 2 packages as native BenchFlow packages.

``TaskMdFormat`` claims a folder whose ``task.md`` does not open with YAML
frontmatter. Every native BenchFlow ``task.md`` (and every v0.6 or robouse
file) opens with ``---``, so the two never claim the same folder; a draft 2
file opens with its instruction (or a canary comment).

Materializing parses the package with the reference parser, refuses it when
the parser reports an error or when any run-affecting field cannot be
honored (``benchflow.taskmd.plan``), and writes a native package::

    <out_root>/tasks/<key>/<name>/
      task.md             native frontmatter (schema 1.3) + the agent's instruction
      environment/        sandbox/ (the build context), or a Dockerfile FROM [sandbox] image;
                          a family instance's agent/ files are baked in (taskmd-instance/)
      verifier/ or tests/ verifier/, where [verifier] mount places it, with verifier.md
                          selecting the taskmd strategy; a family instance's verifier/ at instance/
      oracle/ or solution/  oracle/ (or a control script as solve.sh, for a control variant)
      .taskmd/package/    task.md, verifier/, and referenced files, for the judges (never uploaded)
      .taskmd-source.json provenance

``<name>`` is the source folder's name (``<name>--seed-<n>`` for a family seed,
``<name>--control-<id>`` for a control), so trial names and ``--include``
match the source folder. ``<key>`` hashes the source tree, the reference
tools' commit, this module's format version, and the options, so packages
are content-addressed and written atomically (``atomic_dir``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from benchflow.taskmd import family as family_mod
from benchflow.taskmd.plan import Plan, plan_package
from benchflow.taskmd.reference import errors as reference_errors
from benchflow.taskmd.reference import parse_package, reference_commit

FORMAT_NAME = "taskmd"
FORMAT_VERSION = "1"
FORMAT_LABEL = "task.md draft 2"
SEED_SEP = "--seed-"
CONTROL_SEP = "--control-"
SOURCE_FILE = ".taskmd-source.json"
JUDGE_PACKAGE = ".taskmd/package"
INSTANCE_CONTEXT = "taskmd-instance"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
# The package tree hash leaves these root paths out (docs/package.md, "Tree hash").
_TREE_EXCLUDED = ("evidence", "changes.json", ".git")


class TaskMdError(ValueError):
    """A draft 2 package BenchFlow will not run, with every reason."""

    def __init__(self, task_dir: Path, reasons: list[str], *, headline: str) -> None:
        self.task_dir = task_dir
        self.reasons = reasons
        lines = [f"{task_dir.name}: {headline}"] + [f"  - {r}" for r in reasons]
        super().__init__("\n".join(lines))


def is_taskmd_package(task_dir: Path) -> bool:
    """Whether ``task_dir/task.md`` is a draft 2 document. Cheap, and never raises.

    The first line that is not blank decides, after a byte order mark: native,
    v0.6, and robouse task.md files open with ``---`` (YAML frontmatter), and a
    draft 2 file never does. An empty file is left to the native loader.
    """
    path = Path(task_dir) / "task.md"
    try:
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            for _ in range(10_000):
                raw = handle.readline(65_536)
                if not raw:
                    return False
                line = raw.decode("utf-8", "replace").lstrip("﻿").strip()
                if line:
                    return line != "---"
    except OSError:
        return False
    return False


def package_tree_hash(task_dir: Path) -> str:
    """The package's tree hash (docs/package.md): every regular file in path order,
    each as its relative path, a zero byte, and the SHA-256 of its contents, less
    the root's evidence/, changes.json, and .git. Symbolic links do not count."""
    entries: list[tuple[bytes, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(task_dir):
        here = Path(dirpath)
        if here == task_dir:
            dirnames[:] = [d for d in dirnames if d not in _TREE_EXCLUDED]
        for filename in filenames:
            path = here / filename
            rel = path.relative_to(task_dir).as_posix()
            if here == task_dir and filename in _TREE_EXCLUDED:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            entries.append((rel.encode("utf-8"), hashlib.sha256(path.read_bytes()).digest()))
    digest = hashlib.sha256()
    for key, content in sorted(entries):
        digest.update(key + b"\0" + content)
    return "sha256:" + digest.hexdigest()


def load_plan(task_dir: Path) -> Plan:
    """Parse and plan a package; raise :class:`TaskMdError` unless BenchFlow can run it."""
    task_dir = Path(task_dir).resolve()
    document = parse_package(task_dir)
    problems = reference_errors(document)
    if problems:
        raise TaskMdError(
            task_dir,
            problems,
            headline=f"the {FORMAT_LABEL} reference parser reports errors, so BenchFlow will not run it",
        )
    plan = plan_package(document, task_dir)
    if not plan.ok:
        raise TaskMdError(
            task_dir,
            [str(f) for f in plan.refused],
            headline=f"BenchFlow cannot honor these {FORMAT_LABEL} fields yet, so it refuses the task",
        )
    return plan


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "control"


def controls(task_dir: Path) -> dict[str, str]:
    """The task's declared control scripts: control id -> package path.

    The id is ``<kind>`` for ``no_op``, and ``<kind>-<script name>`` for the lists
    (``known_bad``, ``near_miss``, ``cheats``), with ``_`` as ``-``.
    """
    document = parse_package(Path(task_dir).resolve())
    config = document.config if isinstance(document.config, dict) else {}
    table = config.get("integrity", {}).get("controls", {}) if isinstance(config.get("integrity"), dict) else {}
    out: dict[str, str] = {}
    for kind, value in (table.items() if isinstance(table, dict) else ()):
        if kind == "injection":
            continue
        scripts = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for script in scripts:
            if not isinstance(script, str):
                continue
            ident = _slug(kind) if isinstance(value, str) else _slug(f"{kind}-{Path(script).stem}")
            out[ident] = script
    return out


class TaskMdFormat:
    """task.md draft 2, as a BenchFlow task format (docs/task-authoring-taskmd-v2.md)."""

    name = FORMAT_NAME
    format_version = FORMAT_VERSION

    def detect(self, task_dir: Path) -> bool:
        return is_taskmd_package(Path(task_dir))

    def materialize(self, task_dir: Path, out_root: Path) -> Path:
        plan = load_plan(task_dir)
        if plan.family is not None:
            raise TaskMdError(
                plan.task_dir,
                ["[family]: a family's instance comes from a seed; run it with --seeds (for example --seeds 0-4)"],
                headline="this task is a family of seeded instances",
            )
        return self._materialize(plan, Path(out_root))

    def materialize_variant(
        self,
        task_dir: Path,
        out_root: Path,
        *,
        seed: int | None = None,
        control: str | None = None,
    ) -> Path:
        """The package of one family seed, or of one control (the control script runs as the oracle)."""
        plan = load_plan(task_dir)
        if seed is not None and plan.family is None:
            raise TaskMdError(
                plan.task_dir,
                ["--seeds: the task declares no [family], so it has no seeded instances"],
                headline="this task is not a family",
            )
        if seed is None and plan.family is not None:
            raise TaskMdError(
                plan.task_dir,
                ["[family]: pass a seed"],
                headline="this task is a family of seeded instances",
            )
        return self._materialize(plan, Path(out_root), seed=seed, control=control)

    # ---- writing ------------------------------------------------------------------------------------

    def _materialize(
        self,
        plan: Plan,
        out_root: Path,
        *,
        seed: int | None = None,
        control: str | None = None,
    ) -> Path:
        task_dir = plan.task_dir
        script: str | None = None
        if control is not None:
            declared = controls(task_dir)
            script = declared.get(control)
            if script is None or not (task_dir / script).is_file():
                raise TaskMdError(
                    task_dir,
                    [f"control {control!r} is not a script the task declares; it declares {', '.join(sorted(declared)) or 'none'}"],
                    headline="no such control",
                )
        name = task_dir.name
        if seed is not None:
            name = f"{name}{SEED_SEP}{int(seed)}"
        if control is not None:
            name = f"{name}{CONTROL_SEP}{control}"
        source_tree = package_tree_hash(task_dir)
        settings = json.dumps(
            {
                "format": [FORMAT_NAME, FORMAT_VERSION],
                "reference": reference_commit(),
                "name": name,
                "seed": seed,
                "control": control,
                "shared": {u: hashlib.sha256(p.read_bytes()).hexdigest() for u, p in (plan.judging.shared if plan.judging else {}).items()},
                "module": _module_digest(),
            },
            sort_keys=True,
        )
        key = hashlib.sha256((settings + source_tree).encode()).hexdigest()[:16]
        parent = out_root / "tasks" / key

        def build(tmp: Path) -> None:
            pkg = tmp / name
            pkg.mkdir(parents=True)
            instance = None
            scratch = None
            try:
                if seed is not None:
                    scratch = family_mod.scratch_dir()
                    instance = family_mod.generate(
                        task_dir,
                        plan.document.config,
                        int(seed),
                        scratch / "out",
                        document=plan.document,
                    )
                _write_package(plan, pkg, source_tree=source_tree, instance=instance, control_script=script, control=control)
            finally:
                if scratch is not None:
                    shutil.rmtree(scratch, ignore_errors=True)

        try:
            return atomic_dir(parent, build) / name
        except family_mod.FamilyError as exc:
            raise TaskMdError(
                task_dir,
                [str(exc)],
                headline=f"family@1 could not build the instance for seed {seed}",
            ) from exc


def atomic_dir(dst: Path, build: Any) -> Path:
    """Build a folder in a temporary sibling, then rename it into place.

    Concurrent loads of one task are safe: the first rename wins, and a later
    one finds the folder and returns it.
    """
    import tempfile

    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        build(tmp)
        try:
            tmp.rename(dst)
        except OSError:
            if not dst.exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


def _module_digest() -> str:
    """Identifies this materializer's code, so a changed writer never reuses a stale package."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=_IGNORE, symlinks=False)


def _dump_frontmatter(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=10**6)


def _write_package(
    plan: Plan,
    pkg: Path,
    *,
    source_tree: str,
    instance: family_mod.Instance | None,
    control_script: str | None,
    control: str | None,
) -> None:
    task_dir = plan.task_dir
    document = plan.document
    config: dict[str, Any] = document.config if isinstance(document.config, dict) else {}
    fm = json.loads(json.dumps(plan.frontmatter, default=str))
    placeholders = instance.placeholders if instance is not None else {}

    # environment/ --------------------------------------------------------------------------------
    env = pkg / "environment"
    sandbox_dir = task_dir / "sandbox"
    if sandbox_dir.is_dir():
        _copytree(sandbox_dir, env)
    else:
        env.mkdir()
    dockerfile = env / "Dockerfile"
    bake = instance is not None and (instance.root / "agent").is_dir() and any((instance.root / "agent").iterdir())
    if plan.image is not None:
        dockerfile.write_text(f"FROM {plan.image}\n")
        if not bake:
            fm.setdefault("sandbox", {})["docker_image"] = plan.image
    if bake and instance is not None:
        target = env / INSTANCE_CONTEXT
        if target.exists():
            raise ValueError(f"sandbox/{INSTANCE_CONTEXT} would collide with the family instance's files")
        _copytree(instance.root / "agent", target)
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(0o755 if os.access(path, os.X_OK) else 0o644)
            elif path.is_dir():
                path.chmod(0o755)
        text = dockerfile.read_text()
        dockerfile.write_text(
            text.rstrip("\n")
            + "\n\n# family@1: the instance's agent/ files, agent/ standing for /\n"
            + f"COPY {INSTANCE_CONTEXT}/ /\n"
        )
    if plan.compose is not None:
        (env / "docker-compose.yaml").write_text(yaml.safe_dump(plan.compose, sort_keys=False))

    # verifier/ (or tests/) -----------------------------------------------------------------------
    vdir = pkg / plan.verifier_dirname
    if (task_dir / "verifier").is_dir():
        _copytree(task_dir / "verifier", vdir)
    else:
        vdir.mkdir()
    if instance is not None and (instance.root / "verifier").is_dir():
        if (vdir / "instance").exists():
            raise ValueError("verifier/instance would collide with the family instance's verifier files")
        _copytree(instance.root / "verifier", vdir / "instance")
    strategy = _strategy(plan, config)
    (vdir / "verifier.md").write_text(_verifier_document(strategy))

    # oracle/ (or solution/) ----------------------------------------------------------------------
    odir = pkg / plan.oracle_dirname
    if control_script is not None:
        odir.mkdir()
        shutil.copy2(task_dir / control_script, odir / "solve.sh")
        (odir / "solve.sh").chmod(0o755)
    elif (task_dir / "oracle").is_dir():
        _copytree(task_dir / "oracle", odir)

    # environment variables of a family instance (family@1) ---------------------------------------
    if instance is not None:
        family_env = {"TASKMD_SEED": str(instance.seed), "TASKMD_PARAMS": instance.params_jcs}
        for section in ("oracle", "verifier"):
            table = fm.setdefault(section, {})
            table["env"] = {**table.get("env", {}), **family_env}

    # the judges' copy of the draft 2 package -----------------------------------------------------
    judge_pkg = pkg / JUDGE_PACKAGE
    judge_pkg.mkdir(parents=True)
    shutil.copy2(task_dir / "task.md", judge_pkg / "task.md")
    if (task_dir / "verifier").is_dir():
        _copytree(task_dir / "verifier", judge_pkg / "verifier")
    for rel in plan.judging.extra_files if plan.judging else []:
        src = task_dir / rel
        if src.is_file():
            (judge_pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, judge_pkg / rel)
    shared_files: dict[str, str] = {}
    for n, (url, path) in enumerate(sorted((plan.judging.shared if plan.judging else {}).items())):
        dest = pkg / ".taskmd" / "shared" / f"{n}-{path.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        shared_files[url] = dest.relative_to(pkg).as_posix()

    # task.md -------------------------------------------------------------------------------------
    instruction = family_mod.fill_placeholders(document.instruction, placeholders)
    turns = [
        {"stage": t["stage"], "prompt": family_mod.fill_placeholders(t["prompt"], placeholders)}
        for t in plan.turns
    ]
    metadata = fm.setdefault("metadata", {})
    taskmd: dict[str, Any] = {
        "format": FORMAT_LABEL,
        "format_version": FORMAT_VERSION,
        "source": str(task_dir),
        "source_tree": source_tree,
        "reference": {"repository": "task.md", "commit": reference_commit()},
        "config": json.loads(json.dumps(config, default=str)),
        "verifier_mount": _mount_of(plan.verifier_dirname),
        "oracle_mount": _mount_of(plan.oracle_dirname),
        "grading": strategy["grading"],
        "agent_refused": [{"field": f.field, "reason": f.detail} for f in plan.agent_refused],
    }
    if isinstance(config.get("name"), str):
        taskmd["name"] = config["name"]
    if plan.agent_user is not None:
        taskmd["agent_user"] = plan.agent_user
    if turns:
        taskmd["turns"] = turns
    if shared_files:
        taskmd["shared_rubrics"] = shared_files
    if instance is not None:
        taskmd["family"] = {
            "seed": instance.seed,
            "params": instance.params,
            "split": instance.split,
            "role": instance.role,
        }
    if control is not None:
        taskmd["control"] = {"id": control, "script": control_script}
    metadata["taskmd"] = taskmd
    body = instruction.strip("\n") + "\n"
    (pkg / "task.md").write_text("---\n" + _dump_frontmatter(fm) + "---\n\n" + body)
    (pkg / SOURCE_FILE).write_text(
        json.dumps(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "source": str(task_dir),
                "source_tree": source_tree,
                "seed": instance.seed if instance else None,
                "control": control,
                "reference_commit": reference_commit(),
            },
            indent=2,
        )
        + "\n"
    )


def _mount_of(dirname: str) -> str:
    return {"verifier": "/verifier", "tests": "/tests", "oracle": "/oracle", "solution": "/solution"}[dirname]


def _strategy(plan: Plan, config: dict[str, Any]) -> dict[str, Any]:
    """The ``taskmd`` verifier strategy's settings (benchflow.taskmd.verify)."""
    verifier = config.get("verifier") if isinstance(config.get("verifier"), dict) else {}
    sandbox = config.get("sandbox") if isinstance(config.get("sandbox"), dict) else {}
    isolation = verifier.get("isolation", "shared")
    workdir = None
    if isolation == "separate" and isinstance(verifier.get("sandbox"), dict):
        workdir = verifier["sandbox"].get("workdir")
    workdir = workdir or sandbox.get("workdir")
    strategy: dict[str, Any] = {
        "type": "taskmd",
        "grading": "rubric" if plan.judging is not None and plan.judging.rubric is not None else "script",
        "isolation": isolation,
        "offline": plan.verifier_network == "none",
    }
    if (plan.task_dir / "verifier" / "test.sh").is_file():
        strategy["command"] = "test.sh"
    if isinstance(workdir, str):
        strategy["workdir"] = workdir
    return strategy


def _verifier_document(strategy: dict[str, Any]) -> str:
    front = {
        "document_version": "0.3",
        "verifier": {
            "name": "taskmd",
            "default_strategy": "taskmd",
            "strategies": {"taskmd": strategy},
        },
    }
    return (
        "---\n"
        + _dump_frontmatter(front)
        + "---\n\n## verifier intent\n\n"
        + "Written by BenchFlow's task.md draft 2 format (benchflow.taskmd). The runtime runs "
        + "test.sh in the task's working folder and, for a rubric, decides each test-judged "
        + "criterion from /logs/verifier/ctrf.json, runs the model judges (judge-prompt@1, "
        + "judge-loop@1), and scores the rubric; without a rubric, test.sh writes the reward.\n"
    )


def judge_package_dir(task_path: Path) -> Path:
    """The draft 2 copy the judges compile prompts from, in a materialized package."""
    return Path(task_path) / JUDGE_PACKAGE


def taskmd_metadata(task_path: Path) -> dict[str, Any] | None:
    """``metadata.taskmd`` of a materialized package, or None for any other task."""
    path = Path(task_path) / "task.md"
    if not (Path(task_path) / SOURCE_FILE).is_file() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    try:
        front = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        return None
    metadata = front.get("metadata") if isinstance(front, dict) else None
    taskmd = metadata.get("taskmd") if isinstance(metadata, dict) else None
    return taskmd if isinstance(taskmd, dict) else None


def shared_rubrics(task_path: Path, meta: dict[str, Any]) -> dict[str, str]:
    """URL -> local file of the shared rubrics a package's rubric extends."""
    return {url: str(Path(task_path) / rel) for url, rel in (meta.get("shared_rubrics") or {}).items()}


__all__ = [
    "FORMAT_LABEL",
    "FORMAT_NAME",
    "TaskMdError",
    "TaskMdFormat",
    "controls",
    "is_taskmd_package",
    "judge_package_dir",
    "load_plan",
    "package_tree_hash",
    "shared_rubrics",
    "taskmd_metadata",
]
