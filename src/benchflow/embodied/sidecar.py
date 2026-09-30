"""The simulator-sidecar wiring: embodied task folders as native BenchFlow packages.

`EmbodiedTaskFormat` is a base class for task formats (see docs/task-formats.md) whose tasks run a simulator next
to the agent. A benchmark subclasses it and supplies only what is specific to it: the task.md key of its block, the
simulator image (a Dockerfile plus its code), the episode factory, and its reference solutions. The base class
writes the rest of every package:

    <out_root>/runtime-<hash>/          simulator image build context, shared by every task built from it:
      Dockerfile ...                    the benchmark's (build_runtime)
      benchflow_embodied/benchflow/     a vendored copy of benchflow.embodied under a stub benchflow package
      embodied-sim-entry                the simulator entry point
    <out_root>/tasks/<key>/<task-id>/   the native package; <key> hashes the task, the runtime and the options
      task.md                           schema 1.3; the block moves to metadata.<block_key>; verifier.service: simulator
      environment/Dockerfile, robo      agent image: python + the `robo` client (benchflow/embodied/robo.py)
      environment/docker-compose.yaml   `main` + the trusted `simulator` service (embeds the task definition)
      verifier/{verifier.md,test.sh}    the physical verifier (benchflow.embodied.verifier), run in `simulator`
      oracle/                           reference solution (write_oracle), or the noop control

A simulator Dockerfile must copy ``benchflow_embodied/`` to ``/opt/benchflow_embodied`` (on PYTHONPATH) and
``embodied-sim-entry`` to ``/usr/local/bin/``; see ``SIM_DOCKERFILE_SNIPPET``.

Packages are content-addressed (task files, runtime, format settings and options) and written atomically, so
concurrent loads are safe; each package gets its own random oracle token, so two builds are not byte-identical. Seeded
variants (`materialize_variant(seed=...)`) are named ``<task-id>--seed-<n>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .spec import restore_boundary

EMBODIED_DIR = Path(__file__).resolve().parent
TEMPLATES = EMBODIED_DIR / "templates"
SEED_SEP = "--seed-"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

SIM_DOCKERFILE_SNIPPET = """\
COPY benchflow_embodied/ /opt/benchflow_embodied/
COPY embodied-sim-entry /usr/local/bin/embodied-sim-entry
ENV PYTHONPATH=/opt/benchflow_embodied${PYTHONPATH:+:$PYTHONPATH}
RUN chmod 755 /usr/local/bin/embodied-sim-entry && mkdir -p /rpc /episode /task /logs/verifier
"""

NOOP_SOLVE = """#!/bin/bash
# Negative control (noop oracle): look at the robot, never move it, exit without `robo done`.
# The verifier must close the episode itself (outcome agent_exited) and the reward must be 0.
set -euo pipefail
robo info
robo observe
"""


# ---- small utilities --------------------------------------------------------------------------------------


def read_task_md(path: Path) -> tuple[dict, str]:
    text = Path(path).read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path}: task.md has no YAML frontmatter")
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    return meta, body.lstrip("\n")


def frontmatter(path: Path) -> dict | None:
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return meta if isinstance(meta, dict) else None


def tree_files(root: Path, ignore=_IGNORE) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        skip = ignore(dirpath, dirnames + filenames)
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        out += [Path(dirpath) / f for f in sorted(filenames) if f not in skip]
    return out


def hash_tree(h, root: Path, ignore=_IGNORE) -> None:
    for f in tree_files(root, ignore):
        h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")


def atomic_dir(dst: Path, build: Callable[[Path], None]) -> Path:
    """Build a directory in a temp sibling, then rename it into place (safe under concurrent loads)."""
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


def vendor_embodied(dst: Path) -> Path:
    """Copy benchflow.embodied into ``dst/benchflow/embodied`` under a stub ``benchflow/__init__.py``.

    The simulator image then imports ``benchflow.embodied`` without installing BenchFlow (whose top-level package
    pulls in the whole engine). Returns ``dst``.
    """
    pkg = Path(dst) / "benchflow"
    shutil.copytree(
        EMBODIED_DIR,
        pkg / "embodied",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "templates"),
    )
    (pkg / "__init__.py").write_text(
        '"""Stub: only benchflow.embodied is vendored into this image."""\n'
    )
    return Path(dst)


def embodied_source_hash() -> str:
    h = hashlib.sha256()
    hash_tree(h, EMBODIED_DIR)
    return h.hexdigest()[:16]


def dump_yaml(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=10**6)


def split_seed_name(name: str) -> tuple[str, int | None]:
    """``metaworld-reach--seed-3`` -> (``metaworld-reach``, 3); a name without a seed suffix -> (name, None)."""
    base, sep, tail = name.rpartition(SEED_SEP)
    if sep and tail.isdigit():
        return base, int(tail)
    return name, None


# ---- the format base class --------------------------------------------------------------------------------


class EmbodiedTaskFormat:
    """Base class of task formats whose tasks run a simulator sidecar. Subclasses set the class attributes and
    implement `build_runtime`; they may override `write_oracle`, `runtime_hash_inputs` and `task_id`."""

    name: str = "embodied"
    # the top-level task.md key that marks a task of this format
    block_key: str = "embodied"
    # module:function run in the simulator (see benchflow.embodied.serve)
    episode_factory: str = ""
    format_version: str = "1"
    prompt_prefix: str = ""
    task_name_prefix: str = ""
    sim_cpus: str = "2"
    sim_memory: str = "3G"
    agent_cpus: int = 1
    agent_memory_mb: int = 2048
    wall_margin_s: int = 900  # the episode clock starts when the simulator is up, before the harness is installed
    default_agent_timeout_s: int = 900
    min_verifier_timeout_s: int = 300

    # ---- hooks ----------------------------------------------------------------------------------------------
    def build_runtime(self, dst: Path) -> None:
        """Write the simulator build context (Dockerfile and the benchmark's code) into ``dst``."""
        raise NotImplementedError

    def runtime_hash_inputs(self) -> list[Path]:
        """Folders/files whose content identifies the simulator runtime."""
        return []

    def task_id(self, task_dir: Path, meta: dict) -> str:
        return str((meta.get(self.block_key) or {}).get("id") or Path(task_dir).name)

    def write_oracle(self, task_dir: Path, meta: dict, dst: Path, token: str) -> None:
        """Default: the source folder's oracle/ (or solution/), with the oracle token exported first."""
        src = next(
            (
                p
                for p in (task_dir / "oracle", task_dir / "solution")
                if (p / "solve.sh").exists()
            ),
            None,
        )
        if src is None:
            raise FileNotFoundError(f"{task_dir}: no oracle/solve.sh")
        shutil.copytree(src, dst, ignore=_IGNORE)
        body = (src / "solve.sh").read_text()
        lines = body.splitlines()
        head = lines[:1] if lines and lines[0].startswith("#!") else ["#!/bin/bash"]
        rest = lines[1:] if lines and lines[0].startswith("#!") else lines
        (dst / "solve.sh").write_text(
            "\n".join([*head, f"export ROBO_ORACLE_TOKEN={token}", *rest, ""])
        )
        (dst / "solve.sh").chmod(0o755)

    # ---- TaskFormat protocol ----------------------------------------------------------------------------
    def detect(self, task_dir: Path) -> bool:
        md = Path(task_dir) / "task.md"
        try:
            if not md.is_file():
                return False
        except OSError:
            return False
        meta = frontmatter(md)
        return bool(meta) and isinstance(meta.get(self.block_key), dict)

    def materialize(self, task_dir: Path, out_root: Path) -> Path:
        return self.materialize_variant(task_dir, out_root)

    def materialize_variant(
        self,
        task_dir: Path,
        out_root: Path,
        *,
        seed: int | None = None,
        noop: bool = False,
        timeout: float | None = None,
        flat: bool = False,
    ) -> Path:
        """Write (or reuse) the native package of one task and return its folder.

        seed: override the task's seed (the folder is named ``<id>--seed-<n>``). noop: negative-control oracle.
        timeout: agent timeout in seconds. flat: write ``<out_root>/tasks/<name>`` (replacing an older copy) with
        a relative build context, for a bundle meant to be moved or published.
        """
        task_dir, out_root = Path(task_dir).resolve(), Path(out_root).resolve()
        meta, instruction = read_task_md(task_dir / "task.md")
        if not isinstance(meta.get(self.block_key), dict):
            raise ValueError(
                f"{task_dir}: not a {self.name} task (no {self.block_key}: block in task.md)"
            )
        meta = json.loads(json.dumps(meta))  # deep copy
        tid = self.task_id(task_dir, meta)
        block = meta[self.block_key]
        block.setdefault("id", tid)
        if seed is not None:
            block["seed"] = int(seed)
        if timeout is not None:
            meta.setdefault("agent", {})["timeout_sec"] = int(timeout)
        name = f"{tid}{SEED_SEP}{int(seed)}" if seed is not None else tid
        runtime = self.ensure_runtime(out_root)
        settings = json.dumps(
            {
                "format": [
                    self.name,
                    self.format_version,
                    self.block_key,
                    self.episode_factory,
                ],
                "prompt_prefix": self.prompt_prefix,
                "task_name_prefix": self.task_name_prefix,
                "resources": [
                    self.sim_cpus,
                    self.sim_memory,
                    self.agent_cpus,
                    self.agent_memory_mb,
                ],
                "timeouts": [
                    self.wall_margin_s,
                    self.default_agent_timeout_s,
                    self.min_verifier_timeout_s,
                ],
                "runtime": runtime.name,
                "name": name,
                "options": {"noop": noop, "seed": seed, "timeout": timeout},
            },
            sort_keys=True,
        )
        h = hashlib.sha256(settings.encode())
        hash_tree(h, task_dir)
        key = h.hexdigest()[:16]
        parent = out_root / "tasks" if flat else out_root / "tasks" / key
        variant = {
            "format": self.name,
            "base_task": tid,
            "seed": int(block.get("seed", 0)),
            "seed_override": seed is not None,
            "noop": noop,
        }

        def write(pkg: Path) -> None:
            env = pkg / "environment"
            env.mkdir(parents=True)
            token = secrets.token_hex(16)
            (pkg / "task.md").write_text(
                self.native_task_md(meta, instruction, variant)
            )
            shutil.copy2(TEMPLATES / "agent.Dockerfile", env / "Dockerfile")
            (env / "robo").write_text(
                "#!/usr/bin/env python3\n" + (EMBODIED_DIR / "robo.py").read_text()
            )
            (env / "robo").chmod(0o755)
            # absolute in the cache (BenchFlow may stage a task copy elsewhere); relative in a movable bundle
            context = (
                os.path.relpath(runtime, parent / name / "environment")
                if flat
                else str(runtime)
            )
            (env / "docker-compose.yaml").write_text(self.compose(meta, context, token))
            shutil.copytree(TEMPLATES / "verifier", pkg / "verifier", ignore=_IGNORE)
            if noop:
                (pkg / "oracle").mkdir()
                (pkg / "oracle" / "solve.sh").write_text(NOOP_SOLVE)
                (pkg / "oracle" / "solve.sh").chmod(0o755)
            else:
                self.write_oracle(task_dir, meta, pkg / "oracle", token)
            (pkg / f".{self.name}-source.json").write_text(
                json.dumps(
                    {
                        "source": str(task_dir),
                        "key": key,
                        "runtime": runtime.name,
                        "noop": noop,
                        "seed": seed,
                        "timeout": timeout,
                        "format_version": self.format_version,
                    },
                    indent=2,
                )
            )

        if flat:
            dst = parent / name
            marker = dst / f".{self.name}-source.json"
            if marker.exists() and json.loads(marker.read_text()).get("key") == key:
                return dst
            if dst.exists():
                shutil.rmtree(dst)
            return atomic_dir(dst, write)
        return atomic_dir(parent, lambda d: write(d / name)) / name

    # ---- pieces -------------------------------------------------------------------------------------------
    def runtime_hash(self) -> str:
        h = hashlib.sha256(f"{self.name}|{self.format_version}".encode())
        hash_tree(h, EMBODIED_DIR)
        for p in self.runtime_hash_inputs():
            p = Path(p)
            if p.is_dir():
                hash_tree(
                    h,
                    p,
                    shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".DS_Store", "bundled_tasks"
                    ),
                )
            elif p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def ensure_runtime(self, out_root: Path) -> Path:
        def build(d: Path) -> None:
            self.build_runtime(d)
            vendor_embodied(d / "benchflow_embodied")
            shutil.copy2(TEMPLATES / "sim-entry.sh", d / "embodied-sim-entry")
            (d / "embodied-sim-entry").chmod(0o755)

        return atomic_dir(Path(out_root) / f"runtime-{self.runtime_hash()}", build)

    def native_task_md(
        self, meta: dict, instruction: str, variant: dict | None = None
    ) -> str:
        block = meta[self.block_key]
        task = dict(meta.get("task") or {})
        task.setdefault("name", f"{self.task_name_prefix}{block.get('id')}")
        metadata: dict[str, Any] = {
            "author_name": "benchflow",
            **(meta.get("metadata") or {}),
            self.block_key: block,
        }
        # Every package is an embodied task: metadata.embodied carries the variant
        # and the task's own restore declaration (mode, world_restore,
        # action_replay), which is checked here rather than at branch time.
        declared = metadata.get("embodied") or {}
        if isinstance(declared, dict):
            metadata["embodied"] = {**declared, **(variant or {})}
        restore_boundary(metadata)
        verifier_timeout = max(
            self.min_verifier_timeout_s,
            int((meta.get("verifier") or {}).get("timeout_sec", 0) or 0),
        )
        fm = {
            "schema_version": "1.3",
            "task": task,
            "metadata": metadata,
            "agent": {
                "timeout_sec": int(
                    float(
                        (meta.get("agent") or {}).get(
                            "timeout_sec", self.default_agent_timeout_s
                        )
                    )
                )
            },
            "verifier": {
                "service": "simulator",
                "user": "root",
                "timeout_sec": verifier_timeout,
            },
            "sandbox": {
                "cpus": self.agent_cpus,
                "memory_mb": self.agent_memory_mb,
                "build_timeout_sec": 1800,
            },
        }
        return (
            "---\n"
            + dump_yaml(fm)
            + "---\n\n"
            + self.prompt_prefix
            + instruction.strip()
            + "\n"
        )

    def sim_task_md(self, meta: dict) -> str:
        """The task definition the simulator needs (frontmatter only; the prompt is not in it)."""
        keys = (
            "schema_version",
            "task",
            "metadata",
            "agent",
            "verifier",
            self.block_key,
        )
        return "---\n" + dump_yaml({k: meta[k] for k in keys if k in meta}) + "---\n"

    def compose(self, meta: dict, sim_context: str, token: str) -> str:
        # compose interpolates $VARS in values; `$$` is a literal dollar sign
        block = "\n".join(
            "        " + ln.replace("$", "$$")
            for ln in self.sim_task_md(meta).rstrip().splitlines()
        )
        agent_timeout = float(
            (meta.get("agent") or {}).get("timeout_sec", self.default_agent_timeout_s)
        )
        text = (TEMPLATES / "docker-compose.yaml").read_text()
        return (
            text.replace("__SIM_CONTEXT__", json.dumps(sim_context))
            .replace("__MAX_WALL_S__", str(int(agent_timeout + self.wall_margin_s)))
            .replace("__ORACLE_TOKEN__", token)
            .replace("__FACTORY__", self.episode_factory)
            .replace("__SIM_CPUS__", str(self.sim_cpus))
            .replace("__SIM_MEMORY__", str(self.sim_memory))
            .replace("__TASK_MD__", block)
        )
