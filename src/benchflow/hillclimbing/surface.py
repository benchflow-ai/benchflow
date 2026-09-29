"""The surface ``bench hillclimb`` may edit, its versions, diffs and history.

A surface is one or more paths the agent under test receives through a
mechanism BenchFlow already has:

- a **skills folder** (a directory whose subfolders hold ``SKILL.md``) is
  deployed like ``bench eval run --skills-dir`` (``skills_dir`` with
  ``skill_mode="with-skill"``): mounted at ``/skills`` and linked into each
  agent's skill paths. It replaces a task's bundled skills for the run.
- a **prompt file** is prepended to every task prompt through the task's
  ``agent.prompt_prefix``, set per run with the C-axis overlay
  (``--config-override``), which records it by content hash.

Each version is a folder ``surfaces/vNNN/`` holding ``skills/`` and/or
``prompt.md``. Kept versions are committed to a small git repository,
``surface-history/``, one commit per kept patch.
"""

from __future__ import annotations

import copy
import difflib
import logging
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SurfaceKind = Literal["skills", "prompt"]
SKILLS = "skills"
PROMPT_FILE = "prompt.md"

# Limits on a version, so a runaway proposer cannot fill the disk or bloat
# every image build.
MAX_FILES = 500
MAX_BYTES = 5 * 1024 * 1024
MAX_DIFF_CHARS = 200_000


class SurfaceError(ValueError):
    """A surface path that cannot be used, or an edited surface that is invalid."""


@dataclass(frozen=True)
class SurfaceSpec:
    """One editable path: its kind and where it came from."""

    kind: SurfaceKind
    source: Path

    @property
    def name(self) -> str:
        """Its path inside a version folder."""
        return SKILLS if self.kind == "skills" else PROMPT_FILE

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "source": str(self.source), "name": self.name}


def parse_surface(value: str | Path) -> SurfaceSpec:
    """``skills=PATH``, ``prompt=PATH``, or a bare path (a directory is a skills
    folder, a file a prompt)."""
    text = str(value)
    kind: str | None = None
    if "=" in text and text.split("=", 1)[0] in ("skills", "prompt"):
        kind, text = text.split("=", 1)
    path = Path(text).expanduser()
    if not path.exists():
        raise SurfaceError(f"surface path does not exist: {path}")
    if path.is_symlink():
        raise SurfaceError(f"surface path is a symlink: {path}")
    if kind is None:
        kind = "skills" if path.is_dir() else "prompt"
    if kind == "skills" and not path.is_dir():
        raise SurfaceError(f"a skills surface must be a directory: {path}")
    if kind == "prompt" and not path.is_file():
        raise SurfaceError(f"a prompt surface must be a file: {path}")
    return SurfaceSpec(kind="skills" if kind == "skills" else "prompt", source=path)


def check_specs(specs: Sequence[SurfaceSpec]) -> None:
    if not specs:
        raise SurfaceError(
            "give at least one --surface (a skills folder or a prompt file)"
        )
    kinds = [s.kind for s in specs]
    for kind in ("skills", "prompt"):
        if kinds.count(kind) > 1:
            raise SurfaceError(f"at most one {kind} surface, got {kinds.count(kind)}")


def _copy_tree(src: Path, dest: Path) -> list[str]:
    """Copy regular files and folders; skip symlinks (returned)."""
    skipped: list[str] = []

    def ignore(directory: str, names: list[str]) -> set[str]:
        out = set()
        for name in names:
            p = Path(directory) / name
            if p.is_symlink():
                skipped.append(str(p.relative_to(src)))
                out.add(name)
            elif name in (".git", "__pycache__", ".DS_Store"):
                out.add(name)
        return out

    shutil.copytree(src, dest, ignore=ignore, symlinks=True)
    return skipped


def _inventory(root: Path) -> tuple[int, int]:
    files = bytes_ = 0
    for dirpath, _dirs, names in os.walk(root, followlinks=False):
        for name in names:
            p = Path(dirpath) / name
            if p.is_file() and not p.is_symlink():
                files += 1
                bytes_ += p.stat().st_size
    return files, bytes_


def validate_version(version_dir: Path, specs: Sequence[SurfaceSpec]) -> list[str]:
    """Problems that make a version unusable (empty list: usable)."""
    problems: list[str] = []
    for spec in specs:
        path = version_dir / spec.name
        if spec.kind == "skills":
            if not path.is_dir():
                problems.append(f"{spec.name}/ is missing")
                continue
            for child in sorted(path.iterdir()):
                if child.is_dir() and not (child / "SKILL.md").is_file():
                    problems.append(f"{spec.name}/{child.name} has no SKILL.md")
                elif child.is_dir():
                    text = (child / "SKILL.md").read_text(errors="replace")
                    if not text.startswith("---"):
                        problems.append(
                            f"{spec.name}/{child.name}/SKILL.md has no YAML frontmatter"
                        )
        elif not path.is_file():
            problems.append(f"{spec.name} is missing")
    files, size = _inventory(version_dir)
    if files > MAX_FILES:
        problems.append(f"{files} files (limit {MAX_FILES})")
    if size > MAX_BYTES:
        problems.append(f"{size} bytes (limit {MAX_BYTES})")
    return problems


class SurfaceStore:
    """The version folders under ``<run>/surfaces``."""

    def __init__(self, root: Path, specs: Sequence[SurfaceSpec]) -> None:
        check_specs(specs)
        self.root = root
        self.specs = list(specs)

    def path(self, version: str) -> Path:
        return self.root / version

    def baseline(self) -> tuple[str, list[str]]:
        """Copy the user's surface paths into ``v000``; return it and skipped links."""
        dest = self.path("v000")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        skipped: list[str] = []
        for spec in self.specs:
            if spec.kind == "skills":
                skipped += _copy_tree(spec.source, dest / spec.name)
            else:
                shutil.copyfile(spec.source, dest / spec.name)
        problems = validate_version(dest, self.specs)
        if problems:
            raise SurfaceError(
                "the starting surface is not usable: " + "; ".join(problems)
            )
        return "v000", skipped

    def add(self, version: str, edited: Path) -> tuple[list[str], list[str]]:
        """Import an edited surface as ``version``: copy only the surface's own
        paths, drop symlinks. Returns (problems, skipped symlinks)."""
        dest = self.path(version)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        skipped: list[str] = []
        for spec in self.specs:
            src = edited / spec.name
            if spec.kind == "skills":
                if src.is_dir() and not src.is_symlink():
                    skipped += _copy_tree(src, dest / spec.name)
            elif src.is_file() and not src.is_symlink():
                shutil.copyfile(src, dest / spec.name)
        return validate_version(dest, self.specs), skipped


def deploy_settings(
    version_dir: Path,
    specs: Sequence[SurfaceSpec],
    base_override: dict[str, Any] | None,
) -> dict[str, Any]:
    """The EvaluationConfig fields that give a version to the agent under test."""
    settings: dict[str, Any] = {
        "skills_dir": None,
        "skill_mode": "no-skill",
        "config_override": copy.deepcopy(base_override) if base_override else None,
    }
    for spec in specs:
        path = version_dir / spec.name
        if spec.kind == "skills":
            settings["skills_dir"] = str(path)
            settings["skill_mode"] = "with-skill"
        else:
            text = path.read_text().strip()
            if text:
                override = settings["config_override"] or {}
                agent = dict(override.get("agent") or {})
                agent["prompt_prefix"] = text
                override["agent"] = agent
                settings["config_override"] = override
    return settings


# ---------------------------------------------------------------------------
# Diffs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffStats:
    files_changed: int
    added: int
    removed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "files_changed": self.files_changed,
            "added": self.added,
            "removed": self.removed,
        }


def _files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for dirpath, _dirs, names in os.walk(root, followlinks=False):
        for name in names:
            p = Path(dirpath) / name
            if p.is_file() and not p.is_symlink():
                out[p.relative_to(root).as_posix()] = p
    return out


def _lines(path: Path | None) -> list[str] | None:
    if path is None:
        return []
    data = path.read_bytes()
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def diff_versions(old: Path, new: Path) -> tuple[str, DiffStats]:
    """A unified diff from version folder ``old`` to ``new`` and its line counts."""
    a, b = _files(old), _files(new)
    chunks: list[str] = []
    changed = added = removed = 0
    for rel in sorted(set(a) | set(b)):
        la, lb = _lines(a.get(rel)), _lines(b.get(rel))
        if la is None or lb is None:
            if (
                a.get(rel) is None
                or b.get(rel) is None
                or (a[rel].read_bytes() != b[rel].read_bytes())
            ):
                changed += 1
                chunks.append(f"Binary file {rel} differs\n")
            continue
        if la == lb:
            continue
        changed += 1
        lines = list(
            difflib.unified_diff(
                la,
                lb,
                fromfile=f"a/{rel}" if rel in a else "/dev/null",
                tofile=f"b/{rel}" if rel in b else "/dev/null",
            )
        )
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        chunks.append(
            "".join(line if line.endswith("\n") else line + "\n" for line in lines)
        )
    return "".join(chunks), DiffStats(changed, added, removed)


def added_text(diff: str) -> str:
    """The lines a diff adds, without their ``+`` marks."""
    return "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


# ---------------------------------------------------------------------------
# Pasted-content check
# ---------------------------------------------------------------------------

SHINGLE_WORDS = 12


def _shingles(text: str, n: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def pasted_spans(added: str, sources: dict[str, str]) -> list[dict[str, str]]:
    """Runs of ``SHINGLE_WORDS`` words the added text copies from a source.

    ``sources`` maps a label (``train task instruction: <task>``) to text the
    proposer read. One match per source is reported.
    """
    mine = _shingles(added)
    if not mine:
        return []
    found = []
    for label, text in sources.items():
        common = mine & _shingles(text)
        if common:
            found.append({"source": label, "words": " ".join(min(common))})
    return found


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

_GIT_IDENTITY = (
    "-c",
    "user.name=bench hillclimb",
    "-c",
    "user.email=hillclimb@benchflow.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "init.defaultBranch=main",
)


class SurfaceHistory:
    """A git repository whose tree is the current surface: one commit per kept
    version. Without a ``git`` executable the history is skipped."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.enabled = shutil.which("git") is not None
        self.error: str | None = None if self.enabled else "git is not installed"

    def _git(self, *args: str) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        proc = subprocess.run(
            ["git", *_GIT_IDENTITY, *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def _sync(self, version_dir: Path, names: Iterable[str]) -> None:
        for name in names:
            target = self.root / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            src = version_dir / name
            if src.is_dir():
                _copy_tree(src, target)
            elif src.is_file():
                shutil.copyfile(src, target)

    def commit(
        self, version_dir: Path, names: Iterable[str], message: str
    ) -> str | None:
        """Make the tree equal ``version_dir`` and commit; return the commit id."""
        if not self.enabled:
            return None
        try:
            if not (self.root / ".git").is_dir():
                self.root.mkdir(parents=True, exist_ok=True)
                self._git("init", "-q")
            self._sync(version_dir, list(names))
            self._git("add", "-A")
            self._git("commit", "-q", "--allow-empty", "-m", message)
            return self._git("rev-parse", "HEAD")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            logger.warning("Surface history: %s", exc)
            self.error = str(exc)
            self.enabled = False
            return None


def truncate_diff(diff: str, limit: int = MAX_DIFF_CHARS) -> tuple[str, bool]:
    if len(diff) <= limit:
        return diff, False
    return diff[:limit] + "\n[diff truncated]\n", True
