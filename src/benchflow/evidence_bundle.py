"""One shareable bundle of a run's evidence: ``bench bundle <job|trial>``.

Sharing a run's evidence used to mean collecting it by hand, file by file.
The bundle is one zip that reads without the job folder. For each trial it
holds the result, config and timing; the trajectory; the verifier's output
(test output, reward files, judge results, the guard and trusted-PATH
records); rubric reviews; regrades; collected artifacts; and the saved
workspace (``evidence/``) when the trial saved one. For the job it holds the
run card, ``summary.json`` and ``report.md``.

Every text file is redacted as a trajectory upload is
(:func:`benchflow.publish.redact.redact_value_to_stability`), and a file whose
redaction does not settle is left out. Links and other non-regular files are
left out too. ``bundle.json`` lists each file with its size, SHA-256 and the
values masked in it, and every file left out with the reason. The zip is
deterministic: the same folder gives the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "benchflow.evidence-bundle/1"
MANIFEST = "bundle.json"
#: What a trial folder contributes: these files, and everything under these
#: folders (``evidence`` only with the workspace).
TRIAL_FILES = (
    "result.json",
    "config.json",
    "timing.json",
    "prompts.json",
    "steps.json",
    "regrade.json",
    "verification.json",
    "artifacts-manifest.json",
    "trust-state.json",
    "locked-system-path.json",
    "kill-before-grading.json",
)
TRIAL_FOLDERS = ("trajectory", "verifier", "reviews", "regrade", "rounds", "artifacts")
WORKSPACE_FOLDER = "evidence"
JOB_FILES = (
    "run-card.json",
    "summary.json",
    "report.md",
    "evaluation.json",
    "results.jsonl",
    "regrade-summary.json",
)
MAX_FILE_BYTES = 256 * 1024**2
MAX_TOTAL_BYTES = 4 * 1024**3
_TEXT_LIMIT = 16 * 1024**2
_EPOCH = (1980, 1, 1, 0, 0, 0)


class BundleError(Exception):
    """The bundle could not be made (nothing to bundle, or too big)."""


@dataclass
class BundleResult:
    path: Path
    sha256: str
    trials: list[str]
    files: int
    masked: dict[str, int]
    left_out: list[dict[str, str]] = field(default_factory=list)


def _is_trial(folder: Path) -> bool:
    return (folder / "result.json").is_file()


def _trial_folders(root: Path) -> list[Path]:
    if _is_trial(root):
        return [root]
    return sorted(
        child for child in root.iterdir() if child.is_dir() and _is_trial(child)
    )


def _candidates(folder: Path, names: tuple[str, ...], dirs: tuple[str, ...]):
    for name in names:
        path = folder / name
        if path.exists() or path.is_symlink():
            yield path
    for name in dirs:
        top = folder / name
        if top.is_symlink():
            yield top
        elif top.is_dir():
            yield from sorted(top.rglob("*"))


def _redact_text(text: str, masked: Counter[str]) -> str:
    from benchflow.publish.redact import redact_value_to_stability

    redacted, _ = redact_value_to_stability(text, categories=masked)
    return redacted


def _redacted_bytes(path: Path, data: bytes, masked: Counter[str]) -> bytes:
    """*data* with its secrets masked; unchanged when it holds none or is not
    text. Raises ValueError when the redaction does not settle."""
    from benchflow.publish.redact import redact_value_to_stability

    if len(data) > _TEXT_LIMIT:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    found: Counter[str] = Counter()
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            out = _redact_text(text, found)
        else:
            clean, _ = redact_value_to_stability(value, categories=found)
            out = json.dumps(clean, indent=2, ensure_ascii=False) + "\n"
    elif path.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                lines.append(_redact_text(line, found))
                continue
            line_found: Counter[str] = Counter()
            clean, _ = redact_value_to_stability(value, categories=line_found)
            lines.append(
                json.dumps(clean, separators=(",", ":"), ensure_ascii=False)
                if line_found
                else line
            )
            found.update(line_found)
        out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    else:
        out = _redact_text(text, found)
    if not found:
        return data
    masked.update(found)
    return out.encode("utf-8")


def build_bundle(source: Path, out: Path, *, workspace: bool = True) -> BundleResult:
    """Write the evidence bundle of *source* (a job folder or one trial folder)
    to *out* and return what it holds."""
    from benchflow import __version__

    source = Path(source)
    if not source.is_dir():
        raise BundleError(f"{source} is not a folder")
    trials = _trial_folders(source)
    if not trials:
        raise BundleError(f"{source} holds no trial (no result.json in it or under it)")
    job = source if trials != [source] else None
    paths: list[Path] = []
    if job is not None:
        paths += _candidates(job, JOB_FILES, ())
    folders = (*TRIAL_FOLDERS, WORKSPACE_FOLDER) if workspace else TRIAL_FOLDERS
    for trial in trials:
        paths += _candidates(trial, TRIAL_FILES, folders)
    base = source if job is not None else source.parent
    entries: list[tuple[str, bytes, int]] = []
    left_out: list[dict[str, str]] = []
    masked: Counter[str] = Counter()
    total = 0
    for path in paths:
        name = path.relative_to(base).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            reason = "a link" if stat.S_ISLNK(mode) else "not a regular file"
            left_out.append({"path": name, "reason": reason})
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            left_out.append({"path": name, "reason": f"over {MAX_FILE_BYTES} bytes"})
            continue
        found: Counter[str] = Counter()
        try:
            data = _redacted_bytes(path, path.read_bytes(), found)
        except ValueError:
            left_out.append({"path": name, "reason": "its redaction did not settle"})
            continue
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise BundleError(
                f"the bundle would pass {MAX_TOTAL_BYTES} bytes; "
                "leave the workspace out (--no-workspace)"
            )
        masked.update(found)
        entries.append((name, data, sum(found.values())))
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "benchflow_version": __version__,
        "source": source.name,
        "workspace": workspace,
        "trials": [trial.relative_to(base).as_posix() for trial in trials],
        "files": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "masked": count,
            }
            for name, data, count in sorted(entries)
        ],
        "masked": dict(sorted(masked.items())),
        "left_out": left_out,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in [
            (MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode()),
            *((name, data) for name, data, _ in sorted(entries)),
        ]:
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    partial.replace(out)
    return BundleResult(
        path=out,
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        trials=manifest["trials"],
        files=len(entries),
        masked=manifest["masked"],
        left_out=left_out,
    )
