"""``bench hub show|publish|push`` against a BenchFlow Hub, and the controls block ``bench run -d`` prints.

A dataset's controls are its reference solution (oracle, each task should score 1) and the no-op agent (nop, each
should score 0), run on that exact version. They are what tells a reader, before spending on a model, whether a
low score means a weak agent or a broken task. ``bench`` prints who ran them: the Hub itself (``hosted``), an admin
(``imported``) or the publisher (``uploaded``, reported).
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import console, print_error, print_hint

CONTROL_AGENTS = ("oracle", "nop")
_SOURCES = {
    "hosted": "run by the Hub",
    "imported": "imported by an admin",
    "uploaded": "reported by the publisher",
}
_EXPECT_WORDS = {"oracle": "scored 1", "nop": "scored 0"}
_NAMES = {"oracle": "oracle", "nop": "no-op"}
# What a missing control leaves untested: the oracle shows each task can be solved, the no-op that none passes for
# doing nothing.
_UNTESTED = {
    "oracle": "whether each task can be solved is untested; a broken task scores like a failed agent",
    "nop": "whether a task passes with no work is untested; such a task scores like a capable agent",
}
# A pushed job never carries these: credentials and the operator's own settings.
_PUSH_EXCLUDE = {".env", ".netrc", "auth.json", ".credentials.json"}
# How long _pack_job may mask one over-limit text file line by line.
_LINE_MASK_BUDGET_SEC = 600.0
_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_KEY_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")
# What _pack_job sends as it is: a real binary, by extension or magic bytes.
_BINARY_SUFFIXES = frozenset(
    [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".tif",
        ".tiff",
        ".mp4",
        ".m4v",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".zip",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
        ".tar",
        ".whl",
        ".jar",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".pdf",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".safetensors",
        ".parquet",
        ".so",
        ".pyc",
        ".wasm",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
    ]
)
_BINARY_MAGIC = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"PK\x03\x04",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"(\xb5/\xfd",
    b"SQLite format 3\x00",
    b"%PDF",
    b"\x1aE\xdf\xa3",
    b"\x7fELF",
    b"RIFF",
    b"OggS",
    b"fLaC",
    b"ID3",
)


def _client(hub: str | None):
    from benchflow.hub.client import HubClient, HubError

    try:
        return HubClient.from_env(hub)
    except HubError as e:
        print_error(str(e), next_step=e.hint or "pass --hub <url>")
        raise typer.Exit(getattr(e, "exit_code", 2)) from None


def _short(digest: str | None, n: int = 12) -> str:
    if not digest:
        return "-"
    algo, _, hexpart = digest.partition(":")
    return f"{algo}:{hexpart[:n]}" if hexpart else digest[:n]


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _date(stamp: Any) -> str:
    return str(stamp or "")[:10] or "?"


def control_lines(controls: dict[str, Any] | None) -> list[str]:
    """The controls of one dataset version as plain lines (Rich markup escaped), worst news first."""
    if not isinstance(controls, dict):
        return ["[yellow]controls: the Hub did not say[/yellow]"]
    by_agent = {
        c.get("agent"): c for c in controls.get("controls") or [] if isinstance(c, dict)
    }
    lines: list[str] = []
    for agent in CONTROL_AGENTS:
        c = by_agent.get(agent)
        label = "oracle" if agent == "oracle" else "no-op "
        if c is None:
            lines.append(
                f"[yellow]{label} not run on this version[/yellow]\n         {_UNTESTED[agent]}"
            )
            continue
        tasks, ok = int(c.get("tasks") or 0), int(c.get("as_expected") or 0)
        who = _SOURCES.get(str(c.get("source")), str(c.get("source") or "?"))
        tries = (
            c.get("tries_per_task") if isinstance(c.get("tries_per_task"), dict) else {}
        )
        lo, hi = tries.get("min"), tries.get("max")
        tried = (
            ""
            if lo is None
            else f"{lo} {_plural(int(lo), 'try')} per task"
            if lo == hi
            else f"{lo}-{hi} tries per task"
        )
        runs = int(c.get("runs") or 1)
        ran_with = " on ".join(
            x
            for x in (
                f"bench {c.get('bench_version')}" if c.get("bench_version") else "",
                str(c.get("sandbox") or ""),
            )
            if x
        )
        where = ", ".join(
            x
            for x in (
                f"run {c.get('run_id')}"
                if c.get("run_id")
                else "the run is hidden from your workspace",
                f"latest of {runs} runs" if runs > 1 else "",
                tried,
                ran_with,
                f"finished {_date(c.get('finished_at'))}"
                if c.get("finished_at")
                else "",
            )
            if x
        )
        style = "green" if c.get("ok") else "red"
        # The result and who ran it, then what qualifies it, then where to find the run (each fits 80 columns).
        line = f"[{style}]{label} {ok}/{tasks} {_EXPECT_WORDS[agent]}[/{style}], {escape(who)}"
        if c.get("content_checked") is False:
            line += "\n         [yellow]content not checked[/yellow]: its records don't name this version's digest"
        unexpected = [str(t) for t in c.get("unexpected") or []]
        if unexpected:
            shown = ", ".join(unexpected[:5]) + (
                f" and {len(unexpected) - 5} more" if len(unexpected) > 5 else ""
            )
            line += f"\n         [red]not as expected: {escape(shown)}[/red]"
        errored = int(c.get("errored_tasks") or 0)
        if errored:
            line += f"\n         {errored} {_plural(errored, 'task')} errored, which checks nothing"
        disagreeing = int(c.get("disagreeing_runs") or 0)
        if disagreeing:
            line += (
                f"\n         [red]{disagreeing} other {_plural(disagreeing, 'run')} of it on this version "
                f"came out the other way[/red]"
            )
        line += f"\n         [dim]{escape(where)}[/dim]"
        lines.append(line)
    return lines


def _status(controls: dict[str, Any] | None) -> str:
    """One line: validated or not, and why, with who ran the controls when the Hub didn't."""
    if not isinstance(controls, dict):
        return "[yellow]not validated[/yellow]: the Hub did not say"
    by_agent = {
        c.get("agent"): c for c in controls.get("controls") or [] if isinstance(c, dict)
    }
    if controls.get("validated") is True:
        sources = {str(c.get("source")) for c in by_agent.values()}
        who = (
            "run by the Hub"
            if sources == {"hosted"}
            else "reported by the publisher, not re-run by the Hub"
            if sources == {"uploaded"}
            else "not all run by the Hub"
        )
        return f"[green]validated[/green]: both controls behaved on every task ({who})"
    missing = [a for a in CONTROL_AGENTS if a not in by_agent]
    failed = [a for a, c in by_agent.items() if not c.get("ok")]
    unchecked = [a for a, c in by_agent.items() if c.get("content_checked") is False]
    flaky = [a for a, c in by_agent.items() if int(c.get("disagreeing_runs") or 0)]
    why = (
        "a control did not behave"
        if failed
        else "a control's runs on this version came out both ways"
        if flaky
        else "a control has not run on this version"
        if missing
        else f"content not checked for the {_NAMES.get(str(unchecked[0]), str(unchecked[0]))}'s run"
        if unchecked
        else "a control did not cover every task"
    )
    return f"[bold yellow]not validated[/bold yellow]: {why}"


def print_controls(
    name: str,
    version: str,
    controls: dict[str, Any] | None,
    hub: str | None = None,
    running: str | None = None,
) -> None:
    console.print(
        f"Controls for {escape(name)}@{escape(version)}: {_status(controls)}",
        soft_wrap=True,
    )
    for line in control_lines(controls):
        console.print(f"  {line}", soft_wrap=True)
    if not hub or (isinstance(controls, dict) and controls.get("validated") is True):
        return
    if running in CONTROL_AGENTS:
        # This run is that control: the next step is to push it, not to start it again.
        print_hint(
            f"  this run is the {'oracle' if running == 'oracle' else 'no-op'} control; when it ends, push its job:",
            style="dim",
        )
        print_hint(f"    bench hub push <job> --hub {hub}")
        return
    by_agent = {
        c.get("agent"): c
        for c in (controls or {}).get("controls") or []
        if isinstance(c, dict)
    }
    failed = sorted(
        {
            str(t)
            for c in by_agent.values()
            if c.get("content_checked") is not False
            for t in c.get("unexpected") or []
        }
    )
    if failed:
        # Rerunning the same bytes can't fix a task: a fix is a new version.
        print_hint(
            f"  fix {', '.join(failed[:5])}{' and more' if len(failed) > 5 else ''}, then publish the fixed tasks as a new version:",
            style="dim",
        )
        print_hint(
            f"    bench hub publish <tasks-dir> --name {name}@<new version> --hub {hub}"
        )
        return
    # Controls not yet run on this version, or run on content not checked against it: run them by name, then push.
    to_run = [
        a
        for a in CONTROL_AGENTS
        if a not in by_agent or by_agent[a].get("content_checked") is False
    ]
    if to_run:
        print_hint(
            "  run them by name, so their records name this version, then push each job:",
            style="dim",
        )
        for agent in to_run:
            print_hint(f"    bench run -d {name}@{version} --hub {hub} --agent {agent}")
        print_hint(f"    bench hub push <job> --hub {hub}")


def run_controls_notice(resolved: Any, agent: str | None = None) -> None:
    """Before a run of a Hub dataset: its controls, so a reader sees a broken version before spending on it."""
    from benchflow.hub.client import HUB_REGISTRY_PREFIX, HubClient, HubError

    registry = str(getattr(resolved, "registry", "") or "")
    if not registry.startswith(HUB_REGISTRY_PREFIX):
        return
    try:
        controls = HubClient.from_env(registry[len(HUB_REGISTRY_PREFIX) :]).controls(
            resolved.name, resolved.version
        )
    except HubError as e:
        print_hint(
            f"warning: could not read the dataset's controls: {e}",
            style="yellow",
            stderr=True,
        )
        return
    print_controls(
        resolved.name,
        resolved.version,
        controls,
        registry[len(HUB_REGISTRY_PREFIX) :],
        running=(agent or "").lower() or None,
    )


def show(spec: str, hub: str | None, *, output_json: bool) -> None:
    """A Hub dataset version: what it is, where it came from, its controls and its tasks."""
    from benchflow._utils.dataset_registry import (
        DatasetResolutionError,
        parse_dataset_spec,
    )
    from benchflow.hub.client import HubError

    try:
        name, version = parse_dataset_spec(spec)
    except DatasetResolutionError as e:
        print_error(str(e), next_step="name the dataset as org/name@version")
        raise typer.Exit(2) from None
    client = _client(hub)
    try:
        doc = client.dataset(name, version)
        controls = client.controls(name, version)
    except HubError as e:
        print_error(str(e), next_step=e.hint or "check the name and version")
        raise typer.Exit(2 if e.status == 404 else 1) from None
    if output_json:
        typer.echo(json.dumps({"dataset": doc, "controls": controls}))
        return
    tasks = [t for t in doc.get("tasks") or [] if isinstance(t, dict)]
    kind = f"{doc['kind']} " if doc.get("kind") else ""
    console.print(
        f"[bold]{escape(name)}@{escape(version)}[/bold]: {len(tasks)} {escape(kind)}{_plural(len(tasks), 'task')} "
        f"on the Hub at {escape(client.url)}",
        soft_wrap=True,
    )
    console.print(
        f"  digest {escape(_short(doc.get('digest')))}, published {_date(doc.get('published_at'))}"
        + (f" by {escape(str(doc['published_by']))}" if doc.get("published_by") else "")
    )
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    origin = " @ ".join(
        str(x) for x in (src.get("repo"), str(src.get("commit") or "")[:12]) if x
    )
    exporter = f" ({src['exporter']})" if src.get("exporter") else ""
    # What the publisher passed to `bench hub publish`; the Hub checks the bytes, not these claims.
    console.print(
        f"  source {escape(origin + exporter) if origin else '[dim]not given[/dim]'}"
        + (" [dim](as the publisher declared)[/dim]" if origin else ""),
        soft_wrap=True,
    )
    license_ = str(doc.get("license") or "")
    console.print(
        f"  license {escape(license_) if license_ else '[dim]not given[/dim]'}"
        + (" [dim](as the publisher declared)[/dim]" if license_ else ""),
        soft_wrap=True,
    )
    console.print()
    print_controls(name, version, controls, client.url)
    per_task: dict[str, dict[str, Any]] = {}
    unchecked: set[str] = set()
    for c in (controls or {}).get("controls") or []:
        if isinstance(c, dict) and c.get("agent") in CONTROL_AGENTS:
            if c.get("content_checked") is False:
                unchecked.add(c["agent"])
            for row in c.get("per_task") or []:
                if isinstance(row, dict) and row.get("task"):
                    per_task.setdefault(str(row["task"]), {})[c["agent"]] = row
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Task", style="cyan")
    table.add_column("Digest")
    table.add_column("Oracle", justify="right")
    table.add_column("No-op", justify="right")

    def cell(agent: str, row: dict[str, Any] | None, expect: float) -> str:
        if not row:
            return "[dim]-[/dim]"
        reward = row.get("reward")
        if reward is None:
            return f"[red]{escape(str(row.get('outcome') or 'error'))}[/red]"
        if agent in unchecked:
            # The run may have used other bytes than these digests: show what it reported, not a pass or a fail.
            return f"[yellow]{float(reward):.1f}*[/yellow]"
        style = "green" if float(reward) == expect else "red"
        return f"[{style}]{float(reward):.1f}[/{style}]"

    for t in sorted(tasks, key=lambda t: str(t.get("name"))):
        rows = per_task.get(str(t.get("name")), {})
        table.add_row(
            escape(str(t.get("name"))),
            escape(_short(t.get("digest"))),
            cell("oracle", rows.get("oracle"), 1.0),
            cell("nop", rows.get("nop"), 0.0),
        )
    console.print()
    console.print(table)
    if unchecked:
        print_hint(
            "  * content not checked: what that run reported; its records don't name this version's digest",
            style="dim",
        )
    if isinstance(controls, dict) and controls.get("validated") is True:
        console.print()
        print_hint(
            f"Run it: bench run -d {name}@{version} --hub {client.url} --agent <agent>",
            style="dim",
        )


def publish(
    tasks_dir: Path,
    spec: str,
    hub: str | None,
    *,
    description: str | None,
    license: str | None,
    kind: str | None,
    source_repo: str | None,
    source_commit: str | None,
    exporter: str | None,
) -> None:
    """Pack every task folder of ``tasks_dir`` and publish it as ``org/name@version``."""
    from benchflow import __version__
    from benchflow._utils.dataset_registry import (
        DatasetResolutionError,
        parse_dataset_spec,
    )
    from benchflow.hub.client import HubError, split_dataset_name
    from benchflow.hub.datasets import build_manifest

    try:
        name, version = parse_dataset_spec(spec)
        split_dataset_name(name)
    except (DatasetResolutionError, HubError) as e:
        print_error(
            str(e),
            next_step="name the version as org/name@version, e.g. robouse/metaworld-sample@0.6.1",
        )
        raise typer.Exit(2) from None
    client = _client(hub)
    source = {
        k: v
        for k, v in {
            "repo": source_repo,
            "commit": source_commit,
            "exporter": exporter,
        }.items()
        if v
    }
    with tempfile.TemporaryDirectory(prefix="bench-hub-publish-") as scratch:
        try:
            manifest, archives = build_manifest(
                tasks_dir,
                name,
                version,
                Path(scratch),
                description=description,
                license=license,
                bench_version=f">={__version__}",
                kind=kind,
                source=source or None,
            )
        except HubError as e:
            print_error(
                str(e), next_step="point at a folder that holds one folder per task"
            )
            raise typer.Exit(2) from None
        size = sum(p.stat().st_size for _, p in archives)
        console.print(
            f"Publishing {escape(name)}@{escape(version)}: {len(archives)} {_plural(len(archives), 'task')}, {size / 1e6:.1f} MB, "
            f"digest {escape(_short(manifest['digest']))}"
        )
        try:
            created, _doc = client.publish(manifest, archives)
        except HubError as e:
            step = (
                f"bench hub publish {tasks_dir} --name {name}@<new version> --hub {client.url}"
                if e.status == 409
                else e.hint or "check the Hub address and your key"
            )
            print_error(str(e), next_step=step)
            raise typer.Exit(e.exit_code) from None
    verb = "Published" if created else "Already published, unchanged:"
    console.print(
        f"[green]✓[/green] {verb} {escape(name)}@{escape(version)} on the Hub at {escape(client.url)}"
    )
    print_hint(
        "  run its controls, then push each job so the version shows them:", style="dim"
    )
    for agent in CONTROL_AGENTS:
        print_hint(
            f"    bench run -d {name}@{version} --hub {client.url} --agent {agent}"
        )
    print_hint(f"    bench hub push <job> --hub {client.url}")


def _run_name(given: Path, run: Path) -> str:
    """A pushed run's name without ``--label``: the job's name its
    summary.json records, else the folder named on the command line, never
    the jobs folder above it (``bench hub push jobs/demo2`` showed
    the run as "jobs")."""
    try:
        summary = json.loads((run / "summary.json").read_text())
    except (OSError, json.JSONDecodeError):
        summary = None
    recorded = summary.get("job_name") if isinstance(summary, dict) else None
    if isinstance(recorded, str) and recorded.strip():
        return recorded.strip()
    return given.name


def _job_dataset(job_dir: Path) -> str | None:
    """The dataset a job ran (``org/name@version``), from a trial's config.json, if it recorded one."""
    for config in sorted(job_dir.glob("*/config.json"))[:50]:
        try:
            data = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name, version = data.get("dataset_name"), data.get("dataset_version")
        if isinstance(name, str) and isinstance(version, str) and name and version:
            return f"{name}@{version}"
    return None


def _pack_job(job_dir: Path, dest: Path) -> int:
    """A .tar.gz of the job folder (links, credential files and a hosted
    run's raw evidence left out),
    with BenchFlow's count of it by its scoring rule (``scoring.json``, from
    ``benchflow.setting_groups.scoring_document``) beside the trials, so the
    Hub counts the same way; returns the number of files.

    Every text file goes up with its secrets masked, as ``bench bundle``
    masks them (``evidence_bundle._redacted_bytes``): the agent's own output
    and test.sh's are raw on disk. Only a real binary (``_is_binary``: its
    magic bytes or its extension) goes up as it is; any other file is text,
    a stray non-UTF-8 byte or NUL included, and goes up masked
    (``_masked_bytes``). A text file over the masking limit is masked line by
    line (``_masked_lines``). A text file is left out and named, never sent
    raw, only when its masking does not settle or the line-by-line pass runs
    past its time budget."""
    import io

    from benchflow._utils.raw_evidence import is_raw_evidence
    from benchflow.evidence_bundle import _TEXT_LIMIT
    from benchflow.setting_groups import SCORING_FILE, scoring_document

    count = 0
    ours = f"{job_dir.name}/{SCORING_FILE}"
    masked: Counter[str] = Counter()
    left_out: list[str] = []

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if info.issym() or info.islnk() or not (info.isfile() or info.isdir()):
            return None
        if Path(info.name).name in _PUSH_EXCLUDE or info.name == ours:
            return None
        # A hosted run's raw vf-eval output stays on this machine.
        source = job_dir.parent / info.name
        if info.isdir() and is_raw_evidence(source.parent, source.name):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    def add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
        nonlocal count
        info = keep(tar.gettarinfo(str(path), arcname))
        if info is None:
            return
        if info.isdir():
            tar.addfile(info)
            for child in sorted(os.listdir(path)):
                add(tar, path / child, f"{arcname}/{child}")
            return
        if _is_binary(path):
            with path.open("rb") as handle:
                tar.addfile(info, handle)
            count += 1
            return
        if info.size > _TEXT_LIMIT:
            with tempfile.SpooledTemporaryFile(max_size=64 * 1024**2) as spool:
                if not _masked_lines(path, masked, spool):
                    left_out.append(arcname)
                    return
                info.size = spool.seek(0, os.SEEK_END)
                spool.seek(0)
                tar.addfile(info, spool)
            count += 1
            return
        try:
            data = _masked_bytes(path, path.read_bytes(), masked)
        except ValueError:
            left_out.append(arcname)
            return
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        count += 1

    document = scoring_document(job_dir)
    with tarfile.open(dest, "w:gz") as tar:
        add(tar, job_dir, job_dir.name)
        if document is not None:
            data = json.dumps(document, indent=2, allow_nan=False).encode()
            info = tarfile.TarInfo(ours)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
            count += 1
    for name in left_out:
        console.print(
            f"Not pushed: {escape(name)} (a text file whose secrets could not be masked)"
        )
    return count


def _masked_lines(path: Path, masked: Counter[str], out: Any) -> bool:
    """Write *path*, a text file too big for one masking pass, to *out*
    masked line by line: a line the redactor would change, alone or read with
    the line before it (``password:`` then its value), is masked whole, the
    line before it too in the second case, and so is every line of a private
    key block, whose body lines carry no cue of their own. Bytes that are not
    UTF-8 round-trip unchanged (surrogateescape). False when the pass runs
    past ``_LINE_MASK_BUDGET_SEC``."""
    import time

    from benchflow.evidence_bundle import _redact_text
    from benchflow.publish.redact import REDACTED

    def write(raw: bytes, line: str, mask: bool) -> None:
        if mask:
            ending = line[len(line.rstrip("\r\n")) :]
            out.write((REDACTED + ending).encode("utf-8", "surrogateescape"))
        else:
            out.write(raw)

    deadline = time.monotonic() + _LINE_MASK_BUDGET_SEC
    in_key = False
    held: tuple[bytes, str, bool] | None = None
    with path.open("rb") as handle:
        for raw in handle:
            if time.monotonic() > deadline:
                return False
            line = raw.decode("utf-8", "surrogateescape")
            found: Counter[str] = Counter()
            if in_key or _KEY_BEGIN.search(line):
                in_key = not _KEY_END.search(line)
                found["private key block"] += 1
            else:
                _redact_text(line, found)
            mask = bool(found)
            if not mask and held is not None and not held[2]:
                _redact_text(held[1] + line, found)
                if found:
                    mask = True
                    held = (held[0], held[1], True)
            masked.update(found)
            if held is not None:
                write(*held)
            held = (raw, line, mask)
    if held is not None:
        write(*held)
    return True


def _masked_bytes(path: Path, data: bytes, masked: Counter[str]) -> bytes:
    """*data*, a text file within the masking limit, masked as ``bench
    bundle`` masks it; a stray non-UTF-8 byte does not make it binary: it is
    read with surrogateescape and round-trips unchanged. Raises ValueError
    when the masking does not settle."""
    from benchflow.evidence_bundle import _redact_text, _redacted_bytes

    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        found: Counter[str] = Counter()
        text = _redact_text(data.decode("utf-8", "surrogateescape"), found)
        if not found:
            return data
        masked.update(found)
        return text.encode("utf-8", "surrogateescape")
    return _redacted_bytes(path, data, masked)


def _is_binary(path: Path) -> bool:
    """Whether *path* is a real binary: its extension or magic bytes say so
    (images, video, audio, archives, databases, fonts, compiled code) and its
    first 64 KiB are not clean text (a NUL, or not UTF-8). Some magic is
    plain ASCII (RIFF, ID3, %PDF), so a text log named ``case.db`` or
    starting ``RIFF`` is text. Any other file is text to ``_pack_job``,
    however its bytes decode."""
    import codecs

    with path.open("rb") as handle:
        head = handle.read(64 * 1024)
    claimed = (
        path.suffix.lower() in _BINARY_SUFFIXES
        or head.startswith(_BINARY_MAGIC)
        or head[4:8] == b"ftyp"
    )
    if not claimed:
        return False
    if b"\0" in head:
        return True
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        return True
    return False


def push(
    job_dir: Path, hub: str | None, *, dataset: str | None, label: str | None
) -> None:
    """Upload a finished job (traces, results, an embodied trial's record and video) to the Hub as one run."""
    from benchflow._utils.raw_evidence import raw_evidence_refusal
    from benchflow.hub.client import HubError

    refusal = raw_evidence_refusal(job_dir)
    if refusal is not None:
        print_error(
            refusal, next_step="bench hub push <the job folder that holds the run>"
        )
        raise typer.Exit(2)
    if not (job_dir / "summary.json").is_file():
        print_error(
            f"{job_dir} is not a finished job folder (no summary.json)",
            next_step="pass a job folder under ./jobs (or --jobs-dir) that bench run wrote",
        )
        raise typer.Exit(2)
    # bench run writes jobs/<job>/<time>/<trial>: a job folder holding one run pushes that run.
    runs = sorted(
        p
        for p in job_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and (p / "summary.json").is_file()
    )
    if len(runs) > 1:
        print_error(
            f"{job_dir} holds {len(runs)} runs: "
            + ", ".join(p.name for p in runs[-3:])
            + (" and more" if len(runs) > 3 else ""),
            next_step=f"push one run: bench hub push {job_dir / runs[-1].name}",
        )
        raise typer.Exit(2)
    given = job_dir
    if runs:
        job_dir = runs[0]
    name = label or _run_name(given, job_dir)
    client = _client(hub)
    dataset = dataset or _job_dataset(job_dir)
    if dataset and client.has_registry() is False:
        print_error(
            f"{client.url} has no dataset registry, so it can't link this run to {dataset}; "
            "nothing was uploaded",
            next_step="push it to a Hub that has the dataset registry: bench hub push <job> --hub <url>",
        )
        raise typer.Exit(2)
    with tempfile.TemporaryDirectory(prefix="bench-hub-push-") as scratch:
        archive = Path(scratch) / f"{job_dir.name}.tar.gz"
        files = _pack_job(job_dir, archive)
        console.print(
            f"Pushing {escape(name)}: {files} files, {archive.stat().st_size / 1e6:.1f} MB"
            + (f", dataset {escape(dataset)}" if dataset else ", no dataset recorded")
        )
        try:
            # Named here, so the Hub stores the name bench prints.
            answer = client.upload_run(archive, name=name, dataset=dataset)
        except HubError as e:
            print_error(
                str(e),
                next_step=e.hint or "check the Hub address and your key (scope upload)",
            )
            raise typer.Exit(1) from None
    imported = answer.get("imported") if isinstance(answer, dict) else None
    stored = [r for r in imported or [] if isinstance(r, dict) and r.get("run_id")]
    if not stored:
        print_error(
            f"{client.url} answered the upload without a run: {json.dumps(answer)[:300]}",
            next_step="nothing on the Hub names this job; check the Hub's log, then push again",
        )
        raise typer.Exit(1)
    failures = [
        f for f in [_report_push(client, r, dataset, name) for r in stored] if f
    ]
    if failures:
        for message, step in failures:
            print_error(message, next_step=step)
        raise typer.Exit(1)


_NO_READ_SCOPE = (
    "this key can't read runs back; it needs the read scope as well as upload"
)


def _kept_folders(client: Any, run_id: str) -> tuple[bool | None, str]:
    """Whether the Hub kept a pushed run's job folder, which its trial pages read (the run's ``job_dir``):
    True or False, or None with the reason bench couldn't tell."""
    from benchflow.hub.client import HubError

    try:
        stored = client.run(run_id)
    except HubError as e:
        if e.status in (401, 403):
            return None, _NO_READ_SCOPE
        return None, f"reading the run back failed: {e}"
    return bool(stored.get("job_dir")), ""


def _report_push(
    client: Any, run: dict[str, Any], dataset: str | None, label: str
) -> tuple[str, str] | None:
    """Print a pushed run that the Hub linked to the dataset it was sent and whose trial folders it kept, and
    return None; for anything less, print nothing and return the error and its next step, never a tick."""
    run_id = str(run["run_id"])
    where = f"{client.url}/runs/{run_id}"
    traces = run.get("n_traces", run.get("traces"))
    bits = [f"run {run_id}"] + ([f"{traces} traces"] if traces is not None else [])
    linked = run.get("dataset")
    problems = []
    if dataset and linked != dataset:
        note = run.get("dataset_note")
        problems.append(
            f"it isn't linked to {dataset}"
            + (f" ({note})" if note else "")
            + ", so it doesn't count for that version"
        )
    kept, why = _kept_folders(client, run_id)
    if kept is False:
        problems.append(
            "not kept: the Hub kept no trial folders for it, so its run card, scoring, "
            "verifier output and artifacts aren't on the Hub"
        )
    elif kept is None:
        problems.append(
            f"not checked: bench couldn't check that the Hub kept its trial folders: {why}"
        )
    if problems:
        if not dataset:
            problems.append(
                "not linked: the job recorded no dataset, so the run counts for no "
                "dataset version (--dataset org/name@version names one)"
            )
        # What arrived, then each thing that did not, a line each: the first
        # line alone is not the whole push.
        message = "\n  ".join(
            [
                f"the Hub at {client.url} stored {label} as {', '.join(bits)}, "
                "but not all of it:",
                *problems,
            ]
        )
        if why == _NO_READ_SCOPE:
            return (
                message,
                f"give the key the read scope, then look at the run: {where}",
            )
        return (
            message,
            f"its traces are at {where}"
            + (
                ""
                if dataset
                else "; to link it to a dataset version, push again with "
                "--dataset org/name@version"
            ),
        )
    console.print(
        f"[green]✓[/green] Pushed {escape(label)} to the Hub at {escape(client.url)}: "
        f"{escape(', '.join(bits))}",
        soft_wrap=True,
    )
    if linked and run.get("dataset_content") == "checked":
        console.print(
            f"  linked to {escape(str(linked))}: its records name this version's digest"
        )
    elif linked:
        console.print(
            f"  linked to {escape(str(linked))}, [yellow]content not checked[/yellow]: its records don't name "
            "this version's digest, so it won't count as a control",
            soft_wrap=True,
        )
    if not dataset and not linked:
        console.print(
            "  not linked: the job recorded no dataset, so the run counts for no dataset "
            "version (--dataset org/name@version names one)",
            soft_wrap=True,
        )
    console.print(f"  [dim]{escape(where)}[/dim]")
    return None
