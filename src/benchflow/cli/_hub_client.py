"""``bench hub show|publish|push`` against a BenchFlow Hub, and the controls block ``bench run -d`` prints.

A dataset's controls are its reference solution (oracle, each task should score 1) and the no-op agent (nop, each
should score 0), run on that exact version. They are what tells a reader, before spending on a model, whether a
low score means a weak agent or a broken task. ``bench`` prints who ran them: the Hub itself (``hosted``), an admin
(``imported``) or the publisher (``uploaded``, reported).
"""

from __future__ import annotations

import json
import tarfile
import tempfile
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
            raise typer.Exit(1) from None
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
    """A .tar.gz of the job folder (links and credential files left out),
    with BenchFlow's count of it by its scoring rule (``scoring.json``, from
    ``benchflow.setting_groups.scoring_document``) beside the trials, so the
    Hub counts the same way; returns the number of files."""
    import io

    from benchflow.setting_groups import SCORING_FILE, scoring_document

    count = 0
    ours = f"{job_dir.name}/{SCORING_FILE}"

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        nonlocal count
        if info.issym() or info.islnk() or not (info.isfile() or info.isdir()):
            return None
        if Path(info.name).name in _PUSH_EXCLUDE or info.name == ours:
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        count += info.isfile()
        return info

    document = scoring_document(job_dir)
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(job_dir, arcname=job_dir.name, filter=keep)
        if document is not None:
            data = json.dumps(document, indent=2, allow_nan=False).encode()
            info = tarfile.TarInfo(ours)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
            count += 1
    return count


def push(
    job_dir: Path, hub: str | None, *, dataset: str | None, label: str | None
) -> None:
    """Upload a finished job (traces, results, an embodied trial's record and video) to the Hub as one run."""
    from benchflow.hub.client import HubError

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
    if runs:
        job_dir = runs[0]
    client = _client(hub)
    dataset = dataset or _job_dataset(job_dir)
    with tempfile.TemporaryDirectory(prefix="bench-hub-push-") as scratch:
        archive = Path(scratch) / f"{job_dir.name}.tar.gz"
        files = _pack_job(job_dir, archive)
        console.print(
            f"Pushing {escape(job_dir.name)}: {files} files, {archive.stat().st_size / 1e6:.1f} MB"
            + (f", dataset {escape(dataset)}" if dataset else ", no dataset recorded")
        )
        try:
            answer = client.upload_run(archive, name=label, dataset=dataset)
        except HubError as e:
            print_error(
                str(e),
                next_step=e.hint or "check the Hub address and your key (scope upload)",
            )
            raise typer.Exit(1) from None
    imported = answer.get("imported") if isinstance(answer, dict) else None
    for run in imported or []:
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id")
        traces = run.get("n_traces", run.get("traces"))
        bits = [f"run {run_id}"] + ([f"{traces} traces"] if traces is not None else [])
        console.print(
            f"[green]✓[/green] Pushed {escape(label or job_dir.parent.name)} to the Hub at {escape(client.url)}: "
            f"{escape(', '.join(bits))}",
            soft_wrap=True,
        )
        linked = run.get("dataset")
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
        console.print(f"  [dim]{escape(client.url)}/runs/{escape(str(run_id))}[/dim]")
    if not imported:
        console.print(
            f"[yellow]The Hub answered without a run:[/yellow] {escape(json.dumps(answer)[:300])}"
        )
