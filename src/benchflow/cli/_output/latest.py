"""``latest``: the newest job under the jobs folder, for the commands that
read one job (``bench resume``, ``bench results``, ``bench hub push``).

A job is a folder holding ``evaluation.json`` or ``summary.json``; the newest
is the one whose record was written last. The name it resolves to is printed,
so the run that was read is never a guess.
"""

from __future__ import annotations

from pathlib import Path

LATEST = "latest"


def _record_time(folder: Path) -> float | None:
    times = [
        path.stat().st_mtime
        for path in (folder / "evaluation.json", folder / "summary.json")
        if path.is_file()
    ]
    return max(times) if times else None


def newest_job(jobs_dir: Path) -> Path | None:
    """The newest job folder directly under ``jobs_dir``, or None."""
    if not jobs_dir.is_dir():
        return None
    found = [
        (when, folder)
        for folder in jobs_dir.iterdir()
        if folder.is_dir()
        and not folder.is_symlink()
        and not folder.name.startswith((".", "_"))
        and (when := _record_time(folder)) is not None
    ]
    return max(found)[1] if found else None


def resolve(value: Path | str, jobs_dir: Path | str = "jobs") -> Path:
    """``value`` itself, or for ``latest`` the newest job under ``jobs_dir``
    (said on stderr); a ``latest`` with no job exits 2 with the next step."""
    if str(value) != LATEST or Path(value).exists():
        return Path(value)
    import typer

    from benchflow.cli._shared import err_console, print_error

    found = newest_job(Path(jobs_dir))
    if found is None:
        print_error(
            f"no job under {jobs_dir} for `latest`",
            next_step="run one first (bench run <tasks>), or name the job folder",
        )
        raise typer.Exit(2)
    err_console.print(f"latest: {found}", highlight=False, soft_wrap=True)
    return found
