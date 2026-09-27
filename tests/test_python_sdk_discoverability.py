"""The first help text a user reads.

``help(benchflow)`` opened with a 2025 list of subsystems (Sandbox protocol,
ACP client, trajectory capture...) and named none of the entry points a user
calls. And ``bench eval run --help`` called ``--retry-attempts`` a "reserved"
override although it sets the retry count.
"""

from __future__ import annotations

import click
import typer.main

import benchflow as bf
from benchflow.cli.main import app


def test_the_package_docstring_names_the_entry_points() -> None:
    doc = bf.__doc__ or ""
    for name in (
        "run_sync",
        "arun",
        "RolloutConfig",
        "run_batch",
        "Evaluation",
        "branch",
        "load_job",
        "compare",
        "python-api.md",
    ):
        assert name in doc, name
    for name in doc.split("``")[1::2]:
        head = name.split("(")[0].split(".")[0]
        called = "(" in name or name[:1].isupper()
        if called and head.isidentifier():
            assert hasattr(bf, head), f"docstring names bf.{head}, which does not exist"


def test_retry_attempts_help_says_what_it_does() -> None:
    root = typer.main.get_command(app)
    group = root.get_command(click.Context(root), "eval")
    run = group.get_command(click.Context(group), "run")
    (opt,) = [p for p in run.params if "--retry-attempts" in getattr(p, "opts", [])]
    assert "Reserved" not in (opt.help or "")
    assert "retr" in (opt.help or "").lower()
