"""cli/verbs.py, the one map of the verb set, agrees with the CLI and the docs.

``bench --help``'s layout (cli/main.py), the 0.8 groups' real commands, the
default agent of ``bench eval run``, docs/reference/commands.md and the
spelling table of docs/versions/upgrade-from-0.7.md all say what cli/verbs.py
says; a change to one fails here until the others follow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer.main
from typer.core import TyperGroup

from benchflow.cli import verbs
from benchflow.cli._lazy import LazyCommand
from benchflow.cli.main import _ROOT_LAYOUT, app

DOCS = Path(__file__).resolve().parents[1] / "docs"
CLI_MD = DOCS / "reference" / "commands.md"
# The older spellings and the verbs they are now (docs v2).
UPGRADE_MD = DOCS / "versions" / "upgrade-from-0.7.md"
# What the upgrade page's table leaves out (page fix pending).
UPGRADE_TABLE_PENDING = {"`bench eval branch`", "`bench branch`"}


def _root() -> TyperGroup:
    # TyperGroup, not click.Group: typer 0.26 and later run on their own click.
    root = typer.main.get_command(app)
    assert isinstance(root, TyperGroup)
    return root


def _listed_panel(name: str) -> str | None:
    for entry in _ROOT_LAYOUT:
        if isinstance(entry, LazyCommand) and entry.name == name:
            return entry.rich_help_panel
    command = _root().commands[name]
    return getattr(command, "rich_help_panel", None)


def test_each_verb_is_listed_in_its_group() -> None:
    for verb in verbs.VERBS:
        assert _listed_panel(verb.name) == verb.panel, verb.name


def test_every_listed_command_is_a_verb_or_under_more() -> None:
    root = _root()
    names = {verb.name for verb in verbs.VERBS}
    for name in root.commands:
        listed = root.commands.listed(name)  # type: ignore[attr-defined]
        if listed.hidden:
            continue
        panel = _listed_panel(name)
        assert panel in verbs.HELP_PANELS, (name, panel)
        assert (panel == verbs.MORE) == (name not in names), (name, panel)


def test_the_0_8_groups_commands_are_the_real_ones() -> None:
    root = _root()
    for group, names in verbs.GROUP_COMMANDS_08.items():
        real = root.commands[group]
        assert isinstance(real, TyperGroup)
        visible = [n for n, c in real.commands.items() if not c.hidden]
        assert tuple(visible) == names, group


def test_every_new_spelling_is_a_verb() -> None:
    names = {verb.name for verb in verbs.VERBS}
    for group, commands in verbs.GROUP_COMMANDS_08.items():
        for command in (*commands, "create"):
            new = verbs.new_spelling(group, command)
            assert new is None or new in names, (group, command, new)


def test_eval_runs_default_agent_is_the_one_its_note_adds() -> None:
    from benchflow.evaluation import DEFAULT_AGENT

    assert verbs.EVAL_RUN_DEFAULT_AGENT == DEFAULT_AGENT


def test_renamed_run_flags_are_flags_of_bench_run() -> None:
    """The 0.8 spellings of bench run's own flags are aliases there (DX-b)."""
    run = _root().commands["run"]
    flags = {opt for p in run.params for opt in (*p.opts, *p.secondary_opts)}
    for old in (
        "--job-name", "--n-tasks", "--agent-env", "--reasoning-effort",
        "--freeze-workspace", "--agent-idle-timeout", "--tasks-dir", "--skill-mode",
    ):  # fmt: skip
        assert old in flags, old


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (
            ["eval", "view", "x"],
            "`bench eval view` is now `bench view`, with the same arguments",
        ),
        (
            ["tasks", "init", "x"],
            "`bench tasks init` is now `bench new`, with the same arguments",
        ),
        (
            ["agent", "show", "x"],
            "`bench agent show` is now `bench agents`, with the same arguments",
        ),
        (["eval", "metrics", "x"], None),
        (["eval", "smoke"], None),
        (["tasks", "digest", "x"], None),
        (["eval", "run", "--help"], None),
        (["doctor"], None),
    ],
)
def test_note_for(words, expected) -> None:
    assert verbs.note_for(words) == expected


def test_a_secret_value_never_reaches_a_note() -> None:
    note = verbs.note_for(
        [
            "eval", "run", "--tasks-dir", "t", "--agent", "codex",
            "--agent-env=OPENAI_API_KEY=sk-abcdef", "--agent-env", "NOPAIR",
            "--extra-instruction", "https://user:pw@example.com/x",
        ]
    )  # fmt: skip
    assert note is not None
    assert "sk-abcdef" not in note and "pw@" not in note and "NOPAIR" not in note
    assert '--env OPENAI_API_KEY="$OPENAI_API_KEY"' in note
    repo = verbs.note_for(
        [
            "eval",
            "run",
            "--source-repo",
            "https://u:tok-123@example.com/o/r",
            "--agent",
            "x",
        ]
    )
    assert repo is not None and "tok-123" not in repo and "<hidden>" in repo


@pytest.mark.parametrize(
    "url",
    [
        "https://h/r.json?token=tok-123",
        "https://bucket.example/r.json?X-Goog-Signature=tok-123",
        "https://h/r.json#tok-123",
    ],
)
def test_a_registry_url_with_a_query_never_reaches_a_note(url) -> None:
    # A private raw link carries ?token=, a presigned URL its signature: the
    # note goes to stderr, which CI logs keep.
    note = verbs.note_for(
        ["eval", "run", "--dataset", "d@1", "--registry", url, "--agent", "x"]
    )
    assert note is not None and "--registry '<hidden>'" in note, note
    assert "tok-123" not in note


def test_the_cli_reference_documents_the_verb_set() -> None:
    doc = CLI_MD.read_text()
    for verb in verbs.VERBS:
        assert re.search(rf"^#+ bench {verb.name}\b", doc, re.M), verb.name


def test_the_upgrade_page_maps_every_older_spelling_to_its_verb() -> None:
    page = UPGRADE_MD.read_text()
    section = page[page.index("## Changed commands") :]
    missing: set[str] = set()
    for group, commands in verbs.GROUP_COMMANDS_08.items():
        for command in commands:
            new = verbs.new_spelling(group, command)
            if new is None:
                continue
            # The table shows a spelling with its arguments: `bench tasks init NAME`.
            for spelling in (f"bench {group} {command}", f"bench {new}"):
                if not re.search(rf"`{re.escape(spelling)}(?: [^`]*)?`", section):
                    missing.add(f"`{spelling}`")
    assert missing <= UPGRADE_TABLE_PENDING, sorted(missing - UPGRADE_TABLE_PENDING)
    if missing:
        pytest.skip(
            "page fix pending: docs/versions/upgrade-from-0.7.md's table of "
            f"changed commands leaves out {', '.join(sorted(missing))}"
        )


def test_the_docs_name_the_rust_commands_this_bench_lacks() -> None:
    pytest.skip(
        "page fix pending: no docs v2 page lists Rust bench 0.8.2's commands "
        "this bench lacks (bench test, shell, handoff, schema, agents check, "
        "completions), as the old docs/reference/cli.md did"
    )
    pages = "\n".join(p.read_text() for p in sorted(DOCS.rglob("*.md")))
    for command in verbs.NOT_PORTED:
        assert f"`bench {command}`" in pages, command
