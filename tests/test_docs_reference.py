"""The generated reference pages match the code.

``docs/reference/commands.md``, ``python.md``, ``cli-python.md``,
``files.md`` and ``docs/versions/changelog.md`` are written by
``tools/docs_reference.py`` from the live CLI app (and the 0.9.1 command
inventory in ``tools/cli_baseline_0.9.1.json``), ``benchflow.__all__``,
``benchflow.cli_parity``, the task and run config loaders, the schema files
and ``CHANGELOG.md``. A branch that changes a flag, a docstring, a field or a
schema regenerates them with the command the tool documents, and the diff
shows in review; a hand edit of a generated page fails here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tools.docs_reference import (
    BASELINE,
    CHANGELOG,
    NEW_LABEL,
    NEXT_LABEL,
    PAGES,
    PRE_BASELINE,
    REGENERATE,
    _undash,
    diff,
    render,
)


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return render()


def _section(page: str, heading: str) -> str:
    """The text under ``heading`` up to the next heading of its level."""
    level = heading.split(" ", 1)[0]
    start = page.index(f"\n{heading}\n")
    end = page.find(f"\n{level} ", start + len(heading) + 2)
    return page[start : end if end != -1 else len(page)]


@pytest.mark.parametrize("name", sorted(PAGES))
def test_generated_page_is_current(generated: dict[str, str], name: str) -> None:
    expected = PAGES[name].read_text(encoding="utf-8")
    assert generated[name] == expected, (
        f"{PAGES[name].name} is stale; regenerate with: {REGENERATE}\n"
        + diff(name, expected, generated[name])
    )


def test_command_reference_covers_the_commands_the_journeys_use(
    generated: dict[str, str],
) -> None:
    page = generated["commands"]
    for heading in (
        "## bench run",
        "## bench doctor",
        "## bench new",
        "## bench check",
        "## bench results",
        "## bench train convert",
        "## bench train validate",
        "## bench sandbox list",
    ):
        assert f"\n{heading}\n" in page, heading
    # Hidden commands (deprecated aliases kept for old scripts, internal
    # groups) get no section of their own: the last section lists them.
    assert "\n## bench eval create\n" not in page
    assert "\n## bench embodied\n" not in page
    hidden = _section(page, "## Commands hidden from `bench --help`")
    for command in ("bench eval create", "bench embodied", "bench environment"):
        assert f"| `{command}` |" in hidden, command


def test_command_reference_labels_what_0_9_1_lacked(generated: dict[str, str]) -> None:
    """A command or flag absent from the 0.9.1 inventory is labelled; one
    0.9.1 had, hidden or not, is not."""
    page = generated["commands"]
    label = f" ({NEW_LABEL})"
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert "bench preferences" not in baseline and "bench assess" not in baseline
    assert f"\n## bench preferences{label}\n" in page
    assert f"\n## bench sandbox attach{label}\n" in page
    assert "\n## bench sandbox list\n" in page
    run = _section(page, "## bench run")
    assert f"| `--keep-sandbox`{label} |" in run
    assert f"| `--accept-deviations`{label} |" in run
    # Hidden in 0.9.1's bench run, so not new.
    assert "--trials" in baseline["bench run"]
    for flag in ("--trials", "--integrity", "--max-cost-usd", "--max-tokens"):
        assert f"| `{flag}` |" in run, flag
    # Every label names something the baseline lacks.
    for line in page.splitlines():
        m = re.match(
            r"^## (bench(?: [\w-]+)*) \(" + re.escape(NEW_LABEL) + r"\)$", line
        )
        if m:
            assert m.group(1) not in baseline, m.group(1)


def test_command_reference_labels_what_no_release_has(
    generated: dict[str, str],
) -> None:
    """A command 0.10.0rc1 has is labelled (0.10); one it lacks too is
    labelled (unreleased), so the page never says a release has it."""
    page = generated["commands"]
    prerelease = json.loads(PRE_BASELINE.read_text(encoding="utf-8"))
    assert "bench preferences" in prerelease and "bench hub push" not in prerelease
    assert f"\n## bench preferences ({NEW_LABEL})\n" in page
    assert f"\n## bench hub push ({NEXT_LABEL})\n" in page
    for line in page.splitlines():
        m = re.match(r"^## (bench(?: [\w-]+)*) \((.+)\)$", line)
        if m:
            assert m.group(2) in (NEW_LABEL, NEXT_LABEL), line
            assert (m.group(1) in prerelease) == (m.group(2) == NEW_LABEL), line


def test_command_reference_lists_what_help_all_adds(generated: dict[str, str]) -> None:
    """bench run's hidden flags (``--help-all`` shows them) are in a table of
    their own, after the ones ``--help`` shows."""
    run = _section(generated["commands"], "## bench run")
    shown, _, more = run.partition("**More options (shown by `--help-all`)**")
    assert more, "bench run has no table of the flags --help-all adds"
    assert "| `--help-all` |" in shown
    for flag in ("--tasks-dir", "--source-ref", "--retry-attempts"):
        assert f"| `{flag}`" in more and f"| `{flag}`" not in shown, flag


def test_python_reference_lists_the_run_entry_points(generated: dict[str, str]) -> None:
    page = generated["python"]
    for name in ("run_sync", "arun", "RolloutConfig", "resolve_source"):
        assert f"\n### `{name}`\n" in page, name


def test_files_reference_has_every_task_md_and_run_config_key(
    generated: dict[str, str],
) -> None:
    from benchflow.evaluation import RUN_CONFIG_KEYS
    from benchflow.task._document_parse import _DOCUMENT_ONLY_FRONTMATTER_KEYS
    from benchflow.task.config import TaskConfig

    page = generated["files"]
    task_md = _section(page, "### task.md")
    keys = [
        field.serialization_alias or name
        for name, field in TaskConfig.model_fields.items()
    ]

    # A key 0.9.1 lacked has "(0.10)" or "(unreleased)" after it.
    def row(key: str) -> str:
        return rf"\n\| `{re.escape(key)}`(?: \((?:0\.10|unreleased)\))? \|"

    for key in [*keys, *sorted(_DOCUMENT_ONLY_FRONTMATTER_KEYS)]:
        assert re.search(row(key), task_md), key
    run_config = _section(page, "### Run config")
    for key in sorted(RUN_CONFIG_KEYS):
        assert re.search(row(key), run_config), key


def test_changelog_page_is_the_changelog(generated: dict[str, str]) -> None:
    """The page is CHANGELOG.md under its own title, escaped only where the
    site's MDX build would read text as markup, with a link written from the
    repository root rewritten from the page's folder: a link to an old docs
    page goes to the page docs/redirects.json moved it to, at that entry's
    anchor, and a link to a file the repository doesn't have becomes its
    text. Its prose is set in the docs' punctuation: no em dashes, and an
    ellipsis for three dots."""
    page = generated["changelog"]
    assert page.startswith("# Changelog\n\n")
    # Title, one line of intro, the maintainers' comment, then the changelog.
    body = page.split("\n\n", 3)[3]
    unescaped = (
        body.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#123;", "{")
        .replace("&#125;", "}")
    )
    folder = PAGES["changelog"].parent
    repo = CHANGELOG.parent
    redirects = json.loads((repo / "docs" / "redirects.json").read_text())

    def from_root(m: re.Match[str]) -> str:
        text, target, fragment = m.group("text"), m.group("target"), m.group("anchor")
        if text is None or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            return m.group(0)  # a code span, or a URL
        if not (repo / target).exists():
            moved = None
            if target.startswith("docs/") and target.endswith(".md"):
                old = target.removeprefix("docs/").removesuffix(".md")
                moved = redirects["redirects"].get(old)
            if moved is None:
                return text[1:-1]
            new, _, anchor = moved.removeprefix("github:").partition("#")
            target, fragment = f"docs/{new}.md", f"#{anchor}" if anchor else ""
        relative = Path(os.path.relpath(repo / target, folder)).as_posix()
        return f"{text}({relative}{fragment or ''})"

    source = CHANGELOG.read_text(encoding="utf-8").split("\n", 1)[1].strip("\n")
    expected = re.sub(
        r"`+[^`]*?`+"
        r"|(?P<text>\[(?:[^\[\]`]|`[^`]*`)*\])"
        r"\((?P<target>[^)\s#]+)(?P<anchor>#[^)\s]*)?\)",
        from_root,
        source,
    )
    # In the docs' punctuation: no em dashes, an ellipsis for three dots (_undash).
    assert unescaped == _undash(expected) + "\n"
    # The rewritten links all reach a page.
    for target in re.findall(r"\]\((\.\./[^)\s#]+)", unescaped):
        assert (folder / target).exists(), target
