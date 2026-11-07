"""The error-message index matches the code.

``docs/reference/errors.md`` is written by ``tools/docs_errors.py``: the causes
``bench explain`` names, the warn and fail rows of ``bench doctor``, the error
classes that carry a next step, the CLI's errors, the lines that refuse a
task.md or run config value, the R401 refusals, what the verifier step
reports and the warnings a run logs. A branch that changes a message
regenerates the page with ``uv run python tools/docs_errors.py --write``; a
hand edit, or a message added without regenerating, fails here.
"""

from __future__ import annotations

import pytest

from tools.docs_errors import FIXES, PAGE, REGENERATE, _Fixes, render


@pytest.fixture(scope="module")
def generated() -> str:
    return render()


def test_error_index_is_current(generated: str) -> None:
    expected = PAGE.read_text(encoding="utf-8")
    assert generated == expected, (
        f"docs/reference/errors.md is stale; regenerate with: {REGENERATE}"
    )


def test_error_index_lists_every_cause_bench_explain_names(generated: str) -> None:
    from benchflow.cli.explain import pages

    missing = [p.key for p in pages() if f"| `{p.key}` |" not in generated]
    assert not missing, f"causes missing from docs/reference/errors.md: {missing}"


def test_error_index_lists_the_messages_first_run_readers_hit(generated: str) -> None:
    """Messages the cold-start readers met, each findable by its
    words. ``bench run`` with no target now says it needs a target (round 2
    quoted the older "needs a task folder, a task named by repository or
    --dataset")."""
    for words in (
        "no reward file found",
        "bench run needs a target: a task folder, a folder of tasks, or a repository",
        "the task asks for",
        "no model agent can run",
    ):
        assert words in generated, words


def test_error_index_lists_the_steps_the_cli_adds_itself(generated: str) -> None:
    """The next steps that come from no single message: click's usage errors
    (cli/verbs.py), a missing file (cli/_errors.py), and a run config's
    unknown key (benchflow._utils.config_keys)."""
    for words in (
        "says what each flag takes",
        "lists its flags",
        "check the path, and that you can read it",
        "unknown key (did you mean …?)",
    ):
        assert words in generated, words


def test_every_message_has_a_fix() -> None:
    """Each message on the page says what to do: the step the code prints, or
    the fix written in tools/docs_errors_fixes.tsv. In round 2, 40% had none."""
    fixes = _Fixes()
    render(fixes)
    assert not fixes.missing, (
        f"{len(fixes.missing)} messages have no fix; add a line for each to "
        f"{FIXES.name} (section, key, message, fix): {fixes.missing[:5]}"
    )


def test_written_fixes_are_all_used() -> None:
    """A written fix whose message the code no longer prints, or that the code
    now gives itself, is removed rather than kept."""
    fixes = _Fixes()
    render(fixes)
    stale = sorted(set(fixes.table) - fixes.used)
    assert not stale, (
        f"{len(stale)} lines in {FIXES.name} match no message without a fix: {stale[:5]}"
    )
