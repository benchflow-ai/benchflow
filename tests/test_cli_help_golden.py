"""Every ``bench`` command's ``--help`` text is byte-identical to the golden.

The golden (``tests/fixtures/cli_help_golden.txt``) was first made from
``integ/py-0.8`` before the CLI's imports became lazy, so this test is what
shows that lazy registration changed no help text. A branch that adds a flag
or a verb, or changes any help text, regenerates it on a build VM with the
command ``tools/cli_help_golden.py`` documents and shows the diff in review.
"""

from __future__ import annotations

from tools.cli_help_golden import GOLDEN, REGENERATE, diff, render


def test_every_command_help_matches_the_golden():
    expected = GOLDEN.read_text(encoding="utf-8")
    actual = render()
    assert actual == expected, (
        f"--help differs from the golden; regenerate with: {REGENERATE}\n"
        + diff(expected, actual)
    )


def test_golden_lists_the_root_help_first_and_the_core_commands():
    """The golden covers the commands the README teaches, root help first."""
    golden = GOLDEN.read_text(encoding="utf-8")
    assert golden.startswith("==> bench --help <==\n")
    for command in (
        "bench eval run --help",
        "bench eval --help",
        "bench agent list --help",
        "bench doctor --help",
        "bench tasks --help",
    ):
        assert f"==> {command} <==\n" in golden, command
