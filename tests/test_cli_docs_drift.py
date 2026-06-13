"""Guard CLI/docs drift — every public flag documented in docs/reference/cli.md
must still be present in `bench --help` output.

This is the snapshot half of issue #367: docs and CLI help drifted apart so
worked-examples no longer worked. The test pins documented flags against the
live Typer parser so doc rot is caught in CI.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _help(args: list[str]) -> str:
    result = runner.invoke(app, [*args, "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    return _ANSI_RE.sub("", result.output)


def test_top_level_help_lists_public_groups() -> None:
    """Every public top-level group documented in cli.md is shown in --help."""
    out = _help([])
    for group in ("eval", "skills", "tasks", "compat", "agent", "environment"):
        assert group in out, f"missing public group {group!r} in: {out}"
    # Deprecated, hidden, and removed commands must not show up in public help.
    for hidden in ("run", "job", "agents", "metrics", "view", "eval-batch"):
        assert hidden not in out.split("Commands")[-1].split("─")[0], (
            f"hidden command {hidden!r} unexpectedly shown: {out}"
        )


def test_eval_create_help_lists_all_documented_flags() -> None:
    """docs/reference/cli.md's bench eval create flag table must stay in sync.

    Typer truncates long flag names with '…' in --help, so we match against a
    prefix of each documented flag — long enough to disambiguate from
    sibling flags but short enough to survive Click's right-column truncation.
    """
    out = _help(["eval", "create"])
    documented_flag_prefixes = [
        "--config",
        "--tasks-dir",
        "--source-repo",
        "--source-path",
        "--source-ref",
        "--source-env",
        "--source-env-version",
        "--source-env-arg",
        "--source-env-num-examp",
        "--source-env-rollouts-",
        "--source-env-max-tokens",
        "--source-env-temperatu",
        "--source-env-sampling-",
        "--agent",
        "--model",
        "--sandbox",
        "--environment-manifest",
        "--prompt",
        "--concurrency",
        "--agent-idle-timeout",
        "--jobs-dir",
        "--sandbox-user",
        "--sandbox-setup-timeout",
        "--skills-dir",
        "--skill-mode",
        "--skill-creator-dir",
        "--self-gen-no-internet",
        "--agent-env",
        "--include",
        "--exclude",
        "--json",
    ]
    for flag in documented_flag_prefixes:
        assert flag in out, (
            f"documented flag prefix {flag!r} missing from `bench eval create --help`: {out}"
        )


def test_eval_create_accepts_environment_manifest() -> None:
    """`bench eval create --environment-manifest` is the batch seam for
    Environment-plane rollouts (#398). Guard against silent removal so the
    docs and CLI stay in sync."""
    out = _help(["eval", "create"])
    assert "--environment-manifest" in out


def test_documented_subcommands_exist() -> None:
    """Subcommands referenced in docs/reference/cli.md must resolve."""
    for cmd in (
        ["eval", "create"],
        ["eval", "list"],
        ["agent", "list"],
        ["agent", "show"],
        ["tasks", "init"],
        ["tasks", "check"],
        ["tasks", "generate"],
        ["tasks", "list-sources"],
        ["skills", "list"],
        ["skills", "eval"],
        ["environment", "create"],
        ["environment", "list"],
        ["environment", "show"],
        ["environment", "inspect"],
        ["environment", "cleanup"],
        ["compat", "harbor-registry"],
    ):
        out = _help(cmd)
        assert "Usage:" in out, f"bench {' '.join(cmd)} --help failed: {out}"
