"""Guard CLI/docs drift — every public flag documented in docs/reference/cli.md
must still be present in `bench --help` output.

This is the snapshot half of issue #367: docs and CLI help drifted apart so
worked-examples no longer worked. The test pins documented flags against the
live Typer parser so doc rot is caught in CI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import click
import typer
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLI_MD = _REPO_ROOT / "docs" / "reference" / "cli.md"


def _click_command(path: list[str]) -> click.Command:
    """Resolve a subcommand from the live Typer app (authoritative, untruncated)."""
    cmd = typer.main.get_command(app)
    for seg in path:
        cmd = cast("click.Group", cmd).commands[seg]
    return cmd


def _cli_long_flags(path: list[str]) -> set[str]:
    """Every ``--long`` option the parser actually accepts for ``path`` (minus --help)."""
    flags = {
        opt
        for param in _click_command(path).params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--")
    }
    flags.discard("--help")
    return flags


def _doc_section(header: str) -> str:
    """The cli.md block from ``header`` up to the next ``### `` heading."""
    doc = _CLI_MD.read_text()
    i = doc.index(header)
    nxt = doc.find("\n### ", i + len(header))
    return doc[i : nxt if nxt != -1 else len(doc)]


def _doc_flags(header: str) -> set[str]:
    """Backtick-wrapped ``--flags`` documented under a cli.md heading."""
    return set(re.findall(r"`(--[a-z0-9-]+)`", _doc_section(header)))


def _help(args: list[str]) -> str:
    result = runner.invoke(app, [*args, "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    return _ANSI_RE.sub("", result.output)


def _help_command_names(out: str) -> set[str]:
    """The command names listed in --help, from the panel rows only.

    Rich renders each command as a panel row ``│ <name>   <description>``. Match
    the first token of those rows so the check is robust to (a) the tagline prose
    (which may contain words like "run") and (b) command-panel names (Core /
    Environments / Recovery / the default "Commands").
    """
    names: set[str] = set()
    for line in out.splitlines():
        m = re.match(r"^\s*│\s+([A-Za-z][\w-]*)\s", line)
        if m:
            names.add(m.group(1))
    return names


def test_top_level_help_lists_public_groups() -> None:
    """Every public top-level group documented in cli.md is shown in --help."""
    out = _help([])
    commands = _help_command_names(out)
    for group in ("eval", "skills", "tasks", "hub", "agent", "sandbox"):
        assert group in commands, f"missing public group {group!r} in: {out}"
    # Deprecated, hidden, and removed commands must not show up in public help.
    # `environment` is now a hidden deprecated alias group (→ sandbox / hub env);
    # `adopt` is a hidden deprecated alias group (→ eval adopt).
    for hidden in (
        "run",
        "job",
        "agents",
        "metrics",
        "view",
        "eval-batch",
        "environment",
        "adopt",
    ):
        assert hidden not in commands, (
            f"hidden command {hidden!r} unexpectedly shown: {out}"
        )


def test_eval_create_flags_match_cli_md_bidirectional() -> None:
    """`bench eval create`'s flags and its cli.md table must be set-equal.

    The old guard only checked doc→CLI (a hand-maintained list of documented
    flags must exist in --help). It could not catch the *reverse* — a new CLI
    flag landing undocumented — which is exactly how ``--loop-strategy`` and
    ``--ignore-bench-version`` rotted out of the docs (#731). Deriving both
    sides from ground truth (the live parser + the doc table) drops the
    hand-maintained list and closes both directions.
    """
    cli = _cli_long_flags(["eval", "create"])
    doc = _doc_flags("### bench eval create")
    assert cli == doc, (
        "bench eval create CLI↔cli.md flag drift:\n"
        f"  in CLI but UNDOCUMENTED: {sorted(cli - doc)}\n"
        f"  documented but NOT in CLI: {sorted(doc - cli)}"
    )


def test_documented_defaults_match_cli() -> None:
    """Documented default *values* must match the live param defaults.

    The name-only guard happily passed while ``bench hub check --cache-dir``
    documented the pre-rename ``.cache/compat/harbor`` (the CLI moved to
    ``.cache/hub/harbor``). Pin the defaults that have drift history so a
    stale value in either the code or the doc fails CI.
    """
    checks = [
        (["hub", "check"], "--cache-dir", "### bench hub check", ".cache/hub/harbor"),
    ]
    for path, flag, header, expected in checks:
        param = next(
            p for p in _click_command(path).params if flag in getattr(p, "opts", [])
        )
        assert expected in str(param.default), (
            f"`bench {' '.join(path)} {flag}` default is {param.default!r}, "
            f"expected to contain {expected!r}"
        )
        assert expected in _doc_section(header), (
            f"cli.md {header!r} no longer documents the {flag} default {expected!r}"
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
        ["eval", "metrics"],
        ["eval", "view"],
        # Adoption is canonically under `eval` (cli.md documents `bench eval adopt`).
        ["eval", "adopt", "init"],
        ["eval", "adopt", "convert"],
        ["eval", "adopt", "verify"],
        ["agent", "list"],
        ["agent", "show"],
        ["tasks", "init"],
        ["tasks", "check"],
        ["tasks", "generate"],
        ["tasks", "list-sources"],
        ["skills", "list"],
        ["skills", "eval"],
        ["sandbox", "create"],
        ["sandbox", "list"],
        ["sandbox", "cleanup"],
        # `environment` is now a hidden deprecated alias group; its commands
        # still resolve for back-compat.
        ["environment", "create"],
        ["environment", "list"],
        ["environment", "show"],
        ["environment", "inspect"],
        ["environment", "cleanup"],
        ["hub", "check"],
        ["hub", "env", "list"],
        ["hub", "env", "show"],
        ["hub", "env", "inspect"],
    ):
        out = _help(cmd)
        assert "Usage:" in out, f"bench {' '.join(cmd)} --help failed: {out}"


# ── install-doc guard: no regression to pinned GitHub-release RC wheels ───────

# Install docs that used to pin a concrete RC-wheel URL before 0.6.0 shipped to
# PyPI. They now install from PyPI (`uv tool install … benchflow`); a hand-pinned
# `releases/download/<tag>/benchflow-…rcN.whl` goes stale the instant a newer
# release lands, so this guards against re-introducing that pattern.
_INSTALL_URL_DOCS = (
    "README.md",
    "docs/getting-started.md",
    "docs/release.md",
    "docs/agent-quickstart.md",
    "docs/skill-eval.md",
    ".claude/skills/benchflow/SKILL.md",
)
_RC_WHEEL_URL_RE = re.compile(
    r"releases/download/\d+\.\d+\.\d+-rc\.\d+/"
    r"benchflow-\d+\.\d+\.\d+rc\d+-py3-none-any\.whl"
)


def test_install_docs_use_pypi_not_pinned_rc_wheel() -> None:
    """Install docs must install benchflow from PyPI, not a pinned GitHub-release
    RC wheel (which goes stale on every release). Catches regressions to the
    pre-0.6.0 `releases/download/…rcN.whl` pattern."""
    offenders = [
        rel
        for rel in _INSTALL_URL_DOCS
        if _RC_WHEEL_URL_RE.search((_REPO_ROOT / rel).read_text())
    ]
    assert not offenders, (
        "install docs pin a stale GitHub-release RC wheel instead of installing "
        f"from PyPI: {offenders}"
    )
