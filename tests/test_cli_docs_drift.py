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

from benchflow.cli.main import app, eval_app

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
    for group in ("eval", "train", "skills", "tasks", "hub", "agent", "sandbox"):
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


def test_eval_run_flags_match_cli_md_bidirectional() -> None:
    """`bench eval run`'s flags and its cli.md table must be set-equal.

    The old guard only checked doc→CLI (a hand-maintained list of documented
    flags must exist in --help). It could not catch the *reverse* — a new CLI
    flag landing undocumented — which is exactly how ``--loop-strategy`` and
    ``--ignore-bench-version`` rotted out of the docs (#731). Deriving both
    sides from ground truth (the live parser + the doc table) drops the
    hand-maintained list and closes both directions.
    """
    cli = _cli_long_flags(["eval", "run"])
    doc = _doc_flags("### bench eval run")
    assert cli == doc, (
        "bench eval run CLI↔cli.md flag drift:\n"
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


def test_eval_run_accepts_environment_manifest() -> None:
    """`bench eval run --environment-manifest` is the batch seam for
    Environment-plane rollouts (#398). Guard against silent removal so the
    docs and CLI stay in sync."""
    out = _help(["eval", "run"])
    assert "--environment-manifest" in out


def test_eval_create_is_deprecated_alias_of_run() -> None:
    """`bench eval create` was renamed to `bench eval run`; the old name stays
    as a deprecated alias so existing scripts and downstream repos keep working.
    """
    by_name = {c.name: c for c in eval_app.registered_commands}
    assert "run" in by_name, "primary `bench eval run` command missing"
    assert "create" in by_name, "deprecated `bench eval create` alias missing"
    assert by_name["create"].deprecated, "`create` must be marked deprecated"
    assert not by_name["run"].deprecated, "`run` must not be deprecated"
    # Both names share the same callback so flags/behavior never drift apart.
    assert by_name["run"].callback is by_name["create"].callback


def test_documented_subcommands_exist() -> None:
    """Subcommands referenced in docs/reference/cli.md must resolve."""
    for cmd in (
        ["eval", "run"],
        ["eval", "create"],
        ["eval", "list"],
        ["eval", "metrics"],
        ["eval", "view"],
        ["train", "convert"],
        ["train", "validate"],
        ["train", "run"],
        ["train", "run", "sft"],
        # Adoption is canonically under `eval` (cli.md documents `bench eval adopt`).
        ["eval", "adopt", "init"],
        ["eval", "adopt", "convert"],
        ["eval", "adopt", "verify"],
        ["agent", "list"],
        ["agent", "show"],
        ["tasks", "init"],
        ["tasks", "check"],
        ["tasks", "generate"],
        ["tasks", "overlap"],
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
        ["hub", "list"],
        ["hub", "show"],
        ["hub", "inspect"],
        # `hub env *` is a hidden back-compat alias of the flattened verbs.
        ["hub", "env", "list"],
        ["hub", "env", "show"],
        ["hub", "env", "inspect"],
    ):
        out = _help(cmd)
        assert "Usage:" in out, f"bench {' '.join(cmd)} --help failed: {out}"


# ── install-doc guard: no regression to pinned GitHub-release RC wheels ───────

# Install docs that used to pin a concrete RC-wheel URL before 0.6.0 shipped to
# PyPI. They now install from PyPI with an explicit Python 3.12 tool
# interpreter; a hand-pinned `releases/download/<tag>/benchflow-…rcN.whl` goes
# stale the instant a newer release lands, so this guards against re-introducing
# that pattern.
_INSTALL_URL_DOCS = (
    "README.md",
    "docs/getting-started.md",
    "docs/release.md",
    "docs/agent-quickstart.md",
    "docs/skill-eval.md",
    "docs/llm-judge.md",
    "docs/reference/python-api.md",
    "docs/examples/coder-reviewer-demo.py",
    "docs/examples/scene-patterns.ipynb",
    ".claude/skills/benchflow/SKILL.md",
    ".claude/skills/benchflow/tasks/benchflow-knowledge/environment/benchflow/SKILL.md",
    ".claude/skills/benchflow/tasks/create-simple-task/environment/benchflow/SKILL.md",
)
_RC_WHEEL_URL_RE = re.compile(
    r"releases/download/\d+\.\d+\.\d+-rc\.\d+/"
    r"benchflow-\d+\.\d+\.\d+rc\d+-py3-none-any\.whl"
)
_BENCHFLOW_UV_TOOL_INSTALL_RE = re.compile(
    r"uv tool install\b[^\n`]*\bbenchflow(?:\[[^\]\n`]+\])?\b"
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


def test_install_docs_pin_python_312_for_uv_tool_install() -> None:
    """Guards the fix from PR #899 against Python 3.10/3.11 resolver fallback
    to the old no-entrypoint benchflow wheel."""
    offenders: list[tuple[str, str]] = []
    for rel in _INSTALL_URL_DOCS:
        text = (_REPO_ROOT / rel).read_text()
        for match in _BENCHFLOW_UV_TOOL_INSTALL_RE.finditer(text):
            command = match.group(0)
            if "--python 3.12" not in command:
                offenders.append((rel, command))

    assert not offenders, (
        "benchflow CLI install docs must use `uv tool install --python 3.12 ...` "
        f"to avoid old no-entrypoint wheels on Python 3.10/3.11: {offenders}"
    )
