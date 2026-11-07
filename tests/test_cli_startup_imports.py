"""``bench --version``, ``bench --help`` and ``bench agent list`` stay light.

Each command runs in a fresh interpreter under ``python -X importtime``, the
way the ``bench`` console script starts, and the test reads every module the
run imported. None of the SDK's heavy parts may load: rollouts, sandboxes, the
review stack, the model proxy (``benchflow.providers``) or LiteLLM. Before the
CLI's imports became lazy, ``bench --version`` imported all of them (about a
second on a build VM) to print one line.

A branch that adds a command or a flag keeps this green by importing what the
command needs inside the command, not at the top of its module.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The console script's own body (``bench = "benchflow.cli.main:main"``).
_LAUNCHER = (
    "import sys; from benchflow.cli.main import main; "
    "sys.argv[0] = 'bench'; sys.exit(main())"
)
_FORBIDDEN = (
    "benchflow.rollout",
    "benchflow.sandbox",
    "benchflow.review",
    "benchflow.providers",
    "litellm",
)


def _imported_modules(args: list[str], tmp_path: Path) -> list[str]:
    """Every module ``bench <args>`` imports, in import order."""
    catalog = tmp_path / "agents"
    catalog.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        # An empty local agent catalog: `agent list` reads it without the
        # network (by default it fetches the benchflow-ai/agents catalog).
        "BENCHFLOW_AGENTS_SOURCE": str(catalog),
    }
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", _LAUNCHER, *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    modules = []
    for line in proc.stderr.splitlines():
        # "import time: <self us> | <cumulative us> | <indent><module>"
        if line.startswith("import time:") and line.count("|") == 2:
            name = line.rsplit("|", 1)[1].strip()
            if name != "imported package":
                modules.append(name)
    assert "typer" in modules, "importtime output not parsed:\n" + proc.stderr[-2000:]
    return modules


def _forbidden(modules: list[str]) -> list[str]:
    return [
        m
        for m in modules
        if any(m == prefix or m.startswith(prefix + ".") for prefix in _FORBIDDEN)
    ]


@pytest.mark.parametrize(
    "args",
    [["--version"], ["--help"], ["agent", "list"], ["cache", "--help"]],
    ids=["version", "help", "agent-list", "cache-help"],
)
def test_startup_commands_load_no_heavy_modules(args, tmp_path):
    heavy = _forbidden(_imported_modules(args, tmp_path))
    assert heavy == [], (
        f"bench {' '.join(args)} imported {len(heavy)} heavy modules, first "
        f"{heavy[:8]}; import them inside the command that needs them"
    )


def test_version_loads_no_command_module(tmp_path):
    """``--version`` imports only ``cli.main`` and the helpers its options need."""
    modules = _imported_modules(["--version"], tmp_path)
    loaded = {m for m in modules if m.startswith("benchflow.cli.")}
    assert loaded <= {
        "benchflow.cli._errors",
        "benchflow.cli._lazy",
        "benchflow.cli._options",
        "benchflow.cli.main",
        "benchflow.cli.reviewer_options",
    }, sorted(loaded)


def test_guard_matches_whole_module_names():
    """A heavy module or any of its submodules matches; a lookalike does not."""
    modules = ["typer", "benchflow.cli.main", "benchflow.sandbox", "litellm.utils"]
    assert _forbidden(modules) == [
        "benchflow.sandbox",
        "litellm.utils",
    ]
    assert _forbidden(["benchflow.sandboxes", "litellmx"]) == []
