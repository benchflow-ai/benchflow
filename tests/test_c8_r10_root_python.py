"""C8 round 10: root's Python helpers through one entry point.

PT-C8-R8-2, the b114 review's PT-C8-R9-8 /
PT-C8-R9-6: every stdlib helper BenchFlow runs as root, or verifier-side,
starts through :mod:`benchflow._root_python`: the protected kit's
interpreter, ``-I -S -B``, from ``/``, with no ``PYTHON*`` value; a kit with
no Python refuses. The source scan holds every other Python launch in
BenchFlow to a listed reason.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow._runtime_tree import KIT_PYTHON, RuntimeKit

SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"

# A Python started from shell text: ``python``/``python3``/``python3.12``
# (bare, by path, or a ``$var`` holding one) with ``-c``, ``-m``, a ``-``
# script on stdin or a heredoc.
_LAUNCH = re.compile(
    r"python3?(?:\.\d+)?['\"]?\s+(?:-[A-Za-z]+\s+)*(?:-c\b|-m\b|-\s|-$|<<)",
    re.MULTILINE,
)
_ARGV0 = re.compile(r"(?:^|/)python3?(?:\.\d+)?$")
# The same held in a shell variable (PT-C8-CONF-2: the egress proxy's
# ``"$PY" -c`` and ``"$PY" proxy.py`` passed the scan): ``"$var"`` named for
# a Python, or given Python's own -I/-S/-B, with ``-c``, ``-m``, a ``.py``
# script or an interpolated one (an f-string's value reads as ``{}``).
_VAR_LAUNCH = re.compile(
    r'"(?:\$\{?(?P<var>\w+)\}?|\$\((?P<sub>command -v python[^)]*)\))"\s+'
    r"(?P<flags>(?:-[A-Za-z]+\s+)*)"
    r"(?:-c\b|-m\b|[^\s'\"]*\.py\b|\{\})",
    re.MULTILINE,
)


def _launches(text: str) -> bool:
    if _LAUNCH.search(text):
        return True
    for match in _VAR_LAUNCH.finditer(text):
        if re.search(r"py", match["var"] or match["sub"], re.IGNORECASE) or re.search(
            r"-[A-Za-z]*[ISB]", match["flags"]
        ):
            return True
    return False


def _rendered(node: ast.JoinedStr) -> str:
    return "".join(
        part.value
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
        else "{}"
        for part in node.values
    )


# Every Python launch outside the entry point, with why it is not root's
# stdlib helper on the protected interpreter. A new one fails the scan.
NOT_ROOT_HELPERS = {
    # The kit's own interpreter, smoke-tested while the kit is built (before
    # the agent's turn).
    ("_runtime_tree.py", "kit_prepare_command"): "the kit's build",
    # BenchFlow's own protected runtimes under /opt/benchflow (their venvs,
    # custody-checked in the same command), not stdlib helpers.
    ("providers/litellm_runtime.py", "_ensure_sandbox_litellm"): "proxy venv",
    ("providers/litellm_runtime.py", "proxy_layer_recipe"): "proxy venv build",
    ("providers/litellm_runtime.py", "sandbox_litellm_install_command"): "proxy venv",
    ("sandbox/daytona_layers.py", "layer_run_line"): "layer build",
    # A build step of the AgentCore image (exec-form RUN, the image's user,
    # before any session): the check for bash's start-up names in its ENV.
    ("sandbox/agentcore_image.py", "_STARTUP_NAMES_CHECK"): "image build check",
    ("agents/registry.py", "<module>"): "agent venvs (harvey-lab, deepagents) by path",
    ("acp/runtime.py", "_prepare_openhands_direct_execution"): (
        "OpenHands' own interpreter in /opt/benchflow/uv-tools, custody-checked"
    ),
    # Root's interpreter picked in the sandbox as root_python_sh picks it:
    # the kit's Python, the image's only where no kit exists (PT-C8-CONF-2).
    ("sandbox/egress_denylist.py", "_setup_cmd"): "root_python_pick_sh (the proxy)",
    ("sandbox/_hardening_python.py", "hardening_python_cmd"): "the kit's or judged",
    ("sandbox/_verifier_runtime.py", "probe_interpreter"): "the kit's; image without",
    # The solver's own launch (the oracle joining its group), not after it.
    ("sandbox/oracle_egress.py", "join_group_prefix"): "the oracle's launch",
    # The pytest guard installer (PT-C8-CONF-2b): the image's interpreters
    # as root only where no kit exists (no sandbox user); with a kit it
    # starts none (_INSTALL_PYTEST_PLUGIN_GUARD_KIT_CMD reads the record
    # taken before the agent's turn, and is not listed: it has no launch).
    ("sandbox/lockdown.py", "_INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE"): (
        "no kit: no sandbox user"
    ),
    # Not root: the home's owner, the agent's own process or MCP server.
    ("agents/registry.py", "_json_settings_merge_py"): "runs as the home's owner",
    ("agents/registry.py", "_python_shim_launch"): "the agent's own isolated launch",
    ("experimental/mcp/hooks.py", "mcp_reviewer_hook"): "the agent's MCP server",
    ("sandbox/recovery_network.py", "prepare_recovery_network"): "as the sandbox user",
    # Before any sandbox user exists (no kit yet): installing and probing
    # the capture Python is the image's own business.
    ("review/evidence_runtime.py", "_SETUP_SCRIPT"): "before the kit exists",
    ("review/evidence_runtime.py", "_CAPTURE_TOOLS_PROBE"): "before the kit exists",
    ("review/evidence_runtime.py", "ensure_evidence_python"): "no kit: no sandbox user",
    # The hook clean-up keeps its original search path after PT's protected
    # pass; its startup imports are the U5a-CONF-2 release
    # blocker, not a protected helper.
    # Image build time (a Dockerfile RUN), not a sandbox.
    ("sandbox/setup.py", "_modal_heredoc_decoder"): "image build",
    # Task content BenchFlow writes (a task's own commands), not its helpers.
    ("review/wrapper.py", "assemble_review_task"): "the review task's own setup",
    ("traces/task_gen.py", "_build_solution_sh"): "generated task content",
    ("traces/task_gen.py", "_build_test_sh"): "generated task content",
    ("adapters/*", "*"): "benchmark adapters' task content",
    ("_utils/task_authoring/structural_checks.py", "workspace_plugin_installs"): (
        "describes a task's own commands"
    ),
    # Usage text for BenchFlow's own host-side modules.
    ("_utils/build_context_stage.py", "main"): "host usage text",
    ("agent_router_guide.py", "CONVERSION_GUIDE"): "host usage text",
    ("cli_parity.py", "markdown"): "host usage text",
    ("embodied/serve.py", "main"): "host usage text",
    ("eval_worker.py", "main"): "host usage text",
    ("trajectories/viewer/__main__.py", "<module>"): "host usage text",
}


def _launch_sites():
    """(module, outermost function or ``<module>``) holding a Python launch
    in a string constant (docstrings aside)."""
    sites = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        if rel == "_root_python.py":
            continue
        tree = ast.parse(path.read_text())
        docs = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }

        def visit(node, owner, rel=rel, docs=docs):
            for child in ast.iter_child_nodes(node):
                name = owner
                if owner == "<module>" and isinstance(
                    child, ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    name = child.name
                # A module-level recipe is owned by its constant's name.
                if (
                    owner == "<module>"
                    and isinstance(child, ast.Assign)
                    and len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Name)
                ):
                    name = child.targets[0].id
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and id(child) not in docs
                    and _launches(child.value)
                ) or (isinstance(child, ast.JoinedStr) and _launches(_rendered(child))):
                    sites.add((rel, name))
                # An argv list: ["python3", "-I", "-c", ...].
                if isinstance(child, ast.List | ast.Tuple) and child.elts:
                    words = [
                        e.value
                        for e in child.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
                    first = child.elts[0]
                    if (
                        isinstance(first, ast.Constant)
                        and isinstance(first.value, str)
                        and _ARGV0.search(first.value)
                        and any(w in ("-c", "-m", "-") for w in words)
                    ):
                        sites.add((rel, name))
                visit(child, name)

        visit(tree, "<module>")
    return sites


def _allowed(rel: str, owner: str) -> bool:
    for (path, function), _ in NOT_ROOT_HELPERS.items():
        if (path == rel or (path.endswith("/*") and rel.startswith(path[:-1]))) and (
            function in ("*", owner)
        ):
            return True
    return False


def test_every_root_python_launch_goes_through_the_entry_point():
    """The scan: a Python launch anywhere in BenchFlow outside
    ``_root_python.py`` is a listed exception or fails here."""
    sites = _launch_sites()
    unlisted = sorted(s for s in sites if not _allowed(*s))
    assert not unlisted, (
        "root Python outside benchflow._root_python (use root_python_command "
        f"or root_python_sh, or list why it is not root's helper): {unlisted}"
    )
    stale = sorted(
        key for key in NOT_ROOT_HELPERS if "*" not in "".join(key) and key not in sites
    )
    assert not stale, f"listed launches that are gone: {stale}"


# --- The entry point's rule ------------------------------------------------


def test_the_protected_interpreter_and_its_flags():
    from benchflow._root_python import root_python_argv

    assert root_python_argv(None) == ["python3", "-I", "-S", "-B"]
    kit = RuntimeKit(trusted=("/usr/bin",), python=True, bash=False)
    assert root_python_argv(kit) == [KIT_PYTHON, "-I", "-S", "-B"]
    assert root_python_argv(RuntimeKit(trusted=(), python=False, bash=False)) is None


@pytest.mark.asyncio
async def test_a_kit_without_python_refuses_the_step():
    from types import SimpleNamespace

    from benchflow._root_python import NO_KIT_PYTHON, root_python_command

    no_python = RuntimeKit(trusted=("/usr/bin",), python=False, bash=False)

    async def kit():
        return no_python

    env = SimpleNamespace(runtime_kit_for_route=kit)
    assert await root_python_command(env, "print(1)") == NO_KIT_PYTHON
    done = subprocess.run(
        ["/bin/sh", "-c", NO_KIT_PYTHON], capture_output=True, text=True
    )
    assert done.returncode == 127 and "python3 not found" in done.stderr


def _run_helper(
    command: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    # The test's interpreter stands in for python3 on the route.
    shim = cwd.parent / "shim"
    shim.mkdir(exist_ok=True)
    py = shim / "python3"
    py.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    return subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=cwd,
        env={**env, "PATH": f"{shim}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


_PROBE = (
    "import json, os, sys\n"
    "print(json.dumps([os.getcwd(), sys.flags.isolated, sys.flags.no_site, "
    "sys.dont_write_bytecode, sorted(k for k in os.environ if k.startswith('PYTHON'))]))\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["command", "sh"])
async def test_a_root_helper_imports_nothing_from_a_workspace(tmp_path, form):
    """Run where an agent's ``json.py`` shadows the stdlib, with PYTHONPATH,
    PYTHONSTARTUP and PYTHONHOME pointing into the workspace: the helper
    imports the real stdlib, starts from /, isolated, without site or
    bytecode, and hands no PYTHON* value on."""
    from benchflow._root_python import root_python_command, root_python_sh

    work = tmp_path / "work"
    work.mkdir()
    mark = tmp_path / "imported"
    (work / "json.py").write_text(f"open({str(mark)!r}, 'w').close()\n")
    (work / "startup.py").write_text(f"open({str(mark)!r}, 'w').close()\n")
    if form == "command":
        command = await root_python_command(object(), _PROBE)
    else:
        command = root_python_sh(_PROBE)
    env = {"PYTHONPATH": str(work), "PYTHONSTARTUP": str(work / "startup.py")}
    done = _run_helper(command, work, env)
    assert done.returncode == 0, done.stderr
    import json

    cwd, isolated, no_site, no_bytecode, names = json.loads(done.stdout)
    assert (cwd, isolated, no_site, no_bytecode, names) == ("/", 1, 1, True, [])
    assert not mark.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="the record file stands for root's kit")
def test_the_static_form_refuses_a_kit_without_python(tmp_path, monkeypatch):
    """``root_python_sh`` decides in the sandbox: a RECORD with no kit
    Python refuses (127) instead of falling back to the image's python3."""
    import benchflow._root_python as rp

    record = tmp_path / "RECORD"
    record.write_text("kit=1\npython=0\n")
    monkeypatch.setattr(rp, "KIT_RECORD", str(record))
    monkeypatch.setattr(rp, "KIT_PYTHON", str(tmp_path / "py" / "python3"))
    done = subprocess.run(
        ["/bin/sh", "-c", rp.root_python_sh("print(1)")], capture_output=True, text=True
    )
    assert done.returncode == 127 and "python3 not found" in done.stderr
