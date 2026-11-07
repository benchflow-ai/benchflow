"""C8 round 8: the invariants that hold by construction.

1. One exec choke point: every container exec a backend builds takes its
   program from :meth:`BaseSandbox.exec_route` (a source scan fails on any
   new builder that does not).
2. The route builds root's environment once: no caller env and no root
   helper's text puts PATH, a loader, start-up or Python setting back.
3. Identity per service: a name is looked up in the container that runs the
   command, through its own exec; unknown stays root.

And the round's findings that are argv or text, checked on what BenchFlow
builds (the container tests in ``test_c8_protected_runtime_container.py``
run the rest for real).
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import ExecResult

SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"

# Functions that build a container exec argv (a literal "exec"), each with
# why it is on the route. A new one fails the scan until it takes its shell
# from a route and is listed here.
EXEC_BUILDERS = {
    "sandbox/docker.py": {
        "route_exec_argv": "builds Docker argv from a route only",
        "DockerSandbox.exec": "the backend's exec: exec_route, then route.shell",
        "_DockerCredentialOps.run_as": "root: route_exec_argv; another uid: no root",
        # PERF-B: no shell. The kit's own Python (exec_route's kit route,
        # its loader line read by route_exec_argv); every command it starts
        # is exec()'s route argv (tests/test_exec_channel.py).
        "DockerSandbox._start_exec_channel": "the command channel: the kit route",
        # TRUSTED-RUNTIME v2 I1(a): no shell and no PATH. bfx by absolute path
        # from the read-only /.bfx is the exec's first program.
        "DockerSandbox.bfx_call": "bfx by absolute path, argv only, no shell",
        "DockerSandbox.bfx_tar_out": "bfx by absolute path, argv only, no shell",
        "DockerSandbox.bfx_session": "bfx by absolute path, argv only, no shell",
    },
    "sandbox/apple_container.py": {
        "AppleContainerSandbox.exec": "the backend's exec: route.shell",
        "AppleContainerSandbox.start": "start probe: the system shell's builtin",
        "AppleContainerSandbox._refuse_startup_names": (
            "start probe: /bin/cat by absolute path, argv only, no shell"
        ),
    },
    "sandbox/daytona_dind.py": {
        "_DaytonaDinD._exec": "the strategy's exec: route required, route.shell",
        "_DaytonaDinD._wait_for_main_container": "start probe: system shell builtin",
    },
    "sandbox/process/docker.py": {
        "DockerProcess.start": "launch: the route's bash and PATH",
    },
    "sandbox/process/apple.py": {
        "AppleContainerProcess._write_env_to_container": "the launch route's shell",
        "AppleContainerProcess._remove_env_from_container": "the launch route's shell",
        "AppleContainerProcess.start": "launch: the route's bash and PATH",
    },
    "sandbox/process/daytona.py": {
        "DaytonaProcess.start": "launch: the route's bash and PATH",
        "DaytonaPtyProcess.start": "launch: the route's bash and PATH",
    },
    "sandbox/lockdown.py": {
        "_heredoc_is_shell_script": "compares, builds none",
    },
    "sandbox/remote_docker.py": {
        "RemoteDockerSandbox._run_docker_compose_command": "compares, builds none",
        "RemoteDockerSandbox._run_docker_container_command": "compares, builds none",
        "RemoteDockerSandbox._docker_cli": "compares, builds none",
    },
}
# Start probes run a builtin in the system shell by path before any agent
# exists; every other builder must name a route.
START_PROBES = {
    "AppleContainerSandbox.start",
    "_DaytonaDinD._wait_for_main_container",
}
# Start probes that run no shell at all: STARTUP_ENV_ARGV by absolute path,
# before any agent, to read the image's start environment (startup_name).
ARGV_PROBES = {"AppleContainerSandbox._refuse_startup_names"}
# Builds a fixed shell only for a non-root uid (the user's own process).
NON_ROOT_SHELL = {"_DockerCredentialOps.run_as"}
# Execs into a container of a BenchFlow-built, digest-pinned
# image (PERF-A's cloud-metadata helper, which shares only main's network
# namespace), never into the task's image, whose shell and PATH the route
# exists to avoid. Only MetadataHelper.run; adding to it needs a lead decision
# (test_own_image_execs_are_only_the_metadata_helper). Model: perf-all 3a8eaebe.
OWN_IMAGE_EXECS = {
    "sandbox/_metadata_helper.py": {
        "MetadataHelper.run": "/bin/sh of the pinned helper image (HELPER_TAG)",
    },
}
COMPARES = {
    "_heredoc_is_shell_script",
    "RemoteDockerSandbox._run_docker_compose_command",
    "RemoteDockerSandbox._run_docker_container_command",
    "RemoteDockerSandbox._docker_cli",
}


def _functions(path: Path):
    """(qualname, node, source) for every function in *path*."""
    text = path.read_text()
    tree = ast.parse(text)
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                out.append(
                    (
                        f"{prefix}{child.name}",
                        child,
                        ast.get_source_segment(text, child),
                    )
                )
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return out


def _exec_literals(node) -> bool:
    """Whether *node* (a function) holds a literal "exec" or "attach" in an
    argv it builds (not in a docstring)."""
    doc = ast.get_docstring(node, clean=False)
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Constant)
            and sub.value in ("exec", "attach")
            and sub.value != doc
        ):
            return True
    return False


def test_every_container_exec_builder_takes_its_shell_from_the_route():
    """Invariant 1: a sandbox exec or subprocess that reaches into a
    container is built from exec_route's route (or is a pre-agent start
    probe running a builtin). Fails on any builder not listed above."""
    found: dict[str, set[str]] = {}
    for path in sorted((SRC / "sandbox").rglob("*.py")):
        rel = str(path.relative_to(SRC))
        for qualname, node, source in _functions(path):
            if not _exec_literals(node):
                continue
            inner = [
                q
                for q, n, _ in _functions(path)
                if q.startswith(qualname + ".") and _exec_literals(n)
            ]
            if inner:
                continue  # the nested function is the builder
            if qualname in OWN_IMAGE_EXECS.get(rel, {}):
                # Not the task's container: the helper this object started,
                # by the id its own run printed.
                assert "self.container" in source, qualname
                assert "self.name" not in source, qualname
                continue
            found.setdefault(rel, set()).add(qualname)
            assert qualname in EXEC_BUILDERS.get(rel, {}), (
                f"{rel}:{qualname} builds a container exec argv outside the "
                "route; take the shell and PATH from exec_route"
            )
            if qualname in COMPARES:
                continue
            if qualname in START_PROBES:
                assert "SYSTEM_SHELL" in source and '":"' in source, qualname
                continue
            if qualname in ARGV_PROBES:
                assert "STARTUP_ENV_ARGV" in source, qualname
                assert "SYSTEM_SHELL" not in source and "/bin/sh" not in source
                continue
            assert "route" in source, f"{rel}:{qualname} names no route"
            if qualname in NON_ROOT_SHELL:
                continue
            assert '"/bin/sh"' not in source and "SYSTEM_SHELL," not in source, (
                f"{rel}:{qualname} starts a fixed shell instead of the route's"
            )
    stale = {
        (rel, q)
        for rel, names in EXEC_BUILDERS.items()
        for q in names
        if q not in found.get(rel, set())
    }
    assert not stale, f"listed builders that no longer build an exec: {stale}"


def test_no_backend_runs_a_constant_route_below_exec():
    """Invariant 1: a backend's own root steps below its exec (upload prep,
    the credential scrub, the start probe) ask exec_route for their route,
    so they run on the protected kit once the sandbox has one."""
    for path in sorted((SRC / "sandbox").rglob("*.py")):
        if path.name == "_base.py":
            continue
        text = path.read_text()
        assert "SYSTEM_SHELL_ROUTE" not in text, path.name


# Root-helper modules: what they run as root after the agent's turn.
ROOT_HELPERS = (
    "sandbox/lockdown.py",
    "sandbox/_root_launch.py",
    "sandbox/process/docker.py",
    "sandbox/process/apple.py",
    "sandbox/process/daytona.py",
    "sandbox/process/agentcore.py",
    "review/evidence.py",
    "review/evidence_runtime.py",
    "review/runner.py",
    "providers/litellm_runtime.py",
    "providers/litellm_bedrock_preflight.py",
    "agents/codex_connector_policy.py",
    "native_harness/runtime.py",
)
_SET = re.compile(
    r"(?:^|[\s;&|(`'\"{])(?:export\s+)?(PATH|LD_[A-Z_]+|PYTHON[A-Z_]*|BASH_ENV|ENV)="
)
_OWNED = re.compile(r"PATH|LD_[A-Z_]+|PYTHON[A-Z_]*|BASH_ENV|ENV")
# Every place a root helper still sets one, and why that does not put the
# image's value back (invariant 2).
ENV_SETS = {
    ("sandbox/lockdown.py", "<module>"): "VERIFIER_ENV: the task route's own env",
    ("sandbox/lockdown.py", "_keep_route_path"): (
        "SYSTEM_PATH, V3's system route, for every exec not on the kit route "
        "(no kit, or a non-root user); the kit route keeps its own PATH "
        "(batch 32b review A, M-1; round 2, M-1 and L-1)"
    ),
    ("sandbox/lockdown.py", "_path_inspector_prefix"): (
        "PATH = V3's pre-agent tool copies, a root-only folder made and checked "
        "before the agent ran (merge wave 32b)"
    ),
    ("sandbox/lockdown.py", "_path_inspector_command"): (
        "PATH = the same root-only pre-agent tool copies"
    ),
    ("sandbox/lockdown.py", "_prepare_bash_env_inspector"): (
        "PATH = that folder's own tool copies, while it is made (V3)"
    ),
    ("sandbox/lockdown.py", "_extras_probe_cmd"): (
        "PATH = the system folders, as root's route pins them when there is no "
        "kit (the route's own realpath/stat)"
    ),
    ("sandbox/lockdown.py", "_cloud_metadata_root_prelude"): (
        "keeps only folders its walk proves root's alone"
    ),
    ("sandbox/lockdown.py", "_cloud_metadata_agent_check_cmd"): (
        "the agent user's own check, not root"
    ),
    ("sandbox/lockdown.py", "_discover_pytest_plugins_cmd"): (
        "the verifier's trusted PYTHONPATH, read as the task's (plugin guard)"
    ),
    ("sandbox/lockdown.py", "_distro_pip_env"): "PYTHONNOUSERSITE=1 only",
    # Integ's (235017ae) S2 audit tools and verifier preparation runner.
    ("sandbox/lockdown.py", "_protected_root_prelude"): (
        "fixed system folders, then only those its walk proves root's alone"
    ),
    ("sandbox/lockdown.py", "_prep_env"): "fixed system folders, never the image's",
    ("sandbox/lockdown.py", "_prep_script"): "fixed system folders, never the image's",
    ("sandbox/lockdown.py", "_build_cleanup_cmd"): (
        "the hook clean-up's search-path inputs, data for the protected Python"
    ),
    ("sandbox/lockdown.py", "_trusted_verifier_bash_env"): (
        "the verifier's trusted BASH_ENV, data for test.sh's shell on the task "
        "route (main-sync MAIN-26)"
    ),
    ("sandbox/_root_launch.py", "<module>"): "docstring",
    ("sandbox/_root_launch.py", "inert_task_env"): "docstring",
    ("sandbox/_root_launch.py", "sanitize_root_env"): "pins the route's PATH",
    ("sandbox/_root_launch.py", "root_launch_script"): "keeps the task's, as data",
    ("providers/litellm_runtime.py", "_start_host_litellm"): "host process",
    ("providers/litellm_runtime.py", "_start_sandbox_litellm"): (
        "PYTHONPATH = the checked runtime folder alone"
    ),
    ("providers/litellm_bedrock_preflight.py", "preflight_sandbox_bedrock_patch"): (
        "PYTHONPATH = the runtime folder, checked in the same command"
    ),
}


def _env_sets(rel: str):
    text = (SRC / rel).read_text()
    tree = ast.parse(text)
    sets = []
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            self.generic_visit(node)

        def visit_Dict(self, node):
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _OWNED.fullmatch(key.value)
                ):
                    sets.append((stack[0] if stack else "<module>", key.value))
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for match in _SET.finditer(node.value):
                    sets.append((stack[0] if stack else "<module>", match.group(1)))

    Visitor().visit(tree)
    return sets


def test_no_root_helper_puts_path_loader_or_python_settings_back():
    """Invariant 2: after the route built root's environment, nothing in a
    root helper sets PATH, a loader, start-up or Python value again, except
    the places listed (each with why it does not restore the image's)."""
    for rel in ROOT_HELPERS:
        for function, name in _env_sets(rel):
            assert (rel, function) in ENV_SETS, (
                f"{rel}:{function} sets {name} after the route built root's env"
            )


def test_a_plumbing_exec_drops_route_owned_names_from_its_callers_env():
    """Invariant 2 at the choke point: a caller's PATH, loader, start-up and
    PYTHON* values never reach a plumbing exec; its other values do."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox._persistent_env = {"LD_PRELOAD": "/app/x.so", "TASK": "1"}
    merged = sandbox._merge_env(
        {
            "PATH": "/app/bin",
            "PYTHONPATH": "/app",
            "BASH_ENV": "/app/rc",
            "ENV": "/app/rc",
            "LD_LIBRARY_PATH": "/app/lib",
            "NODE_OPTIONS": "--require /app/x.js",
            "KEY": "v",
        },
        "root",
        plumbing=True,
        explicit_root=True,
    )
    assert merged == {"TASK": "1", "KEY": "v"}


async def _route_sandbox(answers: dict[tuple[str, str], str]):
    """A Docker sandbox whose exec records (service, command) and answers
    ``id -u <name>`` per service as that container's account database would."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = None
    sandbox.logger = logging.getLogger("test")
    calls: list[tuple[str, str]] = []

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        argv = list(argv)
        i = 1
        while argv[i].startswith("-"):
            i += 2 if argv[i] in ("-u", "-w", "-e", "--env") else 1
        service, command = argv[i], argv[-1]
        calls.append((service, command))
        match = re.search(r"id -u (\S+)", command)
        if match:
            uid = answers.get((service, match.group(1)))
            if uid is None:
                return ExecResult(return_code=1, stderr="no such user")
            return ExecResult(return_code=0, stdout=f"{uid}\n")
        return ExecResult(return_code=127, stderr="not found")

    sandbox._run_docker_compose_command = run
    return sandbox, calls


@pytest.mark.asyncio
async def test_a_named_user_is_looked_up_in_the_service_that_runs_the_command():
    """Invariant 3 (PT-C8-R3-4 / R3-5): ``app`` is uid 1000 in
    ``main`` but uid 0 in ``db``; the exec in ``db`` is routed as root."""
    sandbox, calls = await _route_sandbox({("main", "app"): "1000", ("db", "app"): "0"})
    main_route = await sandbox.exec_route("app")
    db_route = await sandbox.exec_route("app", service="db")
    assert main_route.root is False
    assert db_route.root is True and db_route.pin is not None
    looked_up = [s for s, c in calls if "id -u app" in c]
    assert looked_up == ["main", "db"]


@pytest.mark.asyncio
async def test_an_unknown_name_in_a_service_keeps_root_protection():
    """Invariant 3: a name the service's own database does not know is root."""
    sandbox, _ = await _route_sandbox({})
    route = await sandbox.exec_route("ghost", service="db")
    assert route.root is True and route.pin is not None


@pytest.mark.asyncio
async def test_docker_exec_in_a_service_looks_the_user_up_there():
    """PT-C8-R3-4/R3-5: DockerSandbox.exec passes its service to the route."""
    sandbox, calls = await _route_sandbox({("main", "app"): "1000", ("db", "app"): "0"})
    await sandbox.exec("true", user="app", service="db")
    assert any(s == "db" and "id -u app" in c for s, c in calls)
    assert not any(s == "main" and "id -u app" in c for s, c in calls)


def test_restore_starts_pid1_on_an_absolute_sleep_with_empty_loader_settings(
    monkeypatch,
):
    """PT-C8-R3-6: restore's ``docker run`` names its entrypoint by
    absolute path and starts it with the loader variables empty."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.protocol import SandboxImage

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = None
    sandbox.logger = logging.getLogger("test")
    sandbox.task_env_config = SimpleNamespace(network_mode=None)
    sandbox._snapshot_credentials = {}
    sandbox._compose_project = "proj"
    ran: list[list[str]] = []

    async def main_id():
        return "cid"

    async def inspect(_cid):
        return {
            "Config": {"Env": ["LD_PRELOAD=/app/x.so", "A=1"]},
            "HostConfig": {"NetworkMode": "none"},
            "Mounts": [],
        }

    async def remove(_cid):
        return None

    async def cli(argv, check=True, **kwargs):
        ran.append(list(argv))
        return ExecResult(return_code=0, stdout="c" * 64)

    sandbox._main_container_id = main_id
    sandbox._inspect_container = inspect
    sandbox._remove_replaced_container = remove
    sandbox._docker_cli = cli
    image = SandboxImage(provider="docker", ref="snap", meta={"benchflow_kit": "1"})
    from unittest.mock import AsyncMock

    from benchflow.sandbox import _inert_bootstrap

    sandbox._snapshot_image_bind_baseline = AsyncMock(return_value={"main": {}})
    sandbox._verify_replacement_bind_targets = AsyncMock()
    monkeypatch.setattr(_inert_bootstrap, "wait_ready", AsyncMock())
    asyncio.run(sandbox.restore(image))
    run = next(argv for argv in ran if argv[:1] == ["run"])
    entry = run[max(i for i, value in enumerate(run) if value == "--entrypoint") + 1]
    assert entry == "/opt/runtime/rt/bin/sh"
    assert run[-2:] == ["/opt/runtime/rt/bin/sleep", "infinity"]
    assert "trap" in run[run.index("snap") + 2]
    assert "LD_PRELOAD=" in run and "_BF_IMAGE_LD_PRELOAD=/app/x.so" in run


@pytest.mark.asyncio
async def test_the_credential_scrub_and_attach_take_the_routes_shell():
    """PT-C8-R3-7/R3-8: the snapshot credential helper's root step and
    ``attach`` start the route's absolute shell with the loader variables
    empty, never a shell or ``bash`` from the image's PATH."""
    from benchflow._runtime_tree import RuntimeKit
    from benchflow.sandbox.docker import (
        _DockerCredentialOps,
        interactive_root_shell,
        route_exec_argv,
    )

    sandbox, _ = await _route_sandbox({})
    sandbox.set_runtime_kit(
        RuntimeKit(trusted=("/usr/bin", "/bin"), python=True, bash=True)
    )
    ran: list[list[str]] = []

    async def cli(argv, check=True, **kwargs):
        ran.append(list(argv))
        return ExecResult(return_code=0)

    sandbox._docker_cli = cli
    await _DockerCredentialOps(sandbox, "ctr").run("ls /root")
    argv = ran[0]
    assert argv[argv.index("ctr") + 1] == "/opt/runtime/rt/bin/sh"
    assert "LD_PRELOAD=" in argv
    route = await sandbox.exec_route("root", system_shell=True)
    attach = route_exec_argv(
        route, "", container="main", user="0", stdin=True, tty=True,
        shell_argv=interactive_root_shell(route),
    )  # fmt: skip
    after = attach[attach.index("main") + 1 :]
    # Since round 10 (PT-C8-R9-5): from env -i, terminal data only.
    assert after[:2] == ["/opt/runtime/rt/bin/env", "-i"]
    assert after[-4:] == ["/opt/runtime/rt/bin/bash", "--noprofile", "--norc", "-i"]
    assert "-u" in attach and attach[attach.index("-u") + 1] == "0"
    assert f"PATH={route.pin}" in attach


@pytest.mark.asyncio
async def test_the_live_launcher_stages_its_env_privately_and_exclusively():
    """PT-C8-R3-8, on integ's live transport: the
    launch env never touches a container pathname (no fixed /tmp/.runtime_env,
    no staged file) and no value is in argv; it is the exec's first stdin
    line, read by root's bootstrap before anything else, and the launch starts
    the route's bash."""
    import base64

    from benchflow._runtime_tree import RUN_DIR, RuntimeKit
    from benchflow.sandbox.process.docker import DockerProcess

    sandbox, _ = await _route_sandbox({})
    sandbox.set_runtime_kit(RuntimeKit(trusted=("/usr/bin",), python=False, bash=True))
    process = DockerProcess("proj", "/tmp/proj", [], sandbox=sandbox)
    seen: list[list[str]] = []

    class Stdin:
        data = b""

        def write(self, data):
            Stdin.data += data

        async def drain(self):
            pass

    class Proc:
        returncode = 0
        pid = 1
        stdin = Stdin()
        stdout = stderr = None

        async def communicate(self, data=None):
            return b"", b""

    async def fake_exec(*argv, **kwargs):
        seen.append(list(argv))
        return Proc()

    import benchflow.sandbox.process.docker as module

    original = module.asyncio.create_subprocess_exec
    module.asyncio.create_subprocess_exec = fake_exec
    try:
        process._set_process = lambda proc: None
        process._host_env = lambda: {}
        await process.start("agent-acp", env={"KEY": "secret-v"})
    finally:
        module.asyncio.create_subprocess_exec = original
    (launch,) = seen
    text = " ".join(launch)
    assert "secret-v" not in text and "/tmp/.runtime_env" not in text
    assert f"{RUN_DIR}/env-" not in text
    assert launch[launch.index("main") + 1] == "/opt/runtime/rt/bin/bash"
    assert launch[-1].count("IFS= read -r __bf_live_env") == 1
    assert base64.b64decode(Stdin.data.strip()) == b"export KEY=secret-v\n"


def test_the_proxy_child_gets_only_the_checked_runtime_python_path(tmp_path):
    """PT-C8-R3-7: the in-sandbox proxy's launcher builds the child's env
    from the route's: no loader or start-up value, PYTHONPATH exactly the
    runtime folder (no empty entry), started in that folder."""
    from benchflow.providers.litellm_runtime import _sandbox_launcher_source

    runtime = tmp_path / "rt"
    runtime.mkdir()
    child = runtime / "litellm"
    child.write_text(
        # Renamed into place whole: the poll below reads it once it exists.
        f"#!/bin/sh\n{{ env; echo CWD=$(pwd); }} > {runtime}/child-env.tmp\n"
        f"mv {runtime}/child-env.tmp {runtime}/child-env\nsleep 0.1\n"
    )
    child.chmod(0o755)
    launcher = runtime / "launcher.py"
    launcher.write_text(_sandbox_launcher_source())
    cfg = {
        "litellm": str(child),
        "config": "c.yaml",
        "stdout": str(runtime / "out"),
        "stderr": str(runtime / "err"),
        "pid": str(runtime / "pid"),
        "state": str(runtime / "state"),
        "runtime_dir": str(runtime),
        "env": {
            "PYTHONPATH": f"{runtime}:",
            "LD_PRELOAD": "/app/x.so",
            "PYTHONSTARTUP": "/app/s.py",
            "KEY": "v",
        },
    }
    (runtime / "cfg.json").write_text(json.dumps(cfg))
    env = {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/app/lib", "PYTHONHOME": "/x"}
    subprocess.run(
        [sys.executable, "-I", str(launcher), str(runtime / "cfg.json")],
        env=env,
        cwd="/",
        check=True,
        capture_output=True,
    )
    for _ in range(50):
        if (runtime / "child-env").exists():
            break
        time.sleep(0.1)
    lines = dict(
        line.split("=", 1)
        for line in (runtime / "child-env").read_text().splitlines()
        if "=" in line
    )
    assert lines["PYTHONPATH"] == str(runtime)
    assert lines["CWD"] == str(runtime)
    assert lines["KEY"] == "v"
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONSTARTUP", "PYTHONHOME"):
        assert name not in lines, name


@pytest.mark.parametrize(
    "workspace", ["/", "/opt/runtime", "/opt/runtime/x", "/usr", "/etc"]
)
def test_a_workspace_that_would_hand_over_roots_programs_is_refused(workspace):
    """Invariant 2's tree: no grant covers the runtime tree or a system
    folder; an /opt workspace stays root's and sticky."""
    from benchflow._runtime_tree import workspace_grant_command, workspace_grant_problem

    assert workspace_grant_problem(workspace) is not None
    assert workspace_grant_problem("/app") is None
    assert workspace_grant_problem("/opt") is None
    grant = workspace_grant_command("/opt", "agent")
    assert "chmod 1777 /opt" in grant and "! -path /opt/runtime" in grant


def test_root_python_helpers_are_isolated_from_the_working_folder():
    """PT-C8-R3-1: the review's root Python runs ``-I -S`` from ``/`` (the
    kit's by path once it exists); no helper passes a bare ``python3 -c``."""
    from benchflow._runtime_tree import RuntimeKit
    from benchflow.review.evidence import root_python_command

    class Env:
        async def runtime_kit_for_route(self):
            return RuntimeKit(trusted=("/usr/bin",), python=True, bash=False)

    command = asyncio.run(root_python_command(Env(), ["-c", "pass"]))
    # Since round 10 (PT-C8-R9-8): one entry point, -I -S -B.
    assert command.startswith("/opt/runtime/rt/py/python3 -I -S -B ")
    source = (SRC / "review" / "evidence.py").read_text()
    assert '"python3",\n            "-c"' not in source
    assert '["python3", "-c"' not in source


def test_root_never_runs_a_cached_runtime_before_it_is_checked():
    """PT-C8-R3-5/R3-6 and R3-4: uv's version probe, uv's
    managed Python, Codex's root Node and the native CLI probe each come
    after a root-only check of what they run, in the same command."""
    from benchflow._uv_pin import managed_python_install_command, uv_install_command

    uv = uv_install_command()
    assert uv.index("is not root") < uv.index("--version")
    managed = managed_python_install_command()
    assert managed.index("is not root") < managed.index("python install")
    policy = (SRC / "agents" / "codex_connector_policy.py").read_text()
    assert "runtime_custody_check(NODE_RUNTIME_PATHS" in policy
    native = (SRC / "native_harness" / "runtime.py").read_text()
    assert "runtime_custody_check(NODE_RUNTIME_PATHS" in native


def test_kill_and_release_helpers_set_no_path_of_their_own():
    """PT-C8-R3-3: the kill, keeping kill and release commands carry no
    PATH of their own; the route's (the kit's) is the one they run with."""
    lockdown = (SRC / "sandbox" / "lockdown.py").read_text()
    assert "safe_path" not in lockdown
    assert '"PATH": _BASH_ENV_CHECK_PATH' not in lockdown


def test_the_runtime_check_walks_link_hops_before_canonicalizing(tmp_path):
    """PT-C8-R3-9: a runtime reached through a link in a folder the
    current user (standing in for the agent) can write is refused, even
    when the link's target passes today."""
    from benchflow._root_only import root_only_path_check

    if os.geteuid() == 0:
        pytest.skip("the check's positive case needs root; refusal is shown here")
    hop = tmp_path / "hop"
    hop.mkdir()
    (hop / "node").symlink_to("/bin")
    check = root_only_path_check(str(hop / "node"), "BenchFlow's Node")
    done = subprocess.run(["/bin/sh", "-c", check], capture_output=True, text=True)
    assert done.returncode != 0
    ok = subprocess.run(
        ["/bin/sh", "-c", root_only_path_check("/bin/sh", "the shell")],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr


@pytest.mark.asyncio
async def test_a_daytona_branch_child_never_starts_a_root_shell_with_the_preload():
    """On native Daytona the provider's
    root shells must start from a clean environment. A branch child is
    created from a checkpoint the agent wrote, so its values come only from
    the checkpoint's startup receipt (integ's ``_TOOLBOX_STARTUP_META``):
    a malformed receipt is refused before any provider call, and one that
    keeps loader values (LD_PRELOAD, BASH_ENV) starts every shell with those
    names empty and the values aside as data under ``_BF_IMAGE_<name>``
    (DAYTONA-GAPS rule 2; at 29b6ea03c a loader value other than BASH_ENV
    was refused instead)."""
    pytest.importorskip("daytona")
    import daytona

    from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
    from benchflow.sandbox.daytona import _TOOLBOX_STARTUP_META, DaytonaSandbox
    from benchflow.sandbox.protocol import SandboxImage

    mark = "/nonexistent/app/lib/bf-agent-profile.so"
    image_env = {"PATH": "/usr/bin:/bin", "LD_PRELOAD": mark, "BASH_ENV": mark}
    shells: list[dict[str, str]] = []

    class Provider:
        def __init__(self, params):
            self.env = {**image_env, **(getattr(params, "env_vars", None) or {})}

            async def process_exec(command, env=None, timeout=None):
                started = {**self.env, **(env or {})}
                shells.append(started)
                ran = subprocess.run(
                    ["/bin/sh", "-c", command],
                    env=started,
                    capture_output=True,
                    text=True,
                )
                return SimpleNamespace(
                    result=ran.stdout + ran.stderr, exit_code=ran.returncode
                )

            self.process = SimpleNamespace(exec=process_exec)

        async def delete(self):
            return None

    child = object.__new__(DaytonaSandbox)
    child.logger = logging.getLogger("test")
    child._strategy = SimpleNamespace(supports_snapshot=True)
    child._start_snapshot_ref = None

    async def create(params):
        child._sandbox = Provider(params)

    child._create_sandbox = create
    child._require_sandbox = lambda: child._sandbox

    def receipt(**values):
        return SandboxImage(
            provider="daytona",
            ref="ckpt",
            meta={
                _TOOLBOX_STARTUP_META: json.dumps(
                    {**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), **values}
                )
            },
        )

    with pytest.raises(RuntimeError):
        child.start_from_snapshot(
            SandboxImage(
                provider="daytona", ref="ckpt", meta={_TOOLBOX_STARTUP_META: "{}"}
            )
        )
    assert not shells, "a refused receipt reached the provider"

    assert child.start_from_snapshot(receipt(LD_PRELOAD=mark, BASH_ENV=mark)) is True
    params = daytona.CreateSandboxFromSnapshotParams(snapshot="ckpt")
    await child._create_sandbox_without_toolbox_bash_env(
        params, startup_env=child._start_snapshot_env
    )
    assert shells, "the provider started no shell"
    assert all(shell.get("LD_PRELOAD") == "" for shell in shells)
    assert all(shell.get("BASH_ENV") == "" for shell in shells)
    assert all(shell.get("_BF_IMAGE_BASH_ENV") == mark for shell in shells)
    assert all(shell.get("_BF_IMAGE_LD_PRELOAD") == mark for shell in shells)


def test_own_image_execs_are_only_the_metadata_helper():
    """The category holds only the metadata helper's exec;
    adding to it needs a lead decision."""
    assert {rel: set(names) for rel, names in OWN_IMAGE_EXECS.items()} == {
        "sandbox/_metadata_helper.py": {"MetadataHelper.run"}
    }


@pytest.mark.asyncio
async def test_the_metadata_helper_execs_only_the_pinned_container_it_started():
    """Every exec targets the container the helper's
    own `docker run` created, by the id it printed; that container runs the
    image id this process's own build of the digest-pinned recipe printed,
    tagged HELPER_TAG (PERF-A builds, never takes an image found by tag); main (the task's
    container) is never an exec target, and with no id from the helper's own
    run (or main's id) there is no exec at all. Merge wave batch 31, on
    PERF-A dc006aaa's helper (model: perf-all 3a8eaebe)."""
    import hashlib

    from benchflow.sandbox import _metadata_helper as helper_mod

    main = "4d95a10d423d" + "0" * 52
    helper_id = "9e1b2c3d4f50" + "2" * 52
    image_id = "sha256:" + "7" * 64
    assert "@sha256:" in helper_mod.HELPER_BASE
    assert helper_mod.HELPER_DOCKERFILE.startswith(f"FROM {helper_mod.HELPER_BASE}\n")
    recipe = hashlib.sha256(helper_mod.HELPER_DOCKERFILE.encode()).hexdigest()[:16]
    assert f"benchflow-helper:metadata-{recipe}" == helper_mod.HELPER_TAG

    for run_prints in (helper_id, "", main):
        calls: list[list[str]] = []

        async def docker(
            args, *, timeout, stdin=None, run_prints=run_prints, calls=calls
        ):
            calls.append(list(args))
            if args[:1] == ["build"]:
                assert stdin == helper_mod.HELPER_DOCKERFILE.encode()
            out = ""
            if args[:1] == ["build"]:
                out = image_id + "\n"
            elif args[:1] == ["run"]:
                out = run_prints + "\n"
            return SimpleNamespace(stdout=out, stderr="", return_code=0)

        helper_mod._READY.clear()
        helper_mod._FAILED.clear()
        helper = helper_mod.MetadataHelper(
            docker, daemon="merge-wave", name="t-bf-metadata", labels={}
        )
        helper.install_early(main, "echo rules")
        await helper.installed(main, "echo rules")
        await helper.aclose()
        (build,) = [c for c in calls if c[:1] == ["build"]]
        assert build[build.index("--tag") + 1].startswith(
            f"{helper_mod.HELPER_TAG}-build-"
        )
        (run,) = [c for c in calls if c[:1] == ["run"]]
        assert run[run.index("--entrypoint") + 2] == image_id
        execs = [c for c in calls if c[:1] == ["exec"]]
        assert all(c[1] != main for c in execs)
        if run_prints == helper_id:
            assert execs and {c[1] for c in execs} == {helper_id}
        else:
            # No usable id from its own run: nothing is exec'd, by name or
            # otherwise.
            assert execs == []
