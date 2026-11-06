"""mini-swe-agent runs end to end on the scripted fake model (deterministic).

mini-swe is a plugin agent (``mini-swe-acp`` in benchflow-ai/agents). Its
sandbox install used to be a bare ``pip install mini-swe-agent``; from
2026-10-18 that resolved LiteLLM 1.106.0, whose ``completion()`` imports the
optional ``mcp`` package whenever tools are passed. mini-swe passes its bash
tool on every call, so each trial failed with ``ACP error -32603: No module
named 'mcp'`` after a clean install and ~5 minutes of retries, and nothing in
the suite ran mini-swe to notice.

This test loads the plugin from the agents commit BenchFlow pins
(``DEFAULT_AGENTS_SOURCE``), as an installed distribution would, and runs
``bench eval run --agent mini-swe`` on the tier's task. The model is the fake's
chat-completions route behind BenchFlow's LiteLLM proxy (``vllm/fake-policy``),
the same path a real OpenAI-compatible provider such as MiMo takes: the shim
builds ``openai/<model>`` with the proxy as ``api_base`` and sends mini-swe's
bash tool. The fake writes hello.txt, then submits.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
import uuid
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}

AGENT = "mini-swe"
PLUGIN_PATH = "acp/mini-swe-acp"


def _plugin_pythonpath(root: Path) -> str:
    """A ``PYTHONPATH`` on which ``bench`` finds mini-swe-acp as installed.

    The plugin source comes from the pinned agents commit; a minimal
    ``.dist-info`` beside it carries the package's own entry point, so
    BenchFlow's ``benchflow.agents`` autoload registers it like a pip install.
    ``BENCHFLOW_AGENTS_DIR`` can instead name a local checkout or plugin
    subtree. Use the subtree when testing core agents too: a whole catalog
    is an explicit override of their built-in manifests.
    """
    from benchflow._utils.benchmark_repos import resolve_source
    from benchflow.agents.remote_manifests import DEFAULT_AGENTS_SOURCE

    local = os.environ.get("BENCHFLOW_AGENTS_DIR", "").strip()
    if local:
        plugin = Path(local) / PLUGIN_PATH
        if not plugin.is_dir():
            plugin = Path(local)
    else:
        repo, _, ref = DEFAULT_AGENTS_SOURCE.rpartition("@")
        plugin = resolve_source(repo, path=PLUGIN_PATH, ref=ref)
    project = tomllib.loads((plugin / "pyproject.toml").read_text())["project"]
    entry_points = project["entry-points"]["benchflow.agents"]
    site = root / "plugin-site"
    info = site / f"{project['name'].replace('-', '_')}-{project['version']}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\n"
        f"Version: {project['version']}\n"
    )
    (info / "entry_points.txt").write_text(
        "[benchflow.agents]\n"
        + "".join(f"{name} = {value}\n" for name, value in entry_points.items())
    )
    paths = [str(site), str(plugin / "src")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(paths)


def _run_mini_swe(root: Path, task: Path, jobs: str, *extra: str):
    """One ``bench eval run --agent mini-swe`` of ``task``; (run, trial)."""
    fake = None
    if not h.proxy_runs_in_sandbox(SANDBOX):
        fake = h.host_fake_llm(root / f"{jobs}-fake-llm.jsonl")
    url = fake.__enter__() if fake is not None else None
    try:
        run = h.run_bench(
            ["eval", "run"],
            tasks_dir=task,
            jobs_dir=root / jobs,
            sandbox=SANDBOX,
            host_fake_url=url,
            route="vllm",
            # One attempt: a failed trial fails the test on its own error.
            extra=["--retry-attempts", "0", *extra],
            agent=AGENT,
            env_extra={"PYTHONPATH": _plugin_pythonpath(root / f"{jobs}-plugin")},
        )
    finally:
        if fake is not None:
            fake.__exit__(None, None, None)
    return run, run.trial(task.name)


def _assert_solved(run: h.CliRun, trial: Path) -> None:
    result = json.loads((trial / "result.json").read_text())
    agent_log = trial / "agent" / "mini_swe.txt"
    log = agent_log.read_text() if agent_log.is_file() else ""
    assert result.get("error") is None, (result.get("error"), log[-2000:])
    assert (result.get("rewards") or {}).get("reward") == 1.0, run.output[-4000:]
    assert "ModuleNotFoundError" not in log, log[-2000:]
    assert result["agent"] == AGENT

    # Every model call went through the proxy with mini-swe's bash tool, and
    # the two scripted steps ran: the write, then the submit.
    calls = h.llm_calls(trial)
    assert [c["kind"] for c in calls] == ["main", "main"], calls
    commands = [tc["arguments"].get("command") for c in calls for tc in c["tool_calls"]]
    assert [tc["name"] for c in calls for tc in c["tool_calls"]] == ["bash", "bash"]
    assert commands[-1] == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", commands

    # The sandbox ran the plugin's pinned install, ending in its tool-call probe.
    install = (trial / "agent" / "install-stdout.txt").read_text()
    assert (
        "pip --isolated install -q -r /opt/benchflow/mini-swe-requirements.txt"
        in install
    )
    assert "mock_response='ok'" in install
    assert run.returncode == 0, run.output[-4000:]


def test_mini_swe_solves_the_task_through_the_proxy(tmp_path):
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    task = h.materialize_task(
        h.TaskVariant("mini-swe-hello", "mini-swe-hello-pass"), tmp_path / "tasks"
    )
    _assert_solved(*_run_mini_swe(tmp_path, task, "jobs"))


def _without_environment_plane(task: Path) -> Path:
    """The task without its fake-llm service: no environment is provisioned
    before the install, so the agent layer applies (a provisioned
    environment installs in the trial). On Docker the proxy and
    the fake it calls run on the host, so the run needs no in-sandbox fake."""
    task_md = task / "task.md"
    plane = "benchflow:\n  environment:\n    manifest: environment.toml\n"
    text = task_md.read_text()
    assert plane in text
    task_md.write_text(text.replace(plane, "", 1))
    (task / "environment.toml").unlink()
    return task


def test_the_pinned_mini_swe_install_comes_from_its_cached_layer(tmp_path):
    """mini-swe-acp registers its pinned install as cacheable: the first
    trial builds (or finds) the agent layer, the second reuses it, and both
    solve the task without installing in the trial."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if h.proxy_runs_in_sandbox(SANDBOX):
        pytest.skip(f"{SANDBOX} runs the proxy and its fake in the sandbox")
    task = _without_environment_plane(
        h.materialize_task(
            h.TaskVariant("mini-swe-layer", "mini-swe-hello-pass"), tmp_path / "tasks"
        )
    )
    layers = []
    for jobs in ("layer-1", "layer-2"):
        run, trial = _run_mini_swe(tmp_path, task, jobs)
        _assert_solved(run, trial)
        layers.append(h.read_json(trial / "config.json")["layers"]["agent"])
        install = (trial / "agent" / "install-stdout.txt").read_text()
        assert "ran once in the cached agent layer" in install, install[-500:]
    first, second = layers
    assert first["status"] in {"hit", "miss"}, first
    assert first["install"] == "from layer", first
    assert second["install"] == "from layer", second
    # The task image is built from its Dockerfile; a rebuild that yields a new
    # image digest is a new base and rightly misses (the same was seen on
    # the proxy layer). The same base must reuse the layer.
    if second["base"] == first["base"]:
        assert second["status"] == "hit", (first, second)
        assert second["key"] == first["key"]


@pytest.mark.integration
def test_two_pythons_install_identically_fresh_cold_and_warm(tmp_path, request):
    """Guards M1 in 13b7aaa58 and L2 at 092eb18ad: one base for all phases."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("requires a real Docker daemon and a two-Python task image")
    task = _without_environment_plane(
        h.materialize_task(
            h.TaskVariant("mini-swe-two-pythons", "mini-swe-hello-pass"),
            tmp_path / "tasks",
        )
    )
    dockerfile = task / "environment" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text()
        + "\nCOPY --from=python:3.13.9-slim-bookworm /usr/local /opt/task-python\n"
        "RUN ln -s /opt/task-python/lib/libpython3.13.so.1.0 "
        "/usr/local/lib/libpython3.13.so.1.0 && ldconfig && "
        "/opt/task-python/bin/python3 --version && /usr/bin/python3 --version\n"
        'ENV PATH="/opt/task-python/bin:/usr/local/sbin:/usr/local/bin:'
        '/usr/sbin:/usr/bin:/sbin:/bin"\n'
        # Make the first cached run a cold build, including on repeated tests.
        f'LABEL benchflow.test.two-pythons="{uuid.uuid4().hex}"\n'
    )
    # A rebuild can change the base digest even with an unchanged Dockerfile.
    # Pin every phase to the one build, so warm reuse is a strict assertion.
    tag = f"mini-swe-two-pythons-{uuid.uuid4().hex}"
    iidfile = tmp_path / "task-image.id"
    subprocess.run(
        [
            "docker",
            "build",
            "--iidfile",
            str(iidfile),
            "-t",
            tag,
            str(dockerfile.parent),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    request.addfinalizer(
        lambda: subprocess.run(
            ["docker", "image", "rm", tag], check=True, capture_output=True, timeout=60
        )
    )
    image = iidfile.read_text().strip()
    task_md = task / "task.md"
    text = task_md.read_text()
    assert "sandbox:\n" in text
    task_md.write_text(
        text.replace("sandbox:\n", f"sandbox:\n  docker_image: {image}\n", 1)
    )
    verifier = task / "tests" / "test.sh"
    verifier.write_text(
        verifier.read_text()
        + """
/opt/benchflow/mini-swe-venv/bin/python -I - <<'PY' > /logs/verifier/install.json
import hashlib, importlib.metadata, json, pathlib, sys
root = pathlib.Path(sys.prefix)
print(json.dumps({
    "version": sys.version,
    "base_executable": str(pathlib.Path(sys._base_executable).resolve()),
    "pyvenv.cfg": (root / "pyvenv.cfg").read_text(),
    "packages": sorted((d.metadata["Name"], d.version)
                       for d in importlib.metadata.distributions()),
    "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*")) if p.is_file() and p.suffix != ".pyc"},
}, sort_keys=True))
PY
"""
    )
    snapshots, layers = [], []
    for name, extra in (("fresh", ("--fresh-install",)), ("cold", ()), ("warm", ())):
        run, trial = _run_mini_swe(tmp_path, task, name, *extra)
        _assert_solved(run, trial)
        snapshots.append(h.read_json(trial / "verifier" / "install.json"))
        layers.append(h.read_json(trial / "config.json").get("layers", {}).get("agent"))
    fresh, cold, warm = snapshots
    assert fresh["base_executable"].startswith("/opt/benchflow/python/"), fresh
    assert fresh["version"] == cold["version"] == warm["version"]
    assert (
        fresh["base_executable"] == cold["base_executable"] == warm["base_executable"]
    )
    assert fresh["pyvenv.cfg"] == cold["pyvenv.cfg"] == warm["pyvenv.cfg"]
    assert fresh["packages"] == cold["packages"] == warm["packages"]
    assert fresh["files"] == cold["files"] == warm["files"]
    assert layers[1]["status"] == "miss", layers
    assert layers[2]["status"] == "hit", layers
    assert layers[1]["base"] == layers[2]["base"], layers
    assert layers[1]["key"] == layers[2]["key"], layers
