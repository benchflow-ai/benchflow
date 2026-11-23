"""A task can't lift the cloud metadata block by itself, on a real
Docker daemon against the VM's real metadata server.

On 0.10.0 a task's ``allow_cloud_metadata = true`` (or
``metadata.x-allow-cloud-metadata = true``) lifted the block for its trials
with no operator flag and nothing printed: the agent's turn and test.sh both
reached the VM's metadata server. Now the task's key is a request and the
run's ``--allow-cloud-metadata`` the permission:

- without the flag, every phase is refused the endpoints under every network
  policy, as root and as a non-root user, and a task that declares the
  key is refused before anything starts, with the flag named;
- with the flag, the key takes effect and the run card, result.json and the
  reward's provenance each say so, naming the key and the flag;
- remote Docker the same.

The probe asks GCE's metadata server for the instance id only
(``/computeMetadata/v1/instance/id`` with ``Metadata-Flavor: Google``), by
address and by name, directly and through any proxy the sandbox's
environment names; it records the HTTP status and the body's length, never
the body, and never asks for a token or anything under ``service-accounts``.
The tests run only where the host itself gets an answer to that request (a
GCE VM), and are meant for a VM with no service account (the cycle's rule).
They carry no marker: a plain ``pytest tests/test_metadata_optin_docker.py``
runs them where Docker answers, and skips them elsewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

import pytest

LABEL = "bf.test.mdoptin"
INSTANCE_ID = "/computeMetadata/v1/instance/id"
FLAG = "--allow-cloud-metadata"


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _host_reaches_metadata() -> bool:
    """The host's own answer to the instance-id request: the tests mean
    something only where the server answers the host."""
    request = urllib.request.Request(
        f"http://169.254.169.254{INSTANCE_ID}", headers={"Metadata-Flavor": "Google"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=3) as answer:
            return answer.status == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
    pytest.mark.skipif(
        not _host_reaches_metadata(),
        reason="the host gets no answer from a GCE metadata server",
    ),
]

# The probe: one line per address, the status and the body's length only.
PROBE_PY = r"""
import json, os, sys, urllib.error, urllib.request

PHASE = sys.argv[1]
URLS = (
    "http://169.254.169.254__PATH__",
    "http://metadata.google.internal__PATH__",
)
PROXY = {k: v for k, v in os.environ.items()
         if k.lower() in ("http_proxy", "https_proxy", "all_proxy")}
ROUTES = [("direct", {})] + ([("env-proxy", None)] if PROXY else [])


def fetch(url, proxies):
    handler = urllib.request.ProxyHandler(proxies) if proxies is not None else urllib.request.ProxyHandler()
    request = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    row = {}
    try:
        with urllib.request.build_opener(handler).open(request, timeout=6) as answer:
            row["status"] = answer.status
            row["bytes"] = len(answer.read(512))
    except urllib.error.HTTPError as e:
        row["status"] = e.code
    except Exception as e:
        row["status"] = None
        row["error"] = (type(e).__name__ + ": " + str(getattr(e, "reason", e)))[:120]
    return row


for url in URLS:
    for route, proxies in ROUTES:
        row = {"phase": PHASE, "uid": os.getuid(), "url": url, "route": route}
        row.update(fetch(url, proxies))
        print("BF-MDPROBE " + json.dumps(row, sort_keys=True), flush=True)
""".replace("__PATH__", INSTANCE_ID)

# Run as the caller, and when that is root, again as nobody and as the
# sandbox's agent user (when the image has one by then).
PROBE_SH = r"""#!/bin/bash
phase=$1; out=$2
python3 /opt/mdprobe/probe.py "$phase" >> "$out" 2>&1
if [ "$(id -u)" = 0 ]; then
  setpriv --reuid=65534 --regid=65534 --clear-groups \
    python3 /opt/mdprobe/probe.py "$phase-nobody" >> "$out" 2>&1
  if id -u agent >/dev/null 2>&1 && [ "$(id -u agent)" != 0 ]; then
    setpriv --reuid="$(id -u agent)" --regid="$(id -g agent)" --clear-groups \
      python3 /opt/mdprobe/probe.py "$phase-agent" >> "$out" 2>&1
  fi
fi
exit 0
"""

DOCKERFILE = """FROM python:3.12-slim
COPY probe.py probe.sh acp.sh /opt/mdprobe/
RUN chmod 755 /opt/mdprobe /opt/mdprobe/probe.py /opt/mdprobe/probe.sh \\
    /opt/mdprobe/acp.sh && mkdir -p /app /shared && chmod 1777 /shared
WORKDIR /app
"""

# The agent's turn writes its rows to /tmp too: test.sh carries them into
# /logs/verifier, which every backend copies back (remote Docker copies no
# file of the oracle's /logs/agent).
SOLVE_SH = "#!/bin/bash\n/opt/mdprobe/probe.sh oracle /tmp/mdprobe-agent.txt\n"
TEST_SH = (
    "#!/bin/bash\n"
    "/opt/mdprobe/probe.sh verifier /logs/verifier/mdprobe-verifier.txt\n"
    "cp /tmp/mdprobe-agent.txt /logs/verifier/mdprobe-agent.txt 2>/dev/null\n"
    "echo 1 > /logs/verifier/reward.txt\n"
)
SCRIPT_AGENT = "/opt/mdprobe/probe.sh agent /tmp/mdprobe-agent.txt\n"

# A stub ACP agent (no model call) that probes at its prompt: the root
# model agent's row, which a script: agent can't be (it needs a non-root
# sandbox user).
ACP_SH = r"""#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' "$line" | grep -o '"id": *[0-9][0-9]*' | head -1 | grep -o '[0-9][0-9]*$')
  m=$(printf '%s' "$line" | grep -o '"method": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$m" in
    initialize) r='{"protocolVersion":1,"agentInfo":{"name":"mdprobe","version":"1"},"agentCapabilities":{"loadSession":false,"promptCapabilities":{"image":false,"audio":false,"embeddedContext":false}},"authMethods":[]}';;
    session/new) r='{"sessionId":"mdprobe"}';;
    session/prompt)
      /opt/mdprobe/probe.sh agent /tmp/mdprobe-agent.txt
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"mdprobe","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"probed"}}}}\n'
      r='{"stopReason":"end_turn"}';;
    session/set_config_option) r='{"configOptions":[]}';;
    *) r='{}';;
  esac
  [ -n "$id" ] && printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$r"
done
"""
MANIFEST = """contract_version = "1.0"
name = "mdprobe"
default_model = "openai/gpt-4.1-mini"
install_cmd = "test -x /opt/mdprobe/acp.sh"
launch_cmd = "/opt/mdprobe/acp.sh"
"""

POLICIES = {
    "public": "",
    "no-network": 'network_mode = "no-network"\n',
    "denylist": 'network_mode = "denylist"\nblocked_hosts = ["example.org"]\n',
    "allowlist": 'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
}
KEYS = {
    "environment": ("environment", "allow_cloud_metadata = true\n"),
    "metadata": ("metadata", '"x-allow-cloud-metadata" = true\n'),
}


def build_probe_image(tmp_path_factory):
    """Build the probe image once per module; yields its tag, removes it after."""
    context = tmp_path_factory.mktemp("mdoptin-image")
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "probe.py").write_text(PROBE_PY)
    (context / "probe.sh").write_text(PROBE_SH)
    (context / "acp.sh").write_text(ACP_SH)
    tag = f"bf-test-mdoptin:{uuid.uuid4().hex[:12]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.fail(
            f"cannot build the probe image: {built.stderr[-300:]}", pytrace=False
        )
    try:
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


def _task(
    root: Path, image: str, *, policy: str = "public", key: str | None = None
) -> Path:
    """A task.toml task whose oracle, script agent and test.sh each probe."""
    task = root / f"mdoptin-{policy}-{key or 'nokey'}"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    environment = f'docker_image = "{image}"\n' + POLICIES[policy]
    metadata = ""
    if key is not None:
        table, line = KEYS[key]
        if table == "environment":
            environment += line
        else:
            metadata = line
    (task / "task.toml").write_text(
        'version = "1.0"\n\n'
        + (f"[metadata]\n{metadata}\n" if metadata else "")
        + "[agent]\ntimeout_sec = 180.0\n\n"
        "[verifier]\ntimeout_sec = 120.0\n\n"
        f"[environment]\n{environment}"
    )
    (task / "instruction.md").write_text("Probe the cloud metadata endpoints.\n")
    for path, body in (
        (task / "solution" / "solve.sh", SOLVE_SH),
        (task / "tests" / "test.sh", TEST_SH),
    ):
        path.write_text(body)
        path.chmod(0o755)
    return task


def _taskmd(root: Path, image: str) -> Path:
    """The same probe task as task.md with ``sandbox.allow_cloud_metadata``."""
    task = root / "mdoptin-taskmd"
    for sub in ("environment", "oracle", "verifier"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    (task / "task.md").write_text(
        "---\n"
        'schema_version: "1.3"\n'
        "agent:\n  timeout_sec: 180\n"
        "verifier:\n  timeout_sec: 120\n"
        f"sandbox:\n  docker_image: {image}\n  allow_cloud_metadata: true\n"
        "---\n# mdoptin-taskmd\n\n## prompt\n\nProbe the cloud metadata endpoints.\n"
    )
    (task / "verifier" / "verifier.md").write_text(
        "---\n"
        'document_version: "0.3"\n'
        "verifier:\n  name: mdoptin\n  default_strategy: deterministic\n"
        "  strategies:\n    deterministic:\n      type: script\n"
        "      command: ./test.sh\n"
        "---\n\n## verifier intent\n\nProbe and pass.\n"
    )
    for path, body in (
        (task / "oracle" / "solve.sh", SOLVE_SH),
        (task / "verifier" / "test.sh", TEST_SH),
    ):
        path.write_text(body)
        path.chmod(0o755)
    return task


def _probe_rows(trial: Path) -> list[dict]:
    """Both phases' rows, the agent's turn's as test.sh carried them."""
    rows = []
    for name in ("verifier/mdprobe-agent.txt", "verifier/mdprobe-verifier.txt"):
        path = trial / name
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("BF-MDPROBE "):
                rows.append(json.loads(line.split(" ", 1)[1]))
    return rows


def _trial_dirs(jobs: Path) -> list[Path]:
    return sorted(
        p.parent for p in jobs.rglob("result.json") if p.parent.name != jobs.name
    )


def _reached(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == 200]


def _sdk_run(
    tmp_path: Path,
    task: Path,
    agent: str,
    *,
    sandbox_user,
    flag: bool,
    backend: str = "docker",
):
    """One rollout through the SDK: (the result or the exception, its trial
    folder or None)."""
    import benchflow as bf

    jobs = tmp_path / "jobs"
    if agent == "script":
        script = tmp_path / "probe-agent.sh"
        script.write_text(SCRIPT_AGENT)
        agent = f"script:{script}"
    try:
        result = bf.run_sync(
            bf.RolloutConfig(
                task_path=task,
                agent=agent,
                environment=backend,
                sandbox_user=sandbox_user,
                jobs_dir=jobs,
                allow_cloud_metadata=flag,
            )
        )
    except Exception as exc:  # a refusal raised before the rollout made a result
        return exc, None
    trial = getattr(result, "rollout_dir", None)
    return result, Path(trial) if trial else None


def _refusal_text(outcome) -> str:
    if isinstance(outcome, BaseException):
        return f"{type(outcome).__name__}: {outcome}"
    return str(getattr(outcome, "error", "") or "")


RUNNERS = {
    # name: (agent, sandbox_user); the oracle's solve.sh runs as root, and a
    # root model agent is the CLI row below (a script: agent needs a
    # non-root sandbox user).
    "oracle": ("oracle", "agent"),
    "script-agent-user": ("script", "agent"),
}


# ------------------------------------------- no key, no flag: nothing reaches it


@pytest.mark.parametrize("runner", sorted(RUNNERS))
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_no_phase_reaches_the_metadata_server_without_the_flag(
    tmp_path, probe_image, policy, runner
):
    """No key, no flag: the agent's turn (oracle, a script agent as the
    agent user and as root) and test.sh (as root, nobody and the agent user)
    never get an answer from the metadata server, under every policy."""
    agent, user = RUNNERS[runner]
    task = _task(tmp_path, probe_image, policy=policy)
    outcome, trial = _sdk_run(tmp_path, task, agent, sandbox_user=user, flag=False)
    assert trial is not None, _refusal_text(outcome)
    rows = _probe_rows(trial)
    phases = {r["phase"].split("-")[0] for r in rows}
    assert {"verifier"} <= phases and phases & {"oracle", "agent"}, (
        f"the probe did not run in both phases: {rows} ({_refusal_text(outcome)})"
    )
    assert not _reached(rows), f"reached the metadata server: {_reached(rows)}"


# ----------------------------- the key without the flag is refused (0.10.0 lifted it)


@pytest.mark.parametrize("runner", ["oracle", "script-agent-user"])
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_the_task_key_alone_lifts_nothing_and_is_refused_before_start(
    tmp_path, probe_image, policy, runner
):
    """The task's key without the run's flag: refused before the sandbox
    starts, the reason naming the flag; no phase ran, so nothing reached the
    metadata server. On 0.10.0 the trial ran and the key lifted the block."""
    agent, user = RUNNERS[runner]
    task = _task(tmp_path, probe_image, policy=policy, key="environment")
    outcome, trial = _sdk_run(tmp_path, task, agent, sandbox_user=user, flag=False)
    rows = _probe_rows(trial) if trial is not None else []
    assert not rows and FLAG in _refusal_text(outcome), (
        f"the trial ran (reached: {_reached(rows)}; rows: {rows}; "
        f"outcome: {_refusal_text(outcome)!r})"
    )
    assert "allow_cloud_metadata" in _refusal_text(outcome)


@pytest.mark.parametrize("which", ["metadata-key", "taskmd"])
def test_other_spellings_of_the_key_are_refused_too(tmp_path, probe_image, which):
    """``metadata.x-allow-cloud-metadata = true`` and task.md's
    ``sandbox.allow_cloud_metadata: true`` are the same request."""
    task = (
        _task(tmp_path, probe_image, key="metadata")
        if which == "metadata-key"
        else _taskmd(tmp_path, probe_image)
    )
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    rows = _probe_rows(trial) if trial is not None else []
    assert not rows and FLAG in _refusal_text(outcome), (
        f"the trial ran (reached: {_reached(rows)}; outcome: {_refusal_text(outcome)!r})"
    )


def _cli(tmp_path: Path, task: Path, *flags: str, agent: str = "oracle"):
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    jobs = tmp_path / "jobs-cli"
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME", "PATH")}
    env.update(
        PYTHONUNBUFFERED="1",
        BENCHFLOW_SKIP_PREFLIGHT="1",
        HARBOR_TELEMETRY="off",
        DO_NOT_TRACK="1",
        COLUMNS="200",
    )
    if agent == "mdprobe":
        agents = tmp_path / "agents" / "mdprobe"
        agents.mkdir(parents=True, exist_ok=True)
        (agents / "manifest.toml").write_text(MANIFEST)
        env.update(
            BENCHFLOW_AGENTS_DIR=str(agents.parent),
            BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
            # The stub makes no model call; a restricted policy admits its
            # placeholder route only by the operator's override.
            BENCHFLOW_RESTRICTED_WEB_MODELS="openai/gpt-4.1-mini",
        )
        flags = (
            *flags,
            "--agent-env",
            "OPENAI_API_KEY=placeholder",
            "--agent-env",
            "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:9/v1",
            "--agent-env",
            "HARBOR_TELEMETRY=off",
            "--agent-env",
            "DO_NOT_TRACK=1",
        )
    for name in ("BENCHFLOW_REMOTE_DOCKER_HOST", "BENCHFLOW_REMOTE_DOCKER_SSH_CONFIG"):
        if name in os.environ:
            env[name] = os.environ[name]
    done = subprocess.run(
        [
            str(bench),
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            agent,
            "--sandbox",
            "docker",
            "--concurrency",
            "1",
            "--jobs-dir",
            str(jobs),
            "--retry-attempts",
            "0",
            "--usage-tracking",
            "off",
            *flags,
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return done, jobs


def test_cli_refuses_the_task_key_without_the_flag_before_any_sandbox(
    tmp_path, probe_image
):
    """``bench eval run`` on a task with the key and no flag: a non-zero
    exit naming the flag, and no trial started (no result.json, no probe)."""
    task = _task(tmp_path, probe_image, key="environment")
    done, jobs = _cli(tmp_path, task)
    said = done.stdout + done.stderr
    trials = _trial_dirs(jobs) if jobs.exists() else []
    rows = [r for t in trials for r in _probe_rows(t)]
    assert done.returncode != 0 and FLAG in said and not trials, (
        f"rc={done.returncode}; trials={trials}; reached={_reached(rows)}; "
        f"tail={said[-800:]!r}"
    )


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_a_root_model_agent_never_reaches_it_without_the_flag(
    tmp_path, probe_image, policy
):
    """No key, no flag, for a model agent that runs as root (``--sandbox-user null``): its
    turn and test.sh get no answer under every policy (``denylist`` and
    ``allowlist`` refuse a root agent before it starts)."""
    task = _task(tmp_path, probe_image, policy=policy)
    done, jobs = _cli(tmp_path, task, "--sandbox-user", "null", agent="mdprobe")
    said = done.stdout + done.stderr
    trials = _trial_dirs(jobs) if jobs.exists() else []
    assert trials, f"rc={done.returncode}: {said[-1200:]}"
    if policy in ("denylist", "allowlist"):
        result = json.loads((trials[0] / "result.json").read_text())
        error = str(result.get("error") or "")
        assert "requires a sandbox_user" in error and not _probe_rows(trials[0]), error
        return
    rows = _probe_rows(trials[0])
    phases = {r["phase"].split("-")[0] for r in rows}
    assert {"agent", "verifier"} <= phases, (rows, said[-800:])
    assert any(r["uid"] == 0 and r["phase"] == "agent" for r in rows), rows
    assert not _reached(rows), f"reached the metadata server: {_reached(rows)}"


def test_the_flag_alone_keeps_main_s_rule_on_docker(tmp_path, probe_image):
    """Guard for the run's start line: --allow-cloud-metadata with no task
    key leaves main's rule for every user in place on Docker (the agent's
    turn and test.sh still get no answer)."""
    task = _task(tmp_path, probe_image)
    outcome, trial = _sdk_run(tmp_path, task, "script", sandbox_user="agent", flag=True)
    assert trial is not None, _refusal_text(outcome)
    rows = _probe_rows(trial)
    assert {r["phase"].split("-")[0] for r in rows} >= {"agent", "verifier"}, rows
    assert not _reached(rows), f"reached the metadata server: {_reached(rows)}"


# ----------------------------------------------- key and flag: every record says so


@pytest.mark.parametrize("agent", ["oracle", "script"])
def test_with_the_flag_the_key_takes_effect_and_every_record_says_so(
    tmp_path, probe_image, agent
):
    """Key and flag: the block is lifted (both phases get the answer), and
    the run card (run-card.json and its warning at start), result.json and
    the reward's provenance each name the key and the flag."""
    task = _task(tmp_path, probe_image, key="environment")
    flags = [FLAG]
    if agent == "script":
        script = tmp_path / "probe-agent.sh"
        script.write_text(SCRIPT_AGENT)
        agent = f"script:{script}"
    done, jobs = _cli(tmp_path, task, *flags, agent=agent)
    said = done.stdout + done.stderr
    trials = _trial_dirs(jobs)
    assert done.returncode == 0 and len(trials) == 1, said[-1500:]
    trial = trials[0]
    rows = _probe_rows(trial)
    reached_phases = {r["phase"].split("-")[0] for r in _reached(rows)}
    assert {"verifier"} <= reached_phases and reached_phases & {"oracle", "agent"}, rows

    problems = []
    card = json.loads(next(jobs.rglob("run-card.json")).read_text())
    lifted = card.get("cloud_metadata") or {}
    if FLAG not in json.dumps(lifted) or task.name not in json.dumps(lifted):
        problems.append(f"run-card.json cloud_metadata: {lifted!r}")
    warned = [
        line
        for line in said.splitlines()
        if FLAG in line and "allow_cloud_metadata" in line and task.name in line
    ]
    if not warned:
        problems.append("no start warning naming the task, its key and the flag")
    result = json.loads((trial / "result.json").read_text())
    opt_ins = result.get("isolation_opt_ins") or []
    if not any(
        FLAG in json.dumps(o) and "allow_cloud_metadata" in json.dumps(o)
        for o in opt_ins
    ):
        problems.append(f"result.json isolation_opt_ins: {opt_ins!r}")
    cm = result.get("cloud_metadata") or {}
    if agent != "oracle" and FLAG not in json.dumps(cm.get("lifted_by") or {}):
        problems.append(f"result.json cloud_metadata: {cm!r}")
    inputs = trial / "integrity" / "verifier-inputs.json"
    provenance = (
        json.loads(inputs.read_text()).get("provenance") or {}
        if inputs.is_file()
        else {}
    )
    named = json.dumps(provenance.get("isolation_lifted") or [])
    if FLAG not in named or "allow_cloud_metadata" not in named:
        problems.append(
            f"provenance isolation_lifted: {provenance.get('isolation_lifted')!r}"
        )
    assert not problems, problems


# ------------------------------------------------------------- remote Docker


def _remote_or_skip() -> None:
    if not os.environ.get("BENCHFLOW_REMOTE_DOCKER_HOST"):
        pytest.skip("BENCHFLOW_REMOTE_DOCKER_HOST is not set")


def test_remote_docker_guard(tmp_path, probe_image):
    """Remote Docker, no key and no flag: nothing reaches the server."""
    _remote_or_skip()
    task = _task(tmp_path, probe_image)
    outcome, trial = _sdk_run(
        tmp_path,
        task,
        "oracle",
        sandbox_user="agent",
        flag=False,
        backend="remote-docker",
    )
    assert trial is not None, _refusal_text(outcome)
    rows = _probe_rows(trial)
    assert rows and not _reached(rows), rows


def test_remote_docker_refuses_the_key_alone(tmp_path, probe_image):
    """Remote Docker: the key without the flag is refused."""
    _remote_or_skip()
    task = _task(tmp_path, probe_image, key="environment")
    outcome, trial = _sdk_run(
        tmp_path,
        task,
        "oracle",
        sandbox_user="agent",
        flag=False,
        backend="remote-docker",
    )
    rows = _probe_rows(trial) if trial is not None else []
    assert not rows and FLAG in _refusal_text(outcome), (
        f"the trial ran (reached: {_reached(rows)}; outcome: {_refusal_text(outcome)!r})"
    )


def test_remote_docker_flag_and_key_lift_it(tmp_path, probe_image):
    """Remote Docker: with the key and the flag, the server is reached."""
    _remote_or_skip()
    task = _task(tmp_path, probe_image, key="environment")
    outcome, trial = _sdk_run(
        tmp_path,
        task,
        "oracle",
        sandbox_user="agent",
        flag=True,
        backend="remote-docker",
    )
    assert trial is not None, _refusal_text(outcome)
    rows = _probe_rows(trial)
    assert {r["phase"].split("-")[0] for r in _reached(rows)} >= {
        "oracle",
        "verifier",
    }, rows


# ------------------------ key and flag, under each policy and integrity mode


@pytest.mark.parametrize("integrity", ["audit", "off", "strict"])
@pytest.mark.parametrize("policy", ["public", "no-network", "allowlist"])
def test_the_lift_is_named_under_every_policy_and_integrity_mode(
    tmp_path, probe_image, policy, integrity
):
    """Key and flag under each network policy and integrity mode: the run
    card and result.json name the key and the flag in every case, and the
    reward provenance does wherever the run assesses it (off and audit;
    strict ends the agent's processes and grades in its own sandbox)."""
    task = _task(tmp_path, probe_image, policy=policy, key="environment")
    script = tmp_path / "probe-agent.sh"
    script.write_text(SCRIPT_AGENT)
    done, jobs = _cli(
        tmp_path, task, FLAG, "--integrity", integrity, agent=f"script:{script}"
    )
    said = done.stdout + done.stderr
    trials = _trial_dirs(jobs)
    assert done.returncode == 0 and len(trials) == 1, said[-1500:]
    trial = trials[0]
    problems = []
    card = json.loads(next(jobs.rglob("run-card.json")).read_text())
    if task.name not in json.dumps(card.get("cloud_metadata") or {}):
        problems.append(f"run-card.json cloud_metadata: {card.get('cloud_metadata')!r}")
    result = json.loads((trial / "result.json").read_text())
    named = json.dumps(result.get("isolation_opt_ins") or [])
    if FLAG not in named or "allow_cloud_metadata" not in named:
        problems.append(f"result.json isolation_opt_ins: {named}")
    if integrity != "strict":
        inputs = trial / "integrity" / "verifier-inputs.json"
        provenance = (
            json.loads(inputs.read_text()).get("provenance") or {}
            if inputs.is_file()
            else {}
        )
        if FLAG not in json.dumps(provenance.get("isolation_lifted") or []):
            problems.append(f"provenance: {provenance.get('isolation_lifted')!r}")
    assert not problems, problems


def test_a_task_without_the_key_in_a_granted_job_records_nothing(tmp_path, probe_image):
    """A granted two-task job: only the task that asks is named."""
    root = tmp_path / "tasks"
    root.mkdir()
    asked = _task(root, probe_image, key="environment")
    plain = _task(root, probe_image)
    done, jobs = _cli(tmp_path, root, FLAG)
    said = done.stdout + done.stderr
    trials = {t.name.split("__")[0]: t for t in _trial_dirs(jobs)}
    assert done.returncode == 0 and len(trials) == 2, said[-1500:]
    plain_result = json.loads((trials[plain.name] / "result.json").read_text())
    asked_result = json.loads((trials[asked.name] / "result.json").read_text())
    assert not plain_result.get("isolation_opt_ins"), plain_result.get(
        "isolation_opt_ins"
    )
    assert asked_result.get("isolation_opt_ins"), asked_result


def test_a_separate_verifier_takes_the_grant(tmp_path, probe_image):
    """Key and flag with the verifier in its own sandbox: it runs (the
    factory takes the trial's grant), test.sh reaches the server, and the
    records name the lift."""
    task = _task(tmp_path, probe_image, key="environment")
    done, jobs = _cli(tmp_path, task, FLAG, "--verifier-sandbox", "separate")
    said = done.stdout + done.stderr
    trials = _trial_dirs(jobs)
    assert done.returncode == 0 and len(trials) == 1, said[-1500:]
    result = json.loads((trials[0] / "result.json").read_text())
    assert not result.get("error"), result.get("error")
    assert result.get("isolation_opt_ins"), result
    rows = [r for r in _probe_rows(trials[0]) if r["phase"].startswith("verifier")]
    assert _reached(rows), rows
