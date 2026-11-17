"""On a real Docker daemon: a trial's network policy is in force
before the first process that can open a socket starts, in every phase.

Each test runs one trial through ``bench eval run`` of a small task whose
processes report to a listener on the VM host (``0.0.0.0``, reached through
each Docker network's gateway), as FX609R's probes did: a ``report`` is one
TCP connect carrying a line that names the process, and a ``hold`` opens one
connection at start and writes a line on it every second. The listener
records each line with its arrival time, so a line that arrives shows that
the process had the network at that moment. The policies here (allowlist,
denylist, an ``[agent]`` no-network over an online sandbox) refuse the host,
so a held phase must leave nothing at the listener.

The tests skip where no Docker daemon answers. Everything they start is a
trial's own sandbox, removed by the trial.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests.test_cloud_metadata_failclosed_docker import ACP_SH, MANIFEST, _docker_ready

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

# A process reports to the VM host: the default gateway, the listener's port.
PROBE_SH = r"""#!/bin/bash
# /probe.sh <tag> [once|loop|hold|later]
gw() { while read -r _i dest gwh _r; do if [ "$dest" = 00000000 ]; then
  printf '%d.%d.%d.%d' 0x${gwh:6:2} 0x${gwh:4:2} 0x${gwh:2:2} 0x${gwh:0:2}; return; fi; done < /proc/net/route; }
TAG="$1 uid=$(id -u)"
once() { G=$(gw); [ -n "$G" ] && timeout 2 bash -c "exec 3<>/dev/tcp/$G/__PORT__; echo '$TAG t=$(date +%s.%N)' >&3" 2>/dev/null; }
case "${2:-once}" in
  once) once ;;
  loop) while :; do once; sleep 0.5; done ;;
  later) sleep "${3:-30}"; while :; do once; sleep 0.5; done ;;
  hold) G=$(gw); exec 4<>/dev/tcp/$G/__PORT__ || exit 0
        while :; do echo "$TAG hold t=$(date +%s.%N)" >&4 2>/dev/null || exit 0; sleep 1; done ;;
esac
"""

# main's ENTRYPOINT: reports from its start, as root and as the image's agent
# account (uid 1500), then runs the command it is given.
ENTRY_SH = r"""#!/bin/bash
/probe.sh main-entry-root loop &
/probe.sh main-entry-root hold &
setpriv --reuid=1500 --regid=1500 --clear-groups /probe.sh main-entry-agentuid loop &
setpriv --reuid=1500 --regid=1500 --clear-groups /probe.sh main-entry-agentuid hold &
if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi
"""

SIDECAR = """services:
  side:
    build: .
    command: ["/bin/bash", "-c", "/probe.sh side-entry loop & /probe.sh side-entry hold & exec sleep infinity"]
    healthcheck:
      test: ["CMD-SHELL", "/probe.sh side-healthcheck once || true"]
      interval: 2s
      timeout: 5s
      retries: 3
      start_period: 1s
"""

POLICIES = {
    "allowlist": ('network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n', ""),
    "denylist": ('network_mode = "denylist"\nblocked_hosts = ["example.com"]\n', ""),
    "offline": ("", 'network_mode = "no-network"\n'),
}


class Listener:
    """The VM host's end: every line any process sent, with its arrival."""

    def __init__(self) -> None:
        self.lines: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", 0))
        self._server.listen(256)
        self.port = self._server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(600)
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    with self._lock:
                        self.lines.append((time.time(), raw.decode(errors="replace")))
        except OSError:
            pass
        finally:
            conn.close()

    def seen(self, tag: str, after: float = 0.0) -> list[tuple[float, str]]:
        with self._lock:
            return [(t, x) for t, x in self.lines if x.startswith(tag) and t >= after]


@pytest.fixture(scope="module")
def listener() -> Listener:
    return Listener()


def _task(
    root: Path,
    port: int,
    policy: str,
    *,
    entry: str | None = ENTRY_SH,
    sidecar: bool = False,
    separate: bool = False,
    no_web: bool = False,
    agent: bool = False,
    prompt_sh: str = "#!/bin/sh\n",
    base: str = "ubuntu:24.04",
    extra: str = "",
    agent_user: bool = True,
) -> Path:
    name = f"en698-{policy}-{uuid.uuid4().hex[:8]}"
    task = root / name
    env_dir = task / "environment"
    env_dir.mkdir(parents=True)
    probe = PROBE_SH.replace("__PORT__", str(port))
    (env_dir / "probe.sh").write_text(probe)
    files = ["probe.sh"]
    if entry is not None:
        (env_dir / "entry.sh").write_text(entry)
        files.append("entry.sh")
    if agent:
        (env_dir / "acp.sh").write_text(ACP_SH)
        (env_dir / "prompt.sh").write_text(prompt_sh)
        files += ["acp.sh", "prompt.sh"]
    (env_dir / "Dockerfile").write_text(
        f"FROM {base}\n"
        + (
            "RUN groupadd -g 1500 agent && useradd -m -u 1500 -g 1500 -s /bin/bash agent\n"
            if agent_user
            else ""
        )
        + extra
        + "".join(f"COPY {f} /{f}\n" for f in files)
        + "RUN chmod 755 "
        + " ".join(f"/{f}" for f in files)
        + "\n"
        + (
            "RUN mkdir -p /opt/bfprobe && cp /acp.sh /opt/bfprobe/acp.sh && "
            "cp /prompt.sh /opt/bfprobe/probe.sh && "
            "chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh\n"
            if agent
            else ""
        )
        + "WORKDIR /app\n"
        + ('ENTRYPOINT ["/entry.sh"]\n' if entry is not None else "")
    )
    if sidecar:
        (env_dir / "docker-compose.yaml").write_text(SIDECAR)
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\n/probe.sh oracle-solve-start once; sleep 2; "
        "/probe.sh oracle-solve-later once; date +%s.%N > /logs/agent/phase-end.txt\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "probe.sh").write_text(probe)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\nP=/probe.sh; [ -x $P ] || P=/tests/probe.sh\n"
        "$P verifier-test-start once\necho 1 > /logs/verifier/reward.txt\n"
    )
    env_lines, agent_lines = POLICIES.get(policy, ("", ""))
    if no_web:
        env_lines = "allow_internet = false\n"
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n'
        f"[agent]\ntimeout_sec = 120\n{agent_lines}\n"
        "[verifier]\ntimeout_sec = 120\n"
        + ('sandbox_mode = "separate"\n' if separate else "")
        + f"\n[environment]\n{env_lines}"
    )
    (task / "instruction.md").write_text("Probe task.\n")
    return task


def _bench() -> Path:
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    return bench


def _run(tmp_path: Path, task: Path, agent: str, *flags: str) -> dict:
    """One trial; its result.json (``error``), with the run's start and end
    times and, for a ``script:`` agent, when its script started."""
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
    )
    if "PYTHONPATH" in os.environ:  # a tree under test, not this venv's
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    args = ["--agent", agent]
    if agent == "script":
        script = tmp_path / "op.sh"
        script.write_text(
            "#!/bin/bash\ndate +%s.%N > /logs/agent/script-start.txt\n"
            "/probe.sh agent-script-start once; sleep 12; /probe.sh agent-script-later once\n"
            "date +%s.%N > /logs/agent/phase-end.txt\n"
        )
        script.chmod(0o755)
        args = ["--agent", f"script:{script}", "--sandbox-user", "agent"]
    if agent == "bfprobe":
        agents = tmp_path / "agents" / "bfprobe"
        agents.mkdir(parents=True, exist_ok=True)
        (agents / "manifest.toml").write_text(MANIFEST)
        env.update(
            BENCHFLOW_AGENTS_DIR=str(agents.parent),
            BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
            BENCHFLOW_RESTRICTED_WEB_MODELS="openai/gpt-4.1-mini",
        )
        args += [
            "--usage-tracking",
            "off",
            "--agent-env",
            "OPENAI_API_KEY=placeholder",
            "--agent-env",
            "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:9/v1",
        ]
    jobs = tmp_path / f"jobs-{uuid.uuid4().hex[:6]}"
    started = time.time()
    done = subprocess.run(
        [
            str(_bench()),
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--sandbox",
            "docker",
            "--concurrency",
            "1",
            "--jobs-dir",
            str(jobs),
            "--retry-attempts",
            "0",
            *args,
            *flags,
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    results = [p for p in jobs.rglob("result.json") if "__" in p.parent.name]
    result = json.loads(results[0].read_text()) if results else {}
    starts = list(jobs.rglob("agent/script-start.txt"))
    ends = list(jobs.rglob("agent/phase-end.txt"))
    return {
        "phase_end": float(ends[0].read_text()) if ends else None,
        "error": result.get("error") or "",
        "started": started,
        "ended": time.time(),
        "script_start": float(starts[0].read_text()) if starts else None,
        "stderr": done.stderr[-3000:],
        "jobs": jobs,
        "result": result,
    }


def _agent_phase_end(run: dict) -> float:
    """When the agent's turn ended (solve.sh's or the script's last line):
    every phase before it is the agent policy's."""
    assert run["phase_end"] is not None, ("the turn wrote no end", run["stderr"])
    return run["phase_end"]


# --- N3, S1: main's ENTRYPOINT and the oracle's solve.sh ---------------------


@pytest.mark.parametrize("policy", ["allowlist", "denylist", "offline"])
def test_mains_entrypoint_and_the_oracle_start_under_the_policy(
    tmp_path, listener, policy
):
    """N3: main's ENTRYPOINT processes, root's and the agent account's, start
    under the policy (before: they reached the host from main's release
    until the oracle's arm, 13-16 s). S1: under an [agent] no-network over an
    online sandbox a root oracle's solve.sh is held too (before: it reached
    the host). Nothing reaches the host before test.sh starts."""
    run = _run(tmp_path, _task(tmp_path, listener.port, policy), "oracle")
    assert run["error"] == "", (run["error"], run["stderr"])
    until = _agent_phase_end(run)
    for tag in ("main-entry-root", "main-entry-agentuid", "oracle-solve"):
        early = [x for t, x in listener.seen(tag, run["started"]) if t < until]
        assert early == [], (policy, tag, early[:3])


# --- N1: a sidecar's command and healthcheck --------------------------------


@pytest.mark.parametrize("policy", ["denylist", "offline"])
def test_a_sidecar_starts_under_the_policy(tmp_path, listener, policy):
    """N1: a compose sidecar on the default network, whose command reports
    from its first instruction and whose healthcheck reports too, waits in
    the inert bootstrap until its rules are in (before: 12 reports in 5.8 s,
    and a held connection, until the rollout's cover). Its record lists no
    socket seen before coverage."""
    task = _task(tmp_path, listener.port, policy, entry=None, sidecar=True)
    run = _run(tmp_path, task, "oracle")
    assert run["error"] == "", (run["error"], run["stderr"])
    until = _agent_phase_end(run)
    early = [x for t, x in listener.seen("side-", run["started"]) if t < until]
    assert early == [], (policy, early[:3])
    records = list(run["jobs"].rglob("service-network.json"))
    assert records, "no service-network.json"
    side = [
        s
        for s in json.loads(records[0].read_text())["services"]
        if s.get("service") == "side"
    ]
    assert side and side[0].get("seen_before_coverage") in ([], None, {}), side


# --- N2: rc's allowlist refusal of an outbound sidecar, in the plan ----------


def test_an_allowlist_sidecar_with_a_route_out_is_refused_before_it_runs(
    tmp_path, listener
):
    """N2: under allowlist a sidecar on a network that is not internal is
    refused before any container starts (before: the refusal came at the
    agent phase, after the sidecar had reached the host for 5.8 s). The
    sidecar never reports."""
    task = _task(tmp_path, listener.port, "allowlist", entry=None, sidecar=True)
    run = _run(tmp_path, task, "oracle")
    assert "network_mode='allowlist' needs the task's services kept inside" in (
        run["error"] + run["stderr"]
    ), (run["error"], run["stderr"])
    assert listener.seen("side-", run["started"]) == []


# --- N5: the separate verifier sandbox's ENTRYPOINT --------------------------


@pytest.mark.parametrize("policy", ["allowlist", "denylist"])
def test_a_separate_verifier_sandbox_starts_under_its_policy(
    tmp_path, listener, policy
):
    """N5: the separate verifier sandbox's main ENTRYPOINT starts under the
    verifier's network (before: it reached the host until its verifier
    rules, 13.7-15.2 s). Under allowlist and denylist nothing of the trial
    reaches the host at all."""
    task = _task(tmp_path, listener.port, policy, separate=True)
    run = _run(tmp_path, task, "oracle")
    assert run["error"] == "", (run["error"], run["stderr"])
    seen = listener.seen("main-entry", run["started"])
    assert seen == [], (policy, seen[:3])


# --- S2 and N4: a script: agent's turn ---------------------------------------


@pytest.mark.parametrize("policy", ["denylist", "offline"])
def test_a_script_agents_turn_holds_every_process(tmp_path, listener, policy):
    """S2: during a script: agent's turn main's root processes are held too
    (before: only the script's uid). N4: a connection the agent's account
    held open from before its rules carries nothing during the turn, and
    nothing it queued leaves afterwards (before: its queued lines arrived
    when the rules came out)."""
    run = _run(tmp_path, _task(tmp_path, listener.port, policy), "script")
    assert run["error"] == "", (run["error"], run["stderr"])
    assert run["script_start"] is not None, run["stderr"]
    until = _agent_phase_end(run)
    early = [
        x for t, x in listener.seen("main-entry-root", run["started"]) if t < until
    ]
    assert early == [], (policy, early[:3])
    agent_uid = [
        x
        for t, x in listener.seen("main-entry-agentuid", run["started"])
        if t >= run["script_start"]
    ]
    assert agent_uid == [], (policy, agent_uid[:3])
    assert listener.seen("agent-script", run["started"]) == []


def test_the_uid_rules_reset_a_connection_opened_before_them(tmp_path, listener):
    """N4 at the rules themselves: a process of uid 1500 opens a connection
    to the host and writes a line every second; root then adds the offline
    uid rules (lockdown._agent_egress_firewall_cmd) and, three seconds later,
    takes them out. No line may arrive once they are in, those queued while
    they held included (before: the rules only stalled the connection, and
    its queued lines left when they came out)."""
    from benchflow.sandbox.lockdown import _agent_egress_firewall_cmd

    probe = PROBE_SH.replace("__PORT__", str(listener.port))
    rules = _agent_egress_firewall_cmd("agent", refuse_dns=True, uid=1500)
    tag = f"n4-{uuid.uuid4().hex[:8]}"
    name = f"bf-en698-n4-{uuid.uuid4().hex[:8]}"
    script = (
        "set -e; apt-get update -qq >/dev/null && apt-get install -y -qq iptables >/dev/null; "
        'printf "%s" "$PROBE" > /tmp/probe.sh; chmod 755 /tmp/probe.sh; '
        f"setpriv --reuid=1500 --regid=1500 --clear-groups /tmp/probe.sh {tag} hold & "
        "sleep 3; date +%s.%N > /tmp/armed; "
        f"{rules}; cat /tmp/armed; sleep 4; "
        "iptables -F OUTPUT; ip6tables -F OUTPUT 2>/dev/null || true; sleep 4"
    )
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--cap-add",
            "NET_ADMIN",
            "-e",
            f"PROBE={probe}",
            "ubuntu:24.04",
            "bash",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    armed = float(done.stdout.strip().splitlines()[-1])
    before = listener.seen(tag)
    assert any(t < armed for t, _ in before), "the connection never worked"
    after = [x for t, x in before if t >= armed + 0.5]
    assert after == [], after[:3]


# --- the lead's 05:42Z rule: a proof that cannot be made is no breach ----------

# Refuses BenchFlow's helper the re-proof's check (as a loaded host's slow
# helper fails to run it), and passes every other call on.
FAILING_CHECK_DOCKER = """#!/bin/sh
for a in "$@"; do case "$a" in *"BF rule-check"*) echo "shim: check refused" >&2; exit 125;; esac; done
exec __DOCKER__ "$@"
"""


def test_a_reproof_that_cannot_be_made_under_load_is_not_proven_never_a_breach(
    tmp_path, listener
):
    """Four sidecar trials at once (-n 4-style load), and every re-proof check
    BenchFlow's helper is asked to make fails, as a loaded host's slow helper
    does (FXKILL's cooperbench dspy oracle: "service agent1 (start): not
    proven"). Each trial ends "network policy not proven", unscored as
    BenchFlow's infrastructure, never "network policy breached"; and nothing
    of the trial reached the host. A real breach (a rule gone, a connect
    through) stays one: tests/test_policy_reproof.py."""
    from benchflow._utils.scoring import VERIFIER_INFRA, classify_verifier_error

    shim = tmp_path / "shim"
    shim.mkdir()
    real = shutil.which("docker")
    assert real is not None
    (shim / "docker").write_text(FAILING_CHECK_DOCKER.replace("__DOCKER__", real))
    (shim / "docker").chmod(0o755)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for _ in range(4):
        _task(tasks, listener.port, "denylist", entry=None, sidecar=True)
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=f"{shim}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
    )
    jobs = tmp_path / "jobs"
    started = time.time()
    done = subprocess.run(
        [
            str(_bench()),
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--concurrency",
            "4",
            "--jobs-dir",
            str(jobs),
            "--retry-attempts",
            "0",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    results = [p for p in jobs.rglob("result.json") if "__" in p.parent.name]
    assert len(results) == 4, done.stderr[-3000:]
    for path in results:
        result = json.loads(path.read_text())
        ending = result.get("verifier_error") or ""
        assert ending.startswith("network policy not proven: "), (
            ending,
            result.get("error"),
        )
        assert "network policy breached" not in ending
        assert classify_verifier_error(ending) == VERIFIER_INFRA
        assert "the helper did not run the check" in ending
        assert result.get("rewards") in (None, {})
    assert listener.seen("side-", started) == []


# --- the uid arm before the model proxy (the lead's 07:13Z go) ----------------

# The stub agent's prompt: one request to the model endpoint BenchFlow wired
# for it (its in-sandbox proxy, on loopback); the status line it got.
PROMPT_MODEL_SH = r"""#!/bin/bash
u=$(env | sed -n "s/^[A-Z_]*BASE_URL=\(http:\/\/127\.0\.0\.1:[0-9]*\).*/\1/p" | head -1)
hp=${u#*://}; hp=${hp%%/*}; h=${hp%:*}; p=${hp##*:}
line=none
if [ -n "$p" ] && exec 3<>/dev/tcp/$h/$p; then
  printf 'GET /v1/models HTTP/1.0\r\nHost: %s\r\n\r\n' "$hp" >&3
  read -r line <&3 || line=empty
fi
echo "model endpoint $hp: $line" > /logs/agent/model.txt
"""


def _timed_bench(
    tmp_path: Path, task: Path, *, manifest: str = MANIFEST
) -> tuple[list[tuple[float, str]], Path]:
    """The stub agent's trial, each stderr line with its arrival time."""
    agents = tmp_path / "agents" / "bfprobe"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "manifest.toml").write_text(manifest)
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
        BENCHFLOW_AGENTS_DIR=str(agents.parent),
        BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
        BENCHFLOW_RESTRICTED_WEB_MODELS="openai/gpt-4.1-mini",
    )
    if "PYTHONPATH" in os.environ:  # a tree under test, not this venv's
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    jobs = tmp_path / f"jobs-{uuid.uuid4().hex[:6]}"
    proc = subprocess.Popen(
        [
            str(_bench()),
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "bfprobe",
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
            "--sandbox-user",
            "agent",
            "--agent-env",
            "OPENAI_API_KEY=placeholder",
        ],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[tuple[float, str]] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append((time.time(), line.rstrip("\n")))
    proc.wait(timeout=900)
    return lines, jobs


def _first(lines: list[tuple[float, str]], words: str) -> float | None:
    return next((t for t, x in lines if words in x), None)


def test_the_uid_rules_go_in_before_the_model_proxy_which_the_agent_still_reaches(
    tmp_path, listener
):
    """The agent's uid rules are in before BenchFlow starts its model proxy,
    and the agent still reaches the proxy on loopback at its prompt (the
    proxy answers its request)."""
    task = _task(
        tmp_path,
        listener.port,
        "nonet",
        entry=None,
        no_web=True,
        agent=True,
        prompt_sh=PROMPT_MODEL_SH,
    )
    lines, jobs = _timed_bench(tmp_path, task)
    arm = _first(lines, "Sandbox-user egress firewall active")
    proxy = _first(lines, "Sandbox LiteLLM proxy listening")
    assert arm is not None and proxy is not None, [x for _, x in lines][-40:]
    assert arm < proxy, (arm, proxy)
    answers = list(jobs.rglob("agent/model.txt"))
    assert answers, [x for _, x in lines][-40:]
    assert " HTTP/1." in answers[0].read_text(), answers[0].read_text()


@pytest.mark.parametrize("who", ["agentuid", "root"])
def test_a_window_process_has_no_connection_after_the_refusal(tmp_path, listener, who):
    """The looping-connect check (the lead's condition 3 on R1): a process
    the image's ENTRYPOINT starts at main's release connects to the host
    every half second. Root's reaches it (so the check proves something);
    the agent's own account's never does, as its uid rules are in before
    main's release (M1). R1 then refuses the trial, unscored, before the
    agent's process starts (the stub agent writes ``launched`` first thing
    and never does), naming the process, and not one connection arrives
    after the refusal."""
    run_as = (
        "setpriv --reuid=1500 --regid=1500 --clear-groups " if who == "agentuid" else ""
    )
    entry = (
        f"#!/bin/bash\n{run_as}/probe.sh window-{who} loop &\n"
        'if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi\n'
    )
    task = _task(tmp_path, listener.port, "nonet", entry=entry, no_web=True, agent=True)
    started = time.time()
    lines, jobs = _timed_bench(tmp_path, task)
    text = [x for _, x in lines]
    refused = _first(lines, "main must be quiet when a model agent starts")
    assert refused is not None, text[-40:]
    seen = listener.seen(f"window-{who}", started)
    if who == "root":
        assert any(t <= refused for t, _ in seen), (
            "never reached the host",
            text[-40:],
        )
    else:
        assert seen == [], ("the agent's account reached the host", seen[:3])
    late = [(round(t - refused, 2), x) for t, x in seen if t > refused]
    assert late == [], (who, len(late), late[:3])
    assert not list(jobs.rglob("agent/launched")), "the agent's process started"
    results = [p for p in jobs.rglob("result.json") if "__" in p.parent.name]
    assert results, text[-40:]
    result = json.loads(results[0].read_text())
    uid = 1500 if who == "agentuid" else 0
    assert f"(uid {uid}): /bin/bash /probe.sh window-{who} loop" in result["error"]
    assert result.get("error_category") == "sandbox_setup", result.get("error_category")
    assert result.get("rewards") in (None, {}), result.get("rewards")


# --- R1: main is quiet when a model agent starts -------------------------------

# An ENTRYPOINT that leaves one sleep running, as root or as the agent's own
# account, then execs its command.
SLEEP_ENTRY_SH = r"""#!/bin/bash
__RUN_AS__sleep 600 &
if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi
"""
# The same sleep, run from BenchFlow's own program folder with BenchFlow's
# own environment marker set: still the task's.
DISGUISED_ENTRY_SH = r"""#!/bin/bash
env _BENCHFLOW_TASK_ENV_SAVED=1 BENCHFLOW_ACP_LINE_LIMIT=0 /opt/benchflow/bin/sleep 600 &
if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi
"""


def _refused_unscored(run: dict, *named: str) -> None:
    error = run["error"]
    assert "main must be quiet when a model agent starts" in error, (
        error,
        run["stderr"],
    )
    for words in named:
        assert words in error, (words, error)
    assert "drop the image's ENTRYPOINT, or make it exec its command" in error
    result = run["result"]
    assert result.get("error_category") == "sandbox_setup", result.get("error_category")
    assert result.get("rewards") in (None, {}), result.get("rewards")
    assert not list(run["jobs"].rglob("agent/launched")), "the agent's process started"


@pytest.mark.parametrize("who", ["root", "agentuid"])
def test_r1_refuses_an_entrypoint_that_leaves_a_process(tmp_path, listener, who):
    """(a) An ENTRYPOINT that starts ``sleep 600 &`` and then execs its
    command: refused, unscored, before the agent's process starts, the sleep
    named. (b) The same sleep as the agent's own account: refused too (its
    uid rules would hold it, and main must still be quiet)."""
    run_as = (
        "setpriv --reuid=1500 --regid=1500 --clear-groups " if who == "agentuid" else ""
    )
    entry = SLEEP_ENTRY_SH.replace("__RUN_AS__", run_as)
    task = _task(tmp_path, listener.port, "nonet", entry=entry, no_web=True, agent=True)
    run = _run(tmp_path, task, "bfprobe", "--sandbox-user", "agent")
    uid = 1500 if who == "agentuid" else 0
    _refused_unscored(run, f"(uid {uid}): sleep 600", "1 task process is")


def test_r1_runs_an_entrypoint_that_only_execs_its_command(tmp_path, listener):
    """(c) node:22-bookworm-slim's own ENTRYPOINT (docker-entrypoint.sh),
    which execs its command and leaves nothing running: the trial runs, and
    the agent gets its prompt."""
    task = _task(
        tmp_path,
        listener.port,
        "nonet",
        entry=None,
        no_web=True,
        agent=True,
        base="node:22-bookworm-slim",
    )
    run = _run(tmp_path, task, "bfprobe", "--sandbox-user", "agent")
    assert "main must be quiet" not in run["error"], run["error"]
    assert run["error"] == "", (run["error"], run["stderr"])
    assert list(run["jobs"].rglob("agent/launched")), run["stderr"]


def test_r1_counts_a_process_dressed_as_benchflows_as_the_tasks(tmp_path, listener):
    """(d) A task process that sets BenchFlow's own environment marker and
    runs from BenchFlow's program folder is still a task process: BenchFlow
    knows its own from the host's record, not from what a process says."""
    task = _task(
        tmp_path,
        listener.port,
        "nonet",
        entry=DISGUISED_ENTRY_SH,
        no_web=True,
        agent=True,
        extra="RUN mkdir -p /opt/benchflow/bin && cp /bin/sleep /opt/benchflow/bin/sleep\n",
    )
    run = _run(tmp_path, task, "bfprobe", "--sandbox-user", "agent")
    _refused_unscored(run, "(uid 0): /opt/benchflow/bin/sleep 600")


# --- every start path (R698 07:24Z) ------------------------------------------

# One start path in its own Python: the SDK's (Environment.from_task, then
# Runtime.execute starts it), the equivalence battery's, acceptance-live's.
START_PATH_PY = r"""
import asyncio, json, sys, time
from pathlib import Path


async def main(kind, task, jobs):
    task, jobs = Path(task), Path(jobs)
    out = {"started": time.time()}
    if kind == "sdk":
        from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

        try:
            env = Environment.from_task(task, sandbox="docker", jobs_dir=jobs)
        except TypeError:  # a release without jobs_dir
            env = Environment.from_task(task, sandbox="docker")
        try:
            result = await Runtime(
                env, Agent("oracle", model=""), RuntimeConfig(jobs_dir=str(jobs))
            ).execute()
            out["error"] = result.error or ""
            out["rewards"] = result.rewards
        finally:
            await env.stop()
    elif kind == "equivalence":
        from benchflow.task.equivalence import check_equivalence_async

        report = await check_equivalence_async(task, sandbox_type="docker", max_variants=2)
        out["error"] = report.reason if report.status == "error" else ""
        out["status"] = report.status
        out["oracle_reward"] = report.oracle_reward
    else:
        from benchflow.task import acceptance_live as al

        ws = jobs / "ws"
        ws.mkdir(parents=True)
        (ws / "README").write_text("x\n")
        result = await al._run_single_case(
            task,
            sandbox_type="docker",
            workspace=al.LiveAcceptanceWorkspace(source="current-worktree", target="/app/ws"),
            staged_worktree=ws,
            case=al.LiveAcceptanceCase(
                name="c",
                case_type="reference",
                command="true",
                reruns=1,
                expect=al.LiveAcceptanceExpectation(),
            ),
            run_index=1,
        )
        out["error"] = result.error or ""
        out["reward"] = result.reward
    out["ended"] = time.time()
    print("RESULT " + json.dumps(out, default=str), flush=True)


asyncio.run(main(*sys.argv[1:4]))
"""


def _start_path(tmp_path: Path, kind: str, task: Path, python: str = "") -> dict:
    """One start path's run, from its own Python (*python*, default this one)."""
    script = tmp_path / "start_path.py"
    script.write_text(START_PATH_PY)
    jobs = tmp_path / f"jobs-{uuid.uuid4().hex[:6]}"
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
    )
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    done = subprocess.run(
        [python or sys.executable, str(script), kind, str(task), str(jobs)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
    return {**json.loads(lines[-1][len("RESULT ") :]), "stderr": done.stderr[-3000:]}


@pytest.mark.parametrize("kind", ["sdk", "equivalence", "acceptance"])
def test_every_start_path_holds_mains_entrypoint(tmp_path, listener, kind):
    """R698: Runtime.execute starting the sandbox it was handed, the
    equivalence battery and acceptance-live started it without the start
    holds, so main's ENTRYPOINT, root's processes and the agent account's,
    reached the host from release (#308's SDK path: 11 connects each). Now
    each start is start_sandbox's: nothing of main's reaches the host during
    the whole run."""
    run = _start_path(tmp_path, kind, _task(tmp_path, listener.port, "denylist"))
    assert run["error"] == "", (run["error"], run["stderr"])
    for tag in ("main-entry-root", "main-entry-agentuid"):
        seen = [x for t, x in listener.seen(tag, run["started"]) if t <= run["ended"]]
        assert seen == [], (kind, tag, len(seen), seen[:3])


@pytest.mark.parametrize("kind", ["sdk", "equivalence", "acceptance"])
def test_every_start_path_holds_a_sidecar(tmp_path, listener, kind):
    """The same for a compose sidecar's command and healthcheck (#308's SDK
    path: 8 connects)."""
    task = _task(tmp_path, listener.port, "denylist", entry=None, sidecar=True)
    run = _start_path(tmp_path, kind, task)
    assert run["error"] == "", (run["error"], run["stderr"])
    seen = [x for t, x in listener.seen("side-", run["started"]) if t <= run["ended"]]
    assert seen == [], (kind, len(seen), seen[:3])


# --- readiness execs end at their timeout --------------------------------------

# main's image healthcheck: its first run starts a process and then hangs
# past its 3 s timeout; the next one answers.
HANGING_HEALTHCHECK = (
    "HEALTHCHECK --interval=1s --timeout=3s --start-period=0s --retries=3 CMD "
    "if [ -e /tmp/bf-hc-ok ]; then exit 0; fi; touch /tmp/bf-hc-ok; "
    "sleep 3662 & exec sleep 3661\n"
)
# The oracle lists what still runs of the healthcheck when its turn starts.
LEFTOVER_SOLVE_SH = r"""#!/bin/bash
for d in /proc/[0-9]*; do
  a=$(tr '\000' ' ' < $d/cmdline 2>/dev/null)
  case "$a" in "sleep 366"*) echo "${d##*/} $a" >> /logs/agent/leftover.txt;; esac
done
touch /logs/agent/scanned
"""


def test_a_readiness_command_past_its_timeout_is_not_running_afterwards(
    tmp_path, listener
):
    """Item 2: main's healthcheck command (run by the inert bootstrap's
    wait_healthy before release), still running at its 3 s timeout, ends
    with the process it started; when the run's turn starts, neither runs.
    Before: both ran on in main for the rest of the trial."""
    task = _task(
        tmp_path,
        listener.port,
        "nonet",
        entry=None,
        no_web=True,
        extra=HANGING_HEALTHCHECK,
    )
    (task / "solution" / "solve.sh").write_text(LEFTOVER_SOLVE_SH)
    run = _run(tmp_path, task, "oracle")
    assert run["error"] == "", (run["error"], run["stderr"])
    assert list(run["jobs"].rglob("agent/scanned")), run["stderr"]
    leftover = [p.read_text() for p in run["jobs"].rglob("agent/leftover.txt")]
    assert leftover == [], leftover


# --- M1: the uid rules right after release, before the install ----------------

# The stub agent whose install reports to the host as the agent's account
# and as root: root's install keeps the network, the agent's uid is held.
INSTALL_PROBE_MANIFEST = MANIFEST.replace(
    'install_cmd = "test -x /opt/bfprobe/acp.sh"',
    'install_cmd = "setpriv --reuid=1500 --regid=1500 --clear-groups /probe.sh '
    'install-agentuid once; /probe.sh install-root once; test -x /opt/bfprobe/acp.sh"',
)


def test_the_agent_uid_is_held_from_its_install(tmp_path, listener):
    """M1: the agent's uid rules go in right after main's release, before
    the agent's install: a connect the install makes as the agent's account
    never reaches the host, while root's (the install's own) does. Before:
    the rules went in after the install and the sandbox user's setup, and
    the agent's account reached the host from the install."""
    assert INSTALL_PROBE_MANIFEST != MANIFEST
    task = _task(tmp_path, listener.port, "nonet", entry=None, no_web=True, agent=True)
    started = time.time()
    lines, _jobs = _timed_bench(tmp_path, task, manifest=INSTALL_PROBE_MANIFEST)
    text = [x for _, x in lines]
    assert listener.seen("install-root", started), ("no install probe", text[-40:])
    assert listener.seen("install-agentuid", started) == [], text[-40:]


# An ENTRYPOINT whose processes report to the host for 4 s from main's
# release, as the agent's account and as root, then end (gone long before
# the agent starts, so R1 has nothing to refuse).
RELEASE_ENTRY_SH = r"""#!/bin/bash
setpriv --reuid=1500 --regid=1500 --clear-groups timeout 4 /probe.sh release-agentuid loop &
timeout 4 /probe.sh release-root loop &
if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi
"""


@pytest.mark.parametrize("policy", ["nonet", "denylist"])
def test_the_agent_uid_is_held_from_mains_release(tmp_path, listener, policy):
    """M1 under no-network and denylist: the agent's uid rules are in
    before main's release, so a process of the agent's account that main's
    ENTRYPOINT starts never reaches the host, from its first instruction;
    root's does (the rules hold the agent's uid only; so the probe works).
    Before: the rules went in right after release, and the agent's account
    reached the host in that window."""
    task = _task(
        tmp_path,
        listener.port,
        policy,
        entry=RELEASE_ENTRY_SH,
        no_web=policy == "nonet",
        agent=True,
    )
    started = time.time()
    lines, _jobs = _timed_bench(tmp_path, task)
    text = [x for _, x in lines]
    assert listener.seen("release-root", started), ("no release probe", text[-40:])
    assert listener.seen("release-agentuid", started) == [], text[-40:]


# --- L4: a verifier-only rollout starts under the same holds -------------------


def test_a_regrade_starts_main_under_the_policy(tmp_path, listener):
    """L4: a regrade's fresh sandbox (a verifier-only rollout) holds main's
    ENTRYPOINT from release as the run's own oracle trial did: nothing of
    main's reaches the host during the regrade. Before: the oracle's run held
    it, and the regrade's sandbox let root's and the agent account's
    processes reach the host until the verifier's rules."""
    task = _task(tmp_path, listener.port, "denylist")
    run = _run(tmp_path, task, "oracle", "--freeze-workspace")
    assert run["error"] == "", (run["error"], run["stderr"])
    early = listener.seen("main-entry", run["started"])
    assert early == [], early[:3]
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
    )
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    trials = [
        p.parent for p in run["jobs"].rglob("result.json") if "__" in p.parent.name
    ]
    assert trials, run["stderr"]
    started = time.time()
    done = subprocess.run(
        [str(_bench()), "eval", "regrade", str(trials[0].parent), "--json"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    ended = time.time()
    regrades = list(run["jobs"].rglob("regrade.json"))
    assert regrades, (done.returncode, done.stdout[-2000:], done.stderr[-3000:])
    seen = [x for t, x in listener.seen("main-entry", started) if t <= ended]
    assert seen == [], (len(seen), seen[:3])


# --- L1: the plan reads the Compose project as the run resolves it --------------

INCLUDED_SIDECAR = """services:
  side:
    image: ubuntu:24.04
    command: ["sleep", "infinity"]
"""


def test_an_included_allowlist_sidecar_with_a_route_out_is_refused_in_the_plan(
    tmp_path, listener
):
    """L1: under allowlist a sidecar the task's compose file only
    ``include``s, on a network that is not internal, is refused by the dry
    run, as by the run (before: the plan read the raw file, saw no service,
    and passed; the run refused it at its start)."""
    task = _task(tmp_path, listener.port, "allowlist", entry=None)
    env_dir = task / "environment"
    (env_dir / "sidecar.yaml").write_text(INCLUDED_SIDECAR)
    (env_dir / "docker-compose.yaml").write_text("include:\n  - sidecar.yaml\n")
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
    )
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    done = subprocess.run(
        [
            str(_bench()),
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--dry-run",
            "--jobs-dir",
            str(tmp_path / "jobs-dry"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = done.stdout + done.stderr
    assert "network_mode='allowlist' needs the task's services kept inside" in out, (
        done.returncode,
        out[-3000:],
    )


# --- M1: an ENTRYPOINT's own useradd keeps its ids (R698B) ----------------------

# The agent's prompt records the ids the ENTRYPOINT's users got and its own.
PROMPT_IDS_SH = r"""#!/bin/sh
{ echo "app $(id -u app 2>/dev/null)"; echo "svc $(id -u svc 2>/dev/null)";
  echo "agent $(id -u)"; echo "data $(stat -c %u /data 2>/dev/null)"; } > /logs/agent/ids.txt
"""
USERADD_ENTRY = {
    "app": "useradd -m app\nmkdir -p /data\nchown 1000:1000 /data\n",
    "uid-1000": "useradd -u 1000 svc\n",
}


@pytest.mark.parametrize("shape", sorted(USERADD_ENTRY))
@pytest.mark.parametrize("policy", ["nonet", "denylist"])
def test_an_entrypoints_own_useradd_gets_the_ids_it_got_before(
    tmp_path, listener, shape, policy
):
    """R698B: on an image without the sandbox user, the user BenchFlow makes
    before the ENTRYPOINT runs (to put its uid rules in before release) takes
    ids above the image's regular range, so the ENTRYPOINT's own `useradd`
    gets what it got before: `useradd -m app` is uid 1000 and the
    `chown 1000:1000 /data` it makes is app's, and `useradd -u 1000 svc`
    succeeds and the trial runs. Before: the agent took 1000 (app became
    1001 and /data the agent's), and with -u 1000 the ENTRYPOINT failed and
    the trial never started."""
    entry = (
        "#!/bin/bash\nset -e\n"
        + USERADD_ENTRY[shape]
        + 'if [ "$#" -gt 0 ]; then exec "$@"; else exec sleep infinity; fi\n'
    )
    task = _task(
        tmp_path,
        listener.port,
        policy,
        entry=entry,
        no_web=policy == "nonet",
        agent=True,
        prompt_sh=PROMPT_IDS_SH,
        base="python:3.12-slim",
        agent_user=False,
    )
    lines, jobs = _timed_bench(tmp_path, task)
    found = list(jobs.rglob("agent/ids.txt"))
    assert found, [x for _, x in lines][-40:]
    ids = dict(
        line.split(" ", 1) if " " in line else (line, "")
        for line in found[0].read_text().split("\n")
        if line
    )
    assert int(ids["agent"]) > 60000, ids
    if shape == "app":
        assert ids["app"] == "1000" and ids["data"] == "1000", ids
    else:
        assert ids["svc"] == "1000", ids


# --- R698B N1 + N2: a PID 1 dressed as the keep-alive --------------------------

# PID 1 that says it is `sleep infinity` (bash's `exec -a`) and connects to
# the host every half second with bash builtins only, so it has no child a
# scan could catch instead.
DRESSED_PID1_SH = r"""#!/bin/bash
gw() { while read -r _i dest gwh _r; do if [ "$dest" = 00000000 ]; then
  printf -v G '%d.%d.%d.%d' 0x${gwh:6:2} 0x${gwh:4:2} 0x${gwh:2:2} 0x${gwh:0:2}; return; fi; done < /proc/net/route; }
exec 3<>/app/tick
while :; do gw
  { exec 4<>/dev/tcp/$G/__PORT__ && echo "pid1-dressed uid=$UID t=$EPOCHREALTIME" >&4; exec 4>&-; } 2>/dev/null
  read -t 0.5 -u 3
done
"""
DRESSED_ENTRY_SH = (
    "#!/bin/bash\ncd /app\nmkfifo /app/tick\nexec -a sleep bash infinity\n"
)


def test_a_pid_1_dressed_as_the_keep_alive_is_refused_and_stopped(tmp_path, listener):
    """R698B N1 + N2: a root PID 1 whose command line is `sleep infinity`
    but whose program is bash, connecting to the host, is a task process:
    the trial is refused before the agent starts, naming it, and main is
    stopped from the host, so not one connection arrives after the refusal.
    Before: it passed as the keep-alive and connected through the turn."""
    task = _task(
        tmp_path,
        listener.port,
        "nonet",
        entry=DRESSED_ENTRY_SH,
        no_web=True,
        agent=True,
        extra="COPY infinity /app/infinity\n",
    )
    (task / "environment" / "infinity").write_text(
        DRESSED_PID1_SH.replace("__PORT__", str(listener.port))
    )
    started = time.time()
    lines, jobs = _timed_bench(tmp_path, task)
    text = [x for _, x in lines]
    refused = _first(lines, "main must be quiet when a model agent starts")
    assert refused is not None, text[-40:]
    assert any("pid 1 (uid 0): sleep infinity" in x for x in text), text[-40:]
    seen = listener.seen("pid1-dressed", started)
    assert any(t <= refused for t, _ in seen), ("never reached the host", text[-40:])
    late = [(round(t - refused, 2), x) for t, x in seen if t > refused]
    assert late == [], (len(late), late[:3])
    assert not list(jobs.rglob("agent/launched")), "the agent's process started"
