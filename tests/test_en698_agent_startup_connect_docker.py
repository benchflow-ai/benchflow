"""regression: an agent's start-up connection never gets out.

An ACP agent's process can connect out the moment it starts, before it reads
its first instruction. BenchFlow's rules for the agent's uid (an ``[agent]``
no-network policy, an allowlist) must be in force by then. rel-0.9.x before
#309 started the agent and added them about 1.3 s later (MGR5), and a
connection opened in that window outlived them.

The stand-in agent here connects to a listener on the VM host as its very
first act, before it reads stdin, keeps that connection, writes on it again
at its prompt and opens a fresh one then. The verdict is the host's, never a
file the sandbox wrote:

- the listener, which records every connection and line it gets, gets none
  from the trial;
- dockerd's own events show the exec that adds the agent's uid rules ending
  before the exec that launches the agent starts.

The same agent under an open policy is the control: its start-up connection
reaches the listener, so the silence above is the rules', not the network's.

The launch fallback too: an image in which the agent's user cannot be made
before main's release, so nothing is armed then; the user is made
at its setup and its rules go in at the launch. On rel-0.9.x #309 that meant
after the ACP handshake (WIN698).
Both of BenchFlow's Docker paths (the Engine API's and Compose's) run it.
The tests skip where no Docker daemon answers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests.test_cloud_metadata_failclosed_docker import _docker_ready

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

# The agent's uid: unusual, so its rules are told from other trials' rules.
AGENT_UID = 1729

# The stand-in ACP agent. Before anything else, before a byte of stdin: a
# connection to the listener on the default route's gateway (the VM host),
# with a line on it. At the prompt: a line on that connection and a fresh
# connection. Nothing of this goes to stdout, which is the ACP channel.
ACP_SH = r"""#!/bin/bash
while read -r _i d g _r; do [ "$d" = 00000000 ] && h=$g && break; done < /proc/net/route
gw=$(printf '%d.%d.%d.%d' 0x${h:6:2} 0x${h:4:2} 0x${h:2:2} 0x${h:0:2})
exec 3<>/dev/tcp/$gw/__PORT__ 2>/dev/null && echo "startup __TAG__ uid=$(id -u)" 2>/dev/null >&3
while IFS= read -r line; do
  id=$(printf '%s' "$line" | grep -o '"id": *[0-9][0-9]*' | head -1 | grep -o '[0-9][0-9]*$')
  m=$(printf '%s' "$line" | grep -o '"method": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$m" in
    initialize) r='{"protocolVersion":1,"agentInfo":{"name":"bfstartup","version":"1"},"agentCapabilities":{"loadSession":false,"promptCapabilities":{"image":false,"audio":false,"embeddedContext":false}},"authMethods":[]}';;
    session/new) r='{"sessionId":"bfstartup"}';;
    session/prompt)
      echo "prompt __TAG__" 2>/dev/null >&3
      { exec 4<>/dev/tcp/$gw/__PORT__ && echo "fresh __TAG__" >&4; exec 4>&-; } 2>/dev/null
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"bfstartup","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"done"}}}}\n{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"bfstartup","update":{"sessionUpdate":"tool_call","toolCallId":"t1","title":"connect","kind":"execute","status":"completed","rawInput":{}}}}\n'
      r='{"stopReason":"end_turn"}';;
    session/set_config_option) r='{"configOptions":[]}';;
    *) r='{}';;
  esac
  [ -n "$id" ] && printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$r"
done
"""

MANIFEST = """contract_version = "1.0"
name = "bfstartup"
default_model = "openai/gpt-4.1-mini"
install_cmd = "test -d /opt/bfstartup"
launch_cmd = "/opt/bfstartup/acp.sh"
"""

# The policies: none (the control), the agent's own no-network, an allowlist.
POLICIES = {
    "open": ("", ""),
    "no-network": ("", 'network_mode = "no-network"\n'),
    "allowlist": ('network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n', ""),
}


class Listener:
    """The VM host's end: every connection and line from a Docker bridge
    peer, with its arrival time (the host's clock)."""

    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", 0))
        self._server.listen(64)
        self.port = self._server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _note(self, text: str) -> None:
        with self._lock:
            self.events.append((time.time(), text))

    def _accept(self) -> None:
        while True:
            try:
                conn, peer = self._server.accept()
            except OSError:
                return
            self._note(f"accept {peer[0]}:{peer[1]}")
            threading.Thread(target=self._serve, args=(conn, peer), daemon=True).start()

    def _serve(self, conn: socket.socket, peer: tuple[str, int]) -> None:
        buf = b""
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._note(f"line {peer[1]} {line.decode(errors='replace')[:200]}")

    def seen(self, tag: str) -> list[tuple[float, str]]:
        """Every connection and every line of this trial (its tag)."""
        with self._lock:
            lines = [
                (t, x) for t, x in self.events if x.startswith("line ") and tag in x
            ]
            accepts = [(t, x) for t, x in self.events if x.startswith("accept ")]
        return accepts + lines

    def close(self) -> None:
        self._server.close()


@pytest.fixture
def listener():
    lst = Listener()
    yield lst
    lst.close()


def _task(
    root: Path, port: int, policy: str, tag: str, *, fallback: bool = False
) -> Path:
    """*fallback*: an image without the agent's user whose login.defs
    leaves no id above its regular range (UID_MAX 65533), so the user cannot
    be made before main's release (lockdown.arm_agent_uid_before_its_proxy
    arms nothing) and is made at its setup, from uid 1000."""
    task = root / f"en698-startup-{policy}-{tag}"
    env_dir = task / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "acp.sh").write_text(
        ACP_SH.replace("__PORT__", str(port)).replace("__TAG__", tag)
    )
    user = (
        "RUN sed -i 's/^UID_MAX.*/UID_MAX 65533/' /etc/login.defs\n"
        if fallback
        else f"RUN groupadd -g {AGENT_UID} agent && "
        f"useradd -m -u {AGENT_UID} -g {AGENT_UID} -s /bin/bash agent\n"
    )
    (env_dir / "Dockerfile").write_text(
        # python3 for the allowlist's egress proxy.
        "FROM python:3.12-slim\n" + user + "COPY acp.sh /opt/bfstartup/acp.sh\n"
        "RUN chmod 755 /opt/bfstartup /opt/bfstartup/acp.sh\n"
        "WORKDIR /app\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    env_lines, agent_lines = POLICIES[policy]
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n'
        f"[agent]\ntimeout_sec = 120\n{agent_lines}\n"
        "[verifier]\ntimeout_sec = 120\n\n"
        f"[environment]\n{env_lines}"
    )
    (task / "instruction.md").write_text("Say done.\n")
    return task


def _bench() -> Path:
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    return bench


def _trial(tmp_path: Path, task: Path, path: str) -> dict:
    """One trial of the stand-in agent through ``bench eval run``, with
    dockerd's events recorded on the host for its whole run."""
    agents = tmp_path / "agents" / "bfstartup"
    agents.mkdir(parents=True)
    (agents / "manifest.toml").write_text(MANIFEST)
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
        BENCHFLOW_AGENTS_DIR=str(agents.parent),
        BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
        # The stand-in makes no model call; the restricted-web model check
        # admits its placeholder route only by the operator's override.
        BENCHFLOW_RESTRICTED_WEB_MODELS="openai/gpt-4.1-mini",
        # Both of BenchFlow's Docker paths: the Engine API's and Compose's.
        BENCHFLOW_DIRECT_DOCKER="1" if path == "direct" else "0",
    )
    if "PYTHONPATH" in os.environ:  # a tree under test, not this venv's
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    jobs = tmp_path / "jobs"
    events_file = tmp_path / "events.jsonl"
    with events_file.open("w") as sink:
        events = subprocess.Popen(
            [
                "docker",
                "events",
                "--format",
                "{{json .}}",
                "--filter",
                "type=container",
            ],
            stdout=sink,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        try:
            done = subprocess.run(
                [
                    str(_bench()),
                    "eval",
                    "run",
                    "--tasks-dir",
                    str(task),
                    "--agent",
                    "bfstartup",
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
                capture_output=True,
                text=True,
                timeout=900,
            )
        finally:
            time.sleep(1)
            events.terminate()
            events.wait(timeout=30)
    rows = []
    for raw in events_file.read_text().splitlines():
        try:
            rows.append(json.loads(raw))
        except ValueError:
            continue
    results = [json.loads(p.read_text()) for p in jobs.rglob("result.json")]
    return {"done": done, "events": rows, "results": results}


def _assert_rules_before_launch(run: dict, tag: str, tail: str) -> str:
    launch, rules_end, command = _order(run["events"], tag)
    assert launch is not None, ("no agent launch in dockerd's events", tail)
    assert rules_end is not None, ("no uid rules exec in dockerd's events", tail)
    assert rules_end < launch, (
        f"the uid rules ended {(rules_end - launch) / 1e9:+.3f} s after the launch"
    )
    return command


def _order(events: list[dict], tag: str) -> tuple[int | None, int | None, str]:
    """From dockerd's events of this trial's containers (main, BenchFlow's
    helper): when the exec launching the agent started, and when the first
    exec adding the agent's uid rules ended (ns, the host's clock), and
    that exec's command."""
    launch = None
    rules = None
    command_of_rules = ""
    ended: dict[str, int] = {}
    for event in events:
        attrs = (event.get("Actor") or {}).get("Attributes") or {}
        if tag not in attrs.get("name", ""):
            continue
        action = event.get("Action") or ""
        when = int(event.get("timeNano") or 0)
        exec_id = attrs.get("execID", "")
        if action.startswith("exec_start: "):
            command = action[len("exec_start: ") :]
            if (
                launch is None
                and "/opt/bfstartup/acp.sh" in command
                and "BENCHFLOW_ACP" in command
            ):
                launch = when
            if rules is None and "agent_uid=" in command and "--uid-owner" in command:
                rules, command_of_rules = exec_id, command
        elif action.startswith("exec_die"):
            ended.setdefault(exec_id, when)
    return launch, ended.get(rules) if rules is not None else None, command_of_rules


@pytest.mark.parametrize("path", ["direct", "compose"])
@pytest.mark.parametrize("policy", ["no-network", "allowlist"])
def test_an_agents_startup_connection_never_reaches_the_host(
    tmp_path, listener, policy, path
):
    """The stand-in's first act, a connection to the host, gets nowhere:
    the listener gets nothing from the trial (no connection, no line at
    start-up, at the prompt or from a fresh connection then), and the exec
    adding the agent's uid rules ended before the exec launching it began.
    On rel-0.9.x e41cd6195 (no-network, under load) the start-up line and
    the prompt's line both arrived, and the rules went in 5 s after the
    launch."""
    tag = uuid.uuid4().hex[:8]
    task = _task(tmp_path, listener.port, policy, tag)
    run = _trial(tmp_path, task, path)
    time.sleep(1)  # anything still in flight arrives
    tail = run["done"].stderr[-3000:]
    assert run["results"], tail
    assert listener.seen(tag) == [], (listener.seen(tag), tail)
    _assert_rules_before_launch(run, tag, tail)


@pytest.mark.parametrize("path", ["direct", "compose"])
def test_the_control_startup_connection_reaches_the_host(tmp_path, listener, path):
    """The control: under an open policy the same start-up connection and
    its line reach the listener, so the silence above is the rules'."""
    tag = uuid.uuid4().hex[:8]
    task = _task(tmp_path, listener.port, "open", tag)
    run = _trial(tmp_path, task, path)
    time.sleep(1)
    seen = [text for _, text in listener.seen(tag)]
    assert any(text.endswith(f"startup {tag} uid={AGENT_UID}") for text in seen), (
        seen,
        run["done"].stderr[-3000:],
    )


@pytest.mark.parametrize("path", ["direct", "compose"])
def test_the_launch_fallback_holds_the_startup_connection_too(tmp_path, listener, path):
    """Under no-network, an image in which the agent's user cannot be made
    before main's release: its rules go in at the launch, before the agent's
    process starts. The rules exec names the uid the user's setup made (from
    1000, below the early range), so this is the fallback; the listener gets
    nothing. On #309's dabe08d12 they went in after the ACP handshake and
    the start-up line arrived (WIN698's reading)."""
    tag = uuid.uuid4().hex[:8]
    task = _task(tmp_path, listener.port, "no-network", tag, fallback=True)
    run = _trial(tmp_path, task, path)
    time.sleep(1)
    tail = run["done"].stderr[-3000:]
    assert run["results"], tail
    assert listener.seen(tag) == [], (listener.seen(tag), tail)
    command = _assert_rules_before_launch(run, tag, tail)
    uid = re.search(r"agent_uid=(\d+);", command)
    assert uid is not None and int(uid[1]) < 60000, command[-400:]
