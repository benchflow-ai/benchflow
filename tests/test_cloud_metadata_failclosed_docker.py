"""GAP-METADATA-FAILOPEN on a real Docker daemon, through the real
``bench eval run``.

The agent never starts unless the cloud metadata block is proven on, for an
agent that runs as root too, and only an explicit opt-out lets it start
without the block, recorded in the result. Each test runs one trial of a stub
ACP agent (the PERF kit's shape) which, at its prompt, reads its capability
sets, connects to each IPv4 endpoint, tries to remove the rules and to open a
packet socket, and connects again.

Each trial's ``main`` is on an internal network, so nothing in it can reach
anything outside the host, a real metadata server included. The endpoints
are mocked in the sandbox's own network namespace: as soon as
``main`` exists, a test container joined to it adds each IPv4 endpoint to the
loopback and listens on port 80. A connect to an endpoint therefore never
leaves that namespace, and the agent's probe connects to nothing until the
mock is there, so no real metadata server is contacted. The tests skip where
no Docker daemon answers (the gate's container); everything they
start carries the labels below and is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4
from tests._metadata_mock import MOCK_SH

LABEL = "bf.test.mdprobe"
MOCK_LABEL = "bf.test.mdprobe.mock"
# main's healthcheck: the mock's 169.254.169.254 is in main's namespace. On
# the internal network nothing else routes that address, so a block proof
# made before the mock lands connects nowhere and is rightly unproven
# (EN-684: R4b-prime's -n 10 gate). BenchFlow's start runs this check after the
# bootstrap releases main and before the block, so the trial waits for the
# mock; a mock that never comes fails the start by name.
MOCK_READY = {
    "test": ["CMD-SHELL", "grep -q '169\\.254\\.169\\.254' /proc/net/fib_trie"],
    "interval": "1s",
    "timeout": "10s",
    "retries": 3,
    "start_period": "120s",
}
# CAP_NET_ADMIN and CAP_NET_RAW in /proc/<pid>/status's capability masks.
NET_ADMIN, NET_RAW = 1 << 12, 1 << 13


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


pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

# The stub ACP agent: the PERF kit's, with the probe run at the prompt.
ACP_SH = r"""#!/bin/sh
echo launched > /logs/agent/launched
while IFS= read -r line; do
  id=$(printf '%s' "$line" | grep -o '"id": *[0-9][0-9]*' | head -1 | grep -o '[0-9][0-9]*$')
  m=$(printf '%s' "$line" | grep -o '"method": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$m" in
    initialize) r='{"protocolVersion":1,"agentInfo":{"name":"bfprobe","version":"1"},"agentCapabilities":{"loadSession":false,"promptCapabilities":{"image":false,"audio":false,"embeddedContext":false}},"authMethods":[]}';;
    session/new) r='{"sessionId":"bfprobe"}';;
    session/prompt)
      /opt/bfprobe/probe.sh
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"bfprobe","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"probed"}}}}\n{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"bfprobe","update":{"sessionUpdate":"tool_call","toolCallId":"t1","title":"probe","kind":"execute","status":"completed","rawInput":{}}}}\n'
      r='{"stopReason":"end_turn"}';;
    session/set_config_option) r='{"configOptions":[]}';;
    *) r='{}';;
  esac
  [ -n "$id" ] && printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$r"
done
"""

# What the agent does at its prompt, as the agent. Nothing connects until
# the test's mock endpoints are in this namespace.
PROBE_SH = r"""#!/bin/bash
out=/logs/agent/probe.txt
{
  echo "uid $(id -u)"
  grep -E '^Cap(Bnd|Prm|Eff):' /proc/self/status
  for _ in $(seq 1 300); do
    grep -q '169\.254\.169\.254' /proc/net/fib_trie 2>/dev/null && break
    sleep 0.1
  done
  if ! grep -q '169\.254\.169\.254' /proc/net/fib_trie 2>/dev/null; then
    echo "mock missing"
    exit 0
  fi
  probe() {
    for a in __ENDPOINTS__; do
      err=$(timeout 5 bash -c "exec 3<>/dev/tcp/$a/80" 2>&1); rc=$?
      case $rc in
        0) s=reached;;
        124) s=timeout;;
        *) case $err in *[Rr]efused*) s=refused;; *) s=failed;; esac;;
      esac
      echo "$1 $a $s"
    done
  }
  probe connect
  # Remove the rules: that needs NET_ADMIN.
  for t in iptables iptables-nft iptables-legacy; do
    command -v "$t" >/dev/null || continue
    "$t" -w -F OUTPUT >/dev/null 2>&1; echo "flush $t $?"
  done
  # A packet socket (NET_RAW) sends frames the OUTPUT chain never sees.
  perl -e 'socket(my $s, 17, 3, 0) or die "denied: $!\n"; print "opened\n"' 2>&1 \
    | sed 's/^/packet /'
  probe after
  echo done
} > "$out" 2>&1
"""

DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends iptables \\
    && rm -rf /var/lib/apt/lists/*
COPY acp.sh probe.sh /opt/bfprobe/
RUN chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh
"""

MANIFEST = """contract_version = "1.0"
name = "bfprobe"
default_model = "openai/gpt-4.1-mini"
install_cmd = "test -x /opt/bfprobe/acp.sh"
launch_cmd = "/opt/bfprobe/acp.sh"
"""

# Fails the metadata helper's image build, and passes every other call on.
FAILING_HELPER_DOCKER = """#!/bin/sh
case " $* " in
  *" build "*"benchflow.helper=metadata"*)
    echo "forced helper build failure (test)" >&2
    exit 1;;
esac
exec __DOCKER__ "$@"
"""


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    """The task image with the stub agent and the probe, built once."""
    context = tmp_path_factory.mktemp("bfprobe-image")
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "acp.sh").write_text(ACP_SH)
    (context / "probe.sh").write_text(
        PROBE_SH.replace("__ENDPOINTS__", " ".join(CLOUD_METADATA_V4))
    )
    tag = f"bf-test-mdprobe:{uuid.uuid4().hex[:12]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build the probe image: {built.stderr.strip()[-300:]}")
    try:
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


class _Mock:
    """Starts the mock endpoints in main's namespace once main exists, and
    records main's capability grants from ``docker inspect``. ``delay``
    starts it that many seconds late, as a loaded daemon can (EN-684)."""

    def __init__(self, run_id: str, *, delay: float = 0.0) -> None:
        self.run_id = run_id
        self.delay = delay
        self.main = ""
        self.cap_add: list[str] = []
        self.cap_drop: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> _Mock:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        for label in (f"{MOCK_LABEL}={self.run_id}", f"{LABEL}={self.run_id}"):
            left = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={label}"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split()
            if left:
                subprocess.run(["docker", "rm", "--force", *left], capture_output=True)

    def _watch(self) -> None:
        if self._stop.wait(self.delay):
            return
        while not self._stop.is_set() and not self.main:
            found = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label={LABEL}={self.run_id}",
                    "--filter",
                    "status=running",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split()
            if not found:
                time.sleep(0.2)
                continue
            self.main = found[0]
            caps = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .HostConfig.CapAdd}} {{json .HostConfig.CapDrop}}",
                    self.main,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split(" ", 1)
            # Docker lists them as CAP_<NAME>.
            self.cap_add = [c.removeprefix("CAP_") for c in (json.loads(caps[0]) or [])]
            self.cap_drop = [
                c.removeprefix("CAP_")
                for c in (json.loads(caps[1]) if len(caps) > 1 else None) or []
            ]
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--rm",
                    "--label",
                    f"{MOCK_LABEL}={self.run_id}",
                    "--network",
                    f"container:{self.main}",
                    "--cap-drop",
                    "ALL",
                    "--cap-add",
                    "NET_ADMIN",
                    "--cap-add",
                    "NET_BIND_SERVICE",
                    helper_mod.HELPER_BASE,
                    "sh",
                    "-c",
                    MOCK_SH,
                ],
                capture_output=True,
                timeout=120,
            )


def _task(
    root: Path,
    image: str,
    run_id: str,
    extra_toml: str = "",
    *,
    no_web: bool = False,
) -> Path:
    task = root / "md-probe"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n'
        "[agent]\ntimeout_sec = 180.0\n\n"
        "[verifier]\ntimeout_sec = 60.0\n\n"
        f'[environment]\ndocker_image = "{image}"\n'
        # A no-network task: the agent keeps its network for the model and
        # BenchFlow enforces no-web at the agent layer.
        f"allow_internet = {'false' if no_web else 'true'}\n" + extra_toml
    )
    (task / "instruction.md").write_text("Probe the cloud metadata endpoints.\n")
    # An internal network: the trial needs no egress, and with no route out
    # nothing in main can reach a real metadata server, even before the mock
    # is in place or if a rule did not go in.
    (task / "environment" / "docker-compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "image": image,
                        "labels": {LABEL: run_id},
                        "healthcheck": MOCK_READY,
                    }
                },
                # No-web trials run the model proxy in main, which needs the
                # network; their probe connects to nothing but the proxy on
                # loopback and TEST-NET (192.0.2.1).
                **({} if no_web else {"networks": {"default": {"internal": True}}}),
            }
        )
    )
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text("#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n")
    test_sh.chmod(0o755)
    return task


def _run(
    tmp_path: Path,
    image: str,
    *flags: str,
    extra_toml: str = "",
    failing_helper: bool = False,
    no_web: bool = False,
    mock_delay: float = 0.0,
) -> tuple[subprocess.CompletedProcess[str], dict, dict, str, _Mock]:
    """One trial of the stub agent through ``bench eval run``; its result,
    cloud-metadata.json, the probe's output and what the mock saw.
    ``mock_delay`` starts the mock that many seconds late."""
    run_id = uuid.uuid4().hex[:12]
    task = _task(tmp_path, image, run_id, extra_toml, no_web=no_web)
    agents = tmp_path / "agents" / "bfprobe"
    agents.mkdir(parents=True)
    (agents / "manifest.toml").write_text(MANIFEST)
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        BENCHFLOW_AGENTS_DIR=str(agents.parent),
        BENCHFLOW_AGENTS_SOURCE=str(agents.parent),
        PYTHONUNBUFFERED="1",
        # The suite skips the host readiness checks (Docker probe, Claude
        # login file); this env is built from scratch, so keep that here.
        # Under load the Docker probe can give up before any job exists.
        BENCHFLOW_SKIP_PREFLIGHT="1",
    )
    if no_web:
        # The stub agent makes no model call; the restricted-web model check
        # admits its placeholder route only by the operator's override.
        env["BENCHFLOW_RESTRICTED_WEB_MODELS"] = "openai/gpt-4.1-mini"
    if failing_helper:
        shim = tmp_path / "shim"
        shim.mkdir()
        real = shutil.which("docker")
        assert real is not None
        docker = shim / "docker"
        docker.write_text(FAILING_HELPER_DOCKER.replace("__DOCKER__", real))
        docker.chmod(0o755)
        env["PATH"] = f"{shim}:{env['PATH']}"
        # Its own empty record of built helper images: a build recorded on
        # this daemon by an earlier trial would be reused, and the forced
        # build failure would never happen.
        env["BENCHFLOW_METADATA_HELPER_STATE_DIR"] = str(tmp_path / "helper-state")
    jobs = tmp_path / "jobs"
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    command = [
        str(bench),
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
        "--agent-env",
        "OPENAI_API_KEY=placeholder",
        "--agent-env",
        "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:9/v1",
        *flags,
    ]
    with _Mock(run_id, delay=mock_delay) as mock:
        done = subprocess.run(
            command,
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=900,
        )
    results = list(jobs.rglob("result.json"))
    assert len(results) == 1, (done.stdout[-3000:], done.stderr[-3000:])
    trial = results[0].parent
    result = json.loads(results[0].read_text())
    record_path = trial / "cloud-metadata.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    probe_path = trial / "agent" / "probe.txt"
    probe = probe_path.read_text() if probe_path.exists() else ""
    return done, result, record, probe, mock


def _why(done: subprocess.CompletedProcess[str], result: dict, record: dict) -> str:
    """The whole block record, the trial's error and the CLI's last output,
    as text: pytest cuts a dict message at 240 characters, which left
    R4b-prime's 'unproven' record without its reason (EN-684)."""
    return json.dumps(
        {
            "record": record,
            "error": result.get("error"),
            "error_category": result.get("error_category"),
            "stderr": done.stderr[-2000:],
        },
        indent=1,
        default=str,
    )


def _caps(probe: str, name: str) -> int:
    line = next(ln for ln in probe.splitlines() if ln.startswith(f"{name}:"))
    return int(line.split()[1], 16)


def _connects(probe: str, kind: str) -> set[str]:
    return {
        ln.split()[2]
        for ln in probe.splitlines()
        if ln.startswith(f"{kind} ") and len(ln.split()) == 3
    }


def test_a_root_agent_runs_only_behind_the_helpers_block(tmp_path, probe_image):
    """A root agent (``--sandbox-user null``): main starts with neither
    NET_ADMIN nor NET_RAW, the helper's rules refuse root, the block is
    proven on, and the agent can neither remove the rules nor go around
    them with a packet socket."""
    done, result, record, probe, mock = _run(
        tmp_path, probe_image, "--sandbox-user", "null"
    )
    assert result.get("error") in (None, ""), (result.get("error"), done.stderr[-2000:])
    assert record["state"] == "on", record
    assert record["rules_from"] == "helper", record
    assert record.get("root_agent") is True, record
    assert result["cloud_metadata"] == {"state": "on"}
    assert "NET_ADMIN" not in mock.cap_add, mock.cap_add
    assert "NET_RAW" in mock.cap_drop, mock.cap_drop
    assert "uid 0" in probe, probe
    for name in ("CapBnd", "CapPrm", "CapEff"):
        assert not _caps(probe, name) & (NET_ADMIN | NET_RAW), (name, probe)
    assert _connects(probe, "connect") == {"refused"}, probe
    assert all(
        ln.split()[2] != "0" for ln in probe.splitlines() if ln.startswith("flush ")
    ), probe
    assert "packet opened" not in probe and "packet denied" in probe, probe
    assert _connects(probe, "after") == {"refused"}, probe


def test_the_agent_user_runs_only_behind_the_block(tmp_path, probe_image):
    """The default agent user: refused at every IPv4 endpoint, before and
    after it tries to remove the rules, with no NET_ADMIN of its own; the
    block is on from the helper. Main's own grant is what the record says:
    a process's first sandbox keeps NET_ADMIN unless the helper image was
    already built (df3a563d never waits for the build); the root agent's
    test and tests/test_metadata_helper_container.py show main without it."""
    done, result, record, probe, mock = _run(tmp_path, probe_image)
    assert result.get("error") in (None, ""), (result.get("error"), done.stderr[-2000:])
    assert record["state"] == "on" and record["rules_from"] == "helper", record
    assert result["cloud_metadata"] == {"state": "on"}
    assert ("NET_ADMIN" in mock.cap_add) is record["main_net_admin"], (
        mock.cap_add,
        record,
    )
    assert "uid 0" not in probe, probe
    assert not _caps(probe, "CapBnd") & NET_ADMIN, probe
    assert _connects(probe, "connect") == {"refused"}, probe
    assert _connects(probe, "after") == {"refused"}, probe


def test_a_forced_helper_failure_refuses_the_start(tmp_path, probe_image):
    """A root agent whose helper cannot run: main keeps NET_ADMIN for root's
    script, so root could remove any rule, and the block is not installed.
    The trial stops before the agent starts, with the reason and the
    opt-out in its error."""
    done, result, _record, probe, _mock = _run(
        tmp_path, probe_image, "--sandbox-user", "null", failing_helper=True
    )
    error = result.get("error") or ""
    assert error == (
        "RuntimeError: sandbox startup refused: BenchFlow's helper image could not "
        "be built (building the metadata firewall helper image failed: forced helper "
        "build failure (test)), so no network rule of the trial's could go in from "
        "outside main, and its processes would have the cloud metadata endpoints and "
        "the network open to them; only the task's own opt-out "
        "(allow_cloud_metadata = true in its [environment] table) on an open network "
        "starts it without them"
    ), (error, done.stderr[-2000:])
    # EN-609: main's own every-user rule goes in before main is released,
    # from the same helper: its failure stops the start there, earlier than
    # the agent's block, naming the only opt-out (the task's).
    assert "cloud metadata endpoints" in error, error
    assert "forced helper build failure" in error, error
    assert "allow_cloud_metadata = true" in error, error
    launched = list((tmp_path / "jobs").rglob("agent/launched"))
    assert launched == [] and probe == "", (launched, probe)


@pytest.mark.parametrize(
    ("flags", "extra_toml", "opt_out"),
    [
        (("--allow-cloud-metadata",), "", "--allow-cloud-metadata"),
        (
            (),
            "allow_cloud_metadata = true\n",
            "allow_cloud_metadata = true in the task's environment",
        ),
    ],
    ids=["run-flag", "task-environment"],
)
def test_an_explicit_opt_out_starts_the_agent_and_is_recorded(
    tmp_path, probe_image, flags, extra_toml, opt_out
):
    """Only an explicit opt-out lets the agent start without the block, and
    the result says how the run opted out. A root agent, so nothing but the
    opt-out lets it start. Since EN-609 (the lead's 22:40Z ruling) main's own
    every-user rule goes in from sandbox start and only the task opts out of
    it: under the run's flag the endpoints stay refused."""
    done, result, record, probe, _mock = _run(
        tmp_path, probe_image, "--sandbox-user", "null", *flags, extra_toml=extra_toml
    )
    assert result.get("error") in (None, ""), (result.get("error"), done.stderr[-2000:])
    assert record["state"] == "off", record
    assert record.get("opt_out") == opt_out, record
    assert result["cloud_metadata"] == {"state": "off", "opt_out": opt_out}
    task_opts_out = "task" in opt_out
    assert _connects(probe, "connect") == {"reached" if task_opts_out else "refused"}, (
        probe
    )


# --- Images that cannot make the test connect (PERF-A round 4) ---------------

# The probe for an image without bash or timeout: BusyBox's wget on an ftp://
# URL, which names a refused connect and sends nothing on one that gets
# through (the mock sends no FTP greeting).
ALPINE_PROBE_SH = r"""#!/bin/sh
out=/logs/agent/probe.txt
{
  echo "uid $(id -u)"
  i=0
  while [ $i -lt 300 ]; do
    grep -q '169\.254\.169\.254' /proc/net/fib_trie 2>/dev/null && break
    i=$((i + 1)); sleep 0.1
  done
  if ! grep -q '169\.254\.169\.254' /proc/net/fib_trie 2>/dev/null; then
    echo "mock missing"
    exit 0
  fi
  for a in __ENDPOINTS__; do
    err=$(env -i /usr/bin/wget -q -Y off -T 3 -O /dev/null "ftp://$a:80/" 2>&1)
    case $err in
      *[Rr]efused*) s=refused;;
      *"timed out"*|*"error getting response"*) s=reached;;
      *) s=failed;;
    esac
    echo "connect $a $s"
  done
  echo done
} > "$out" 2>&1
"""

# Alpine. BenchFlow's bind check runs /usr/bin/env and /usr/bin/stat:
# coreutils gives the first and puts stat in /bin, so the image links it, as
# the check's own message asks. The firewall-power check's root route calls
# /bin/readlink by path, and BusyBox keeps it in /usr/bin, so the image links
# that too (without it that check refuses a non-root agent on any BusyBox
# image; it is that check's rule, not the block's). setpriv is the agent
# launcher. __TOOLS__ decides what the block's own test connect lacks.
ALPINE_DOCKERFILE = f"""FROM {helper_mod.HELPER_BASE}
RUN apk add --no-cache coreutils setpriv __PACKAGES__ \\
    && ln -s /bin/stat /usr/bin/stat && ln -s /usr/bin/readlink /bin/readlink \\
    && __TOOLS__
COPY acp.sh probe.sh /opt/bfprobe/
RUN chmod 755 /opt/bfprobe /opt/bfprobe/acp.sh /opt/bfprobe/probe.sh
"""
ALPINE_IMAGES = {
    # BenchFlow starts every agent through /bin/bash and decodes its launch
    # environment with /usr/bin/base64 (coreutils keeps it in /bin here), so
    # an image an agent can run in has both; this one has no timeout, so the
    # block's test connect cannot be made in main.
    "bash-no-timeout": (
        "bash",
        "ln -s /bin/base64 /usr/bin/base64 "
        "&& rm -f /usr/bin/timeout /bin/timeout && ! command -v timeout",
    ),
    # Neither bash nor timeout (no agent launches here at all).
    "no-bash": ("", "! command -v bash"),
}


@pytest.fixture(scope="module")
def alpine_images(tmp_path_factory):
    """Both Alpine task images with the stub agent and the probe."""
    tags = {}
    try:
        for kind, (packages, tools) in ALPINE_IMAGES.items():
            context = tmp_path_factory.mktemp(f"bfprobe-alpine-{kind}")
            (context / "Dockerfile").write_text(
                ALPINE_DOCKERFILE.replace("__PACKAGES__", packages).replace(
                    "__TOOLS__", tools
                )
            )
            (context / "acp.sh").write_text(ACP_SH)
            (context / "probe.sh").write_text(
                ALPINE_PROBE_SH.replace("__ENDPOINTS__", " ".join(CLOUD_METADATA_V4))
            )
            tag = f"bf-test-mdprobe-alpine-{kind}:{uuid.uuid4().hex[:12]}"
            built = subprocess.run(
                [
                    "docker",
                    "build",
                    "--quiet",
                    "--label",
                    f"{LABEL}=image",
                    "-t",
                    tag,
                    ".",
                ],
                cwd=context,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if built.returncode != 0:
                pytest.skip(f"cannot build {kind}: {built.stderr.strip()[-300:]}")
            tags[kind] = tag
        yield tags
    finally:
        for tag in tags.values():
            subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


@pytest.mark.parametrize(
    ("flags", "uid0"),
    [((), False), (("--sandbox-user", "null"), True)],
    ids=["agent-user", "root-agent"],
)
def test_an_image_without_timeout_starts_behind_a_block_the_helper_proves(
    tmp_path, alpine_images, flags, uid0
):
    """An Alpine task image without timeout: the block's test connect is
    made from BenchFlow's helper in main's namespace, as the uid the agent's
    launch gives, and is refused and counted, so the block is on and the
    agent starts; its own connects are refused. On 55c66cf4 the record was
    unproven ("no bash and timeout") and the trial stopped before the agent."""
    done, result, record, probe, _mock = _run(
        tmp_path, alpine_images["bash-no-timeout"], *flags
    )
    assert result.get("error") in (None, ""), _why(done, result, record)
    assert record["state"] == "on", _why(done, result, record)
    assert record["rules_from"] == "helper", record
    assert record["test_connect_from"] == "helper", record
    assert (record["test_connect_ids"] == "0:0") is uid0, record
    assert result["cloud_metadata"] == {"state": "on"}
    assert ("uid 0" in probe) is uid0, probe
    assert _connects(probe, "connect") == {"refused"}, probe


def test_an_image_without_bash_is_proven_and_stops_only_at_the_agents_launch(
    tmp_path, alpine_images
):
    """A root agent on an Alpine image with no bash: the block is proven on
    from the helper (on 55c66cf4 it was unproven and the block stopped the
    trial). The trial still stops, where it stopped before: BenchFlow
    starts the agent through /bin/bash, which the image lacks."""
    done, result, record, probe, _mock = _run(
        tmp_path, alpine_images["no-bash"], "--sandbox-user", "null"
    )
    assert record["state"] == "on", _why(done, result, record)
    assert record["test_connect_from"] == "helper", record
    assert record["test_connect_ids"] == "0:0", record
    assert result["cloud_metadata"] == {"state": "on"}
    assert "CloudMetadata" not in (result.get("error") or ""), result.get("error")
    assert result.get("error"), done.stderr[-2000:]
    assert probe == ""


def test_a_late_mock_still_leaves_the_no_bash_block_proven(tmp_path, alpine_images):
    """EN-684: under R4b-prime's -n 10 gate the no-bash trial above came out
    'unproven' ("root's connect to 169.254.169.254:80 was not refused by
    the rule (failed)"): the mock's address arrived after the block's proof,
    and on the internal network nothing else routes it. Started 30 s late,
    the mock lost that race every time. main's healthcheck (MOCK_READY)
    makes BenchFlow's start wait for it, so the proof still finds the rule."""
    done, result, record, _probe, _mock = _run(
        tmp_path, alpine_images["no-bash"], "--sandbox-user", "null", mock_delay=30
    )
    assert record["state"] == "on", _why(done, result, record)
    assert record["test_connect_from"] == "helper", record
    assert result["cloud_metadata"] == {"state": "on"}
    assert "CloudMetadata" not in (result.get("error") or ""), result.get("error")


# --- A uid-0 agent on a no-web task ---------------------------

# The probe for a no-web trial, as the agent. It connects only to the model
# proxy (on loopback, from BENCHFLOW_PROVIDER_BASE_URL) and to 192.0.2.1
# (TEST-NET-1, never routed): a refused connect there is the uid firewall's
# REJECT; an open network gives a timeout. It never connects to a metadata
# address.
NOWEB_PROBE_SH = r"""#!/bin/bash
out=/logs/agent/probe.txt
{
  echo "ids $(id -u) $(id -g)"
  grep -E '^Cap(Bnd|Prm|Eff):' /proc/self/status
  t() {
    err=$(timeout 5 bash -c "exec 3<>/dev/tcp/$1/$2" 2>&1); rc=$?
    case $rc in
      0) s=reached;;
      124) s=timeout;;
      *) case $err in *[Rr]efused*) s=refused;; *[Uu]nreachable*) s=unreachable;; *) s=failed;; esac;;
    esac
    echo "$3 $s"
  }
  hp=${BENCHFLOW_PROVIDER_BASE_URL#http://}; hp=${hp%%/*}
  echo "proxy-url $BENCHFLOW_PROVIDER_BASE_URL"
  t "${hp%:*}" "${hp##*:}" proxy
  t 192.0.2.1 80 outside
  for x in iptables iptables-nft iptables-legacy; do
    command -v "$x" >/dev/null || continue
    "$x" -w -F OUTPUT >/dev/null 2>&1; echo "flush $x $?"
  done
  t 192.0.2.1 80 after
  echo done
} > "$out" 2>&1
"""


# The no-web stub agent connects to TEST-NET (192.0.2.1) as soon as it is
# launched, before ACP's initialize: the uid firewall must already refuse it
# (EN-698: the firewall went in only after session/new, and this connect
# left the sandbox unheld, timing out instead of being refused).
NOWEB_ACP_SH = ACP_SH.replace(
    "echo launched > /logs/agent/launched\n",
    "echo launched > /logs/agent/launched\n"
    'err=$(timeout 3 bash -c "exec 3<>/dev/tcp/192.0.2.1/80" 2>&1); rc=$?\n'
    "case $rc in 0) s=reached;; 124) s=timeout;; *) case $err in"
    " *[Rr]efused*) s=refused;; *[Uu]nreachable*) s=unreachable;;"
    " *) s=failed;; esac;; esac\n"
    'echo "launch $s" > /logs/agent/launch.txt\n',
)
assert NOWEB_ACP_SH != ACP_SH


def _launch_connect(tmp_path: Path) -> str:
    """What the stub agent's connect at its launch got."""
    found = list((tmp_path / "jobs").rglob("agent/launch.txt"))
    return found[0].read_text().strip() if found else "(no launch.txt)"


@pytest.fixture(scope="module")
def noweb_image(tmp_path_factory):
    """The task image with the stub agent and the no-web probe."""
    context = tmp_path_factory.mktemp("bfprobe-noweb")
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "acp.sh").write_text(NOWEB_ACP_SH)
    (context / "probe.sh").write_text(NOWEB_PROBE_SH)
    tag = f"bf-test-mdprobe-noweb:{uuid.uuid4().hex[:12]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if built.returncode != 0:
        pytest.skip(
            f"cannot build the no-web probe image: {built.stderr.strip()[-300:]}"
        )
    try:
        yield tag
    finally:
        subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


def _lines(probe: str, kind: str) -> list[str]:
    return [ln.split()[1] for ln in probe.splitlines() if ln.startswith(f"{kind} ")]


@pytest.mark.parametrize("user", ["root", "0", "null"])
def test_a_uid_0_agent_on_a_no_web_task_is_held_to_loopback(
    tmp_path, noweb_image, user
):
    """A no-web task's agent that runs as uid 0 reaches the
    model proxy on loopback and nothing else: a connect to 192.0.2.1 is
    refused by the uid firewall, it cannot flush the rules, and it holds
    neither NET_ADMIN nor NET_RAW. On 5c2d86755 the same agent had an open
    network (the connect timed out instead of being refused)."""
    from benchflow.sandbox.lockdown import NO_WEB_ROOT_GID

    done, result, _record, probe, _mock = _run(
        tmp_path, noweb_image, "--sandbox-user", user, no_web=True
    )
    assert result.get("error") in (None, ""), (result.get("error"), done.stderr[-3000:])
    # Held from its launch, before ACP's initialize (EN-698).
    assert _launch_connect(tmp_path) == "launch refused", _launch_connect(tmp_path)
    assert f"ids 0 {NO_WEB_ROOT_GID}" in probe, probe
    for name in ("CapBnd", "CapPrm", "CapEff"):
        assert not _caps(probe, name) & (NET_ADMIN | NET_RAW), (name, probe)
    assert _lines(probe, "proxy") == ["reached"], probe
    assert _lines(probe, "outside") == ["refused"], probe
    flushes = [ln.split()[2] for ln in probe.splitlines() if ln.startswith("flush ")]
    assert flushes and all(rc != "0" for rc in flushes), probe
    assert _lines(probe, "after") == ["refused"], probe


def test_a_no_web_agent_user_is_held_from_its_launch(tmp_path, noweb_image):
    """EN-698: a no-web task's agent that runs as the sandbox user is held
    from the moment it starts. Its connect to TEST-NET at launch, before
    ACP's initialize, is refused by the uid firewall. On 040c1d0c1 the
    firewall went in only after session/new, and that connect left the
    sandbox (it timed out instead of being refused)."""
    done, result, _record, probe, _mock = _run(
        tmp_path, noweb_image, "--sandbox-user", "agent", no_web=True
    )
    assert result.get("error") in (None, ""), (result.get("error"), done.stderr[-3000:])
    assert _launch_connect(tmp_path) == "launch refused", _launch_connect(tmp_path)
    assert _lines(probe, "outside") == ["refused"], probe
