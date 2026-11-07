"""The hill-climb demo on real Docker sandboxes, with the scripted fake model.

Part of the deterministic tier (``tests/integration/README.md``): only the
model is scripted. The agent under test is ``claude-agent-acp`` in each
task's sandbox, the demo's skills folder baked into its image; the optimizer
is ``claude-agent-acp`` in the demo's generated task, with the evidence
uploaded and locked read-only. Both reach the fake provider through
BenchFlow's LiteLLM proxy on the host, so this is Docker-only.

The agent writes the right answer only when a skill says ``WRITE-HELLO``;
the optimizer's script adds that rule and, from inside its sandbox, lists
what it was given and walks the whole filesystem for the test tasks. It runs
with ``open_network``: without it the model proxy moves into the sandbox and
cannot reach the host fake. tests/test_hillclimb_example.py checks the
no-network task.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

DEMO = Path(__file__).resolve().parents[1] / "docs" / "examples" / "hillclimb"
sys.path.insert(0, str(DEMO))
import hillclimb  # noqa: E402
from hillclimb_proposer import ProposerSettings  # noqa: E402

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
TRAIN = ["hc-train-1", "hc-train-2"]
TEST = ["hc-test-1", "hc-test-2"]

AGENT_SCRIPT = [
    {
        "text": "Following the skills.",
        "tool": "Bash",
        "input": {
            "command": (
                "if grep -rqs 'WRITE-HELLO' /skills; then "
                "printf 'Hello, world!\\n' > /app/hello.txt; "
                "else printf 'Goodbye, world!\\n' > /app/hello.txt; fi"
            ),
            "description": "Follow the skills",
        },
    },
    {"text": "Done."},
]

PROPOSER_PROGRAM = r"""
import json, os, pathlib
seen = sorted(str(p) for p in pathlib.Path('/hillclimb').rglob('*') if p.is_file())
found = []
for root, dirs, files in os.walk('/'):
    if root.startswith(('/proc', '/sys', '/dev')):
        dirs[:] = []
        continue
    found += [os.path.join(root, n) for n in dirs + files if 'hc-test' in n]
text_hits = [s for s in seen if 'hc-test' in pathlib.Path(s).read_text(errors='replace')]
skill = pathlib.Path('/app/surface/skills/hello/SKILL.md')
skill.write_text(skill.read_text() + '\nWRITE-HELLO: write exactly the greeting the task asks for.\n')
json.dump({
    'root_cause': 'the agent writes the wrong greeting',
    'change': 'added a WRITE-HELLO rule to the hello skill',
    'rationale': 'every task asks for an exact greeting',
    'evidence': [],
    'seen': seen,
    'test_paths_found': found,
    'test_text_found': text_hits,
}, open('/app/proposal.json', 'w'))
"""

PROPOSER_SCRIPT = [
    {
        "text": "Reading the failures and editing the skill.",
        "tool": "Bash",
        "input": {
            "command": f"python3 - <<'PY'\n{PROPOSER_PROGRAM}\nPY",
            "description": "Propose one change",
        },
    },
    {"text": "Proposed one change."},
]


@pytest.fixture
def fake_llm(tmp_path):
    """The deterministic tier's fake, on the host, with this scenario's scripts."""
    fake = h._load_fake_llm_module()
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    scripts.update({"hc-agent": AGENT_SCRIPT, "hc-propose": PROPOSER_SCRIPT})
    fake._Handler.scripts = scripts
    fake._Handler.log_path = tmp_path / "fake-llm.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake._Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def hermetic_env(monkeypatch):
    """A developer's real provider settings must not reach a scripted run."""
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_")) or key in {
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
        }:
            monkeypatch.delenv(key)


def test_the_demo_climbs_on_docker_and_the_optimizer_never_sees_the_test_split(
    tmp_path, fake_llm, hermetic_env
):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "the host fake needs Docker")
    root = Path(os.environ.get(h.KEEP_JOBS_ENV) or tmp_path) / "hillclimb-demo"
    tasks = root / "tasks"
    for name in TRAIN + TEST:
        task = h.materialize_task(h.TaskVariant(name, "hc-agent"), tasks)
        with (task / "task.md").open("a") as handle:  # distinct instructions
            handle.write(f"\nThis is task {name}.\n")
    skill = root / "skills" / "hello"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: hello\ndescription: Greeting files.\n---\nWrite greeting files in /app.\n"
    )
    split = root / "split.json"
    split.write_text(json.dumps({"train": TRAIN, "test": TEST}))
    agent_env = h.route_env("proxy", "docker", fake_llm)
    h.check_route_is_hermetic("proxy", agent_env)
    s = hillclimb.Settings(
        tasks_dir=tasks,
        skills=root / "skills",
        out=root / "run",
        split_file=split,
        agent=h.AGENT,
        model=h.MODEL,
        agent_env=agent_env,
        sandbox="docker",
        concurrency=int(os.environ.get("BENCHFLOW_DETERMINISTIC_CONCURRENCY", "4")),
        trials=2,
        rounds=1,
        min_gain=0.1,
        retry_attempts=0,
        bootstrap_samples=200,
        proposer=ProposerSettings(
            agent=h.AGENT,
            model=h.MODEL,
            agent_env=agent_env,
            sandbox="docker",
            open_network=True,
            timeout_sec=600,
            extra_instructions="\n" + h.marker("hc-propose") + "\n",
        ),
    )
    doc = asyncio.run(hillclimb.climb(s))

    assert doc["status"] == "finished", doc["stop"]
    assert (
        doc["controls"]["excluded"] == []
    )  # the oracle passes, doing nothing does not
    for split_name in ("train", "test"):
        assert doc["baseline"][split_name]["infra_errors"] == 0
        assert doc["baseline"][split_name]["score"]["value"] == 0.0
    assert doc["noise_gate"]["passed"]
    entry = doc["rounds"][0]
    assert entry["candidate"]["status"] == "ok", entry["candidate"]["error"]
    assert entry["decision"] == "keep", entry["reasons"]
    assert entry["train_delta"]["value"] == 1.0 and entry["test_delta"]["value"] == 1.0
    assert doc["best"]["verdict"]["exceeds_noise"]
    # What the optimizer saw, reported from inside its own sandbox.
    proposal = json.loads(
        (
            Path(entry["candidate"]["rollout_dir"]) / "verifier" / "proposal.json"
        ).read_text()
    )
    assert proposal["seen"] and all(
        p.startswith("/hillclimb/") for p in proposal["seen"]
    )
    assert any(
        p.startswith("/hillclimb/train/failures/hc-train-1/") for p in proposal["seen"]
    )
    assert proposal["test_paths_found"] == [] and proposal["test_text_found"] == []
    seen = entry["candidate"]["mounted"]
    assert (
        seen["test_tasks_in_paths"] == [] and seen["test_instructions_in_files"] == []
    )
    assert seen["train_tasks"] == TRAIN
    assert "What the optimizer saw" in (s.out / "report.html").read_text()
