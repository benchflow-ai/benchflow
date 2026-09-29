"""``bf.hillclimb`` on real Docker sandboxes with the scripted fake model.

Part of the deterministic tier (``tests/integration/README.md``): nothing is
faked but the model. The agent under test is ``claude-agent-acp`` in each
task's sandbox, its skills folder baked into the image; the optimizer is
``claude-agent-acp`` in the generated proposer task's sandbox, with the
evidence uploaded and locked read-only. Both talk to the fake provider through
BenchFlow's LiteLLM proxy on the host, so the scenario is Docker-only (on
Daytona the proxy runs in the sandbox, where these scripts are not served).

The agent's script writes the right answer only when a skill says
``WRITE-HELLO``; the optimizer's script adds that line, and on its way records
every file it was given and searches the whole sandbox for the test tasks.
The optimizer runs with ``open_network`` here: with ``allow_internet: false``
the model proxy moves into the sandbox and cannot reach the host fake. The
no-network wrapper is checked by the unit tests.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import benchflow as bf
from benchflow.hillclimbing import HillclimbConfig, ProposerSettings
from tests.integration.deterministic import harness as h

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
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_")) or key in {
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
        }:
            monkeypatch.delenv(key)


def test_a_real_climb_keeps_a_patch_and_the_optimizer_never_sees_the_test_split(
    tmp_path, fake_llm, hermetic_env
):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "the host fake needs Docker")
    root = Path(os.environ.get(h.KEEP_JOBS_ENV) or tmp_path) / "hillclimb"
    tasks = root / "tasks"
    for name in TRAIN + TEST:
        task = h.materialize_task(h.TaskVariant(name, "hc-agent"), tasks)
        # Distinct instructions: identical ones would be (rightly) reported as
        # a test task duplicated in the train split.
        with (task / "task.md").open("a") as handle:
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

    result = bf.hillclimb(
        HillclimbConfig(
            tasks=tasks,
            surface=root / "skills",
            out=root / "run",
            split_file=split,
            agent=h.AGENT,
            model=h.MODEL,
            environment="docker",
            agent_env=agent_env,
            concurrency=int(os.environ.get("BENCHFLOW_DETERMINISTIC_CONCURRENCY", "4")),
            trials=2,
            rounds=1,
            min_gain=0.1,
            retry_attempts=0,
            bootstrap_samples=200,
            proposer=ProposerSettings(
                agent=h.AGENT,
                model=h.MODEL,
                environment="docker",
                agent_env=agent_env,
                open_network=True,
                timeout_sec=600,
                extra_instructions=h.marker("hc-propose"),
            ),
        )
    )
    doc = result.record
    assert doc.status == "finished", doc.stop

    # The graders: the template's oracle passes, doing nothing does not.
    assert doc.controls.ran and doc.controls.grader_bugs == []
    # Baseline: the skill lacks the rule, every trial writes the wrong greeting.
    for split_doc in (doc.baseline.train, doc.baseline.test):
        assert split_doc.infra_errors == 0, split_doc.infra_error_categories
        assert split_doc.score.value == 0.0
    assert doc.noise_gate.passed

    cand = doc.rounds[0].candidates[0]
    assert cand.proposer.status == "ok", cand.proposer.error
    assert cand.decision == "keep", cand.reasons
    assert cand.train_delta.value == 1.0 and cand.test_delta.value == 1.0
    assert "WRITE-HELLO" in cand.diff
    assert doc.best.version == cand.version and doc.best.verdict.exceeds_noise

    # What the optimizer itself saw, from inside its sandbox.
    proposal = json.loads(
        (
            result.run_dir / cand.proposer.rollout_dir / "verifier" / "proposal.json"
        ).read_text()
    )
    assert proposal["seen"], "the optimizer saw no evidence"
    assert all(p.startswith("/hillclimb/") for p in proposal["seen"])
    assert any(
        p.startswith("/hillclimb/train/failures/hc-train-1/") for p in proposal["seen"]
    )
    assert proposal["test_paths_found"] == []
    assert proposal["test_text_found"] == []
    seen = cand.proposer.exposure
    assert seen.test_tasks_in_paths == [] and seen.test_instructions_in_files == []
    assert seen.train_tasks == TRAIN

    # Every evaluation is a real job; the kept patch is in the history.
    for split_name, names in (("train", TRAIN), ("test", TEST)):
        folder = result.run_dir / "evals" / cand.id / split_name / "trial-01" / "job"
        rewards = {
            json.loads(p.read_text())["task_name"]: json.loads(p.read_text())["rewards"]
            for p in folder.glob("*/result.json")
        }
        assert rewards == {n: {"reward": 1.0} for n in names}
    assert (result.run_dir / "surface-history" / ".git").is_dir()
    assert "What the optimizer saw" in result.report.read_text()
