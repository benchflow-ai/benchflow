"""a branch child carries its source's test.sh verdict; it never re-judges.

RUSAGE's finding (2026-11-16, on #301 f79700076): ``bench branch
--from-checkpoint`` restores the source trial's sandbox after its agent ran,
and each child's install re-ran the test.sh preflight there, so a root agent
that had removed the command test.sh runs first under errexit got its
children refused where the source ended as the agent's failure. The rule
(F2b, 07:40Z): the verdict is decided once, before the source's turn on the
untouched image, and a child reuses it, refused or clean; a checkpoint that
recorded none (made before the check) is decided on a fresh sandbox of the
image, and where that cannot be done the child ends unscored as BenchFlow's
setup, never the task's fault. A child also runs as its source's sandbox
user (the job's evaluation.json): a root agent's checkpoint had no sandbox
user to quiesce.

- default-user source: its children carry its verdict and score;
- the same checkpoint with its verdict removed: decided on a fresh image;
- RUSAGE's repro, a root agent that removes the tool: its children are the
  agent's failure (no usable reward), never refused;
- a source refused at its preflight: no child starts.

Deterministic tier (``tests/integration/README.md``): real Docker, the fake
provider, no key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic
SANDBOX, SKIP_REASON = h.select_sandbox()
TOOL = (
    "RUN mkdir -p /opt/x/bin && printf '#!/bin/sh\\nexit 0\\n' > /opt/x/bin/mytool "
    "&& chmod 755 /opt/x/bin/mytool\nENV PATH=/opt/x/bin:$PATH\n"
)
TEST_SH = (
    "#!/bin/bash\nset -e\nmytool --check\nREWARD=0\n"
    "if [ -f /app/hello.txt ]; then REWARD=1; fi\n"
    'echo "$REWARD" > /logs/verifier/reward.txt\n'
)
HELLO = {
    "tool": "bash",
    "input": {"command": "printf 'Hello, world!' > /app/hello.txt"},
}
DONE = {
    "tool": "bash",
    "input": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
}
SCRIPTS = {
    "en715-hello": [dict(HELLO, text="Writing the file."), DONE],
    "en715-del": [
        {
            "text": "Removing the tool.",
            "tool": "bash",
            "input": {
                "command": "rm -f /opt/x/bin/mytool; printf 'Hello, world!' > /app/hello.txt"
            },
        },
        DONE,
    ],
    "en715-noop": [DONE],
}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    from tests.test_deterministic_mini_swe import _plugin_pythonpath

    load = h._load_fake_llm_module

    def fake_module():
        module = load()
        original = module._Handler.do_POST

        def post(handler):
            handler.scripts.update(SCRIPTS)
            return original(handler)

        module._Handler.do_POST = post
        return module

    monkeypatch.setattr(h, "_load_fake_llm_module", fake_module)
    with h.host_fake_llm(tmp_path / "fake.jsonl") as url:
        yield url, {"PYTHONPATH": _plugin_pythonpath(tmp_path / "plugin")}


def _task(root: Path, script: str, *, with_tool: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    task = h.materialize_task(h.TaskVariant("en715-branch", script), root)
    dockerfile = task / "environment" / "Dockerfile"
    if with_tool:
        dockerfile.write_text(dockerfile.read_text().rstrip("\n") + "\n" + TOOL)
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text(TEST_SH)
    test_sh.chmod(0o755)
    return task


def _run(fake, task, jobs, *extra, command=("eval", "run")):
    url, env = fake
    return h.run_bench(
        list(command),
        tasks_dir=task,
        jobs_dir=jobs,
        sandbox=SANDBOX,
        host_fake_url=url,
        route="vllm",
        agent="mini-swe",
        env_extra=env,
        extra=list(extra),
    )


def _branch(fake, task, jobs, trial):
    prompt = "Continue " + h.marker("en715-noop")
    return _run(
        fake,
        task,
        jobs,
        "--from-checkpoint",
        str(trial),
        "--child",
        f"label=a,prompt={prompt}",
        "--child",
        f"label=b,prompt={prompt}",
        command=("branch",),
    )


def _branch_trial(jobs: Path) -> tuple[dict, list[dict]]:
    # The source trial's own record: since that change each isolated child has one too.
    (trial,) = [
        p.parent
        for p in jobs.rglob("result.json")
        if "children" not in p.relative_to(jobs).parts
    ]
    (labels,) = trial.glob("branches/*/labels.json")
    return h.read_json(trial / "result.json"), h.read_json(labels)["children"]


def test_children_carry_the_source_verdict_or_one_decided_on_a_fresh_image(
    tmp_path, fake
):
    task = _task(tmp_path / "tasks", "en715-hello")
    parent = _run(
        fake,
        task,
        tmp_path / "parent",
        "--checkpoints",
        "prompt:1",
        "--retry-attempts",
        "0",
    )
    trial = parent.trial(task.name)
    try:
        source = h.read_json(trial / "result.json")["test_sh_preflight"]
        assert (source["checked"], source["fatal"], source["missing"]) == (
            True,
            ["mytool"],
            [],
        ), parent.output[-2000:]
        assert "inherited" not in source
        for jobs, recorded in ((tmp_path / "branch", True), (tmp_path / "old", False)):
            if not recorded:
                # A checkpoint from before the check: no verdict recorded.
                data = json.loads((trial / "result.json").read_text())
                data.pop("test_sh_preflight")
                (trial / "result.json").write_text(json.dumps(data))
            branch = _branch(fake, task, jobs, trial)
            assert branch.returncode == 0, branch.output[-3000:]
            assert "startup refused" not in branch.output
            result, children = _branch_trial(jobs)
            carried = result["test_sh_preflight"]
            assert carried["inherited"] is True, carried
            assert (carried["fatal"], carried["missing"]) == (["mytool"], [])
            assert "refused" not in carried
            assert ("decided_on" in carried) is not recorded
            assert [(c["label"], c["status"], c["reward"]) for c in children] == [
                ("a", "scored", 1.0),
                ("b", "scored", 1.0),
            ]
    finally:
        h.release_kept_checkpoints(trial, SANDBOX)


def test_children_of_a_root_agent_that_removed_the_tool_stay_its_failure(
    tmp_path, fake
):
    """RUSAGE's repro: the source ended as the agent's failure, and so do its
    children; none is refused."""
    task = _task(tmp_path / "tasks", "en715-del")
    parent = _run(
        fake,
        task,
        tmp_path / "parent",
        "--sandbox-user",
        "none",
        "--checkpoints",
        "prompt:1",
        "--retry-attempts",
        "0",
    )
    trial = parent.trial(task.name)
    try:
        source = h.read_json(trial / "result.json")
        assert source["test_sh_preflight"]["fatal"] == ["mytool"]
        assert source["verifier_error"].startswith("verifier wrote no usable reward: ")
        branch = _branch(fake, task, tmp_path / "branch", trial)
        assert "startup refused" not in branch.output, branch.output[-3000:]
        assert "Invalid sandbox UID" not in branch.output
        result, children = _branch_trial(tmp_path / "branch")
        assert result["test_sh_preflight"]["inherited"] is True
        assert "refused" not in result["test_sh_preflight"]
        assert [c["label"] for c in children] == ["a", "b"]
        for child in children:
            # Run to the end and graded: the agent's failure (no usable
            # reward), never a refusal and never left unstarted.
            assert child["status"] not in ("not_started", "refused"), child
            assert child["reward"] in (None, 0.0), child
    finally:
        h.release_kept_checkpoints(trial, SANDBOX)


def test_a_source_refused_at_its_preflight_starts_no_child(tmp_path, fake):
    task = _task(tmp_path / "tasks", "en715-hello", with_tool=False)
    run = _run(
        fake,
        task,
        tmp_path / "branch",
        "--prompt",
        "@instruction",
        "--checkpoint-after-prompt",
        "1",
        "--child",
        "label=a",
        "--child",
        "label=b",
        command=("eval", "branch"),
    )
    (trial,) = [
        p.parent
        for p in (tmp_path / "branch").rglob("result.json")
        if "children" not in p.relative_to(tmp_path / "branch").parts
    ]
    result = h.read_json(trial / "result.json")
    # Refused at its own preflight, before its turn; the branch's outcome and
    # its CLI line carry the refusal.
    assert result["test_sh_preflight"]["refused"] == "mytool"
    assert "sandbox startup refused: test.sh needs `mytool`" in run.output
    labels = list(trial.glob("branches/*/labels.json"))
    started = [
        c
        for f in labels
        for c in h.read_json(f)["children"]
        if c["status"] != "not_started"
    ]
    assert started == [], run.output[-3000:]


def test_children_of_a_test_sh_the_check_does_not_read_run(tmp_path, fake):
    """RUSAGE: a ``#!/usr/bin/env python3`` test.sh is run by the verifier but
    never read by the check, so its source records no verdict and its
    children have none to miss: they run and score."""
    task = _task(tmp_path / "tasks", "en715-hello")
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text(
        "#!/usr/bin/env python3\nimport os\n"
        "ok = os.path.exists('/app/hello.txt')\n"
        "open('/logs/verifier/reward.txt', 'w').write('1' if ok else '0')\n"
    )
    test_sh.chmod(0o755)
    parent = _run(
        fake,
        task,
        tmp_path / "parent",
        "--checkpoints",
        "prompt:1",
        "--retry-attempts",
        "0",
    )
    trial = parent.trial(task.name)
    try:
        source = h.read_json(trial / "result.json")
        assert source["rewards"] == {"reward": 1.0}, parent.output[-2000:]
        assert source["test_sh_preflight"] is None
        branch = _branch(fake, task, tmp_path / "branch", trial)
        assert branch.returncode == 0, branch.output[-3000:]
        assert "startup refused" not in branch.output
        result, children = _branch_trial(tmp_path / "branch")
        assert result["test_sh_preflight"] is None
        assert [(c["label"], c["status"], c["reward"]) for c in children] == [
            ("a", "scored", 1.0),
            ("b", "scored", 1.0),
        ]
    finally:
        h.release_kept_checkpoints(trial, SANDBOX)
