# %% [markdown]
# # BenchFlow from Python: a quickstart in cells
#
# A plain Python file with `# %%` cell markers: VS Code, PyCharm and Jupytext
# run it cell by cell like a notebook, and `python quickstart.py` runs it top
# to bottom. No notebook packages are needed.
#
# - `BF_SANDBOX=daytona` (default `docker`) picks the sandbox.
# - The agent cells run only when a Claude credential is in the environment
#   (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, or `ANTHROPIC_API_KEY`).
# - `BF_QUICKSTART_OFFLINE=1` skips everything that needs a sandbox and runs the
#   reading and comparing cells on two small stand-in jobs written to disk.

# %%
import json
import logging
import os
from pathlib import Path

import benchflow as bf

SANDBOX = os.environ.get("BF_SANDBOX", "docker")
OFFLINE = os.environ.get("BF_QUICKSTART_OFFLINE") == "1"
HAS_CLAUDE = bool(
    os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
)
JOBS = Path("jobs/quickstart")


def hello_task(root: Path) -> Path:
    """A tiny task on a prebuilt image (nothing to build), with its own
    solution for the oracle and a verifier that writes the reward."""
    task = root / "hello"
    (task / "solution").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\n"
        "sandbox: {docker_image: 'python:3.12-slim', workdir: /app}\n"
        "---\n\n## prompt\n\nWrite `Hello, world!` to `hello.txt`.\n"
    )
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\necho 'Hello, world!' > /app/hello.txt\n"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\ngrep -qx 'Hello, world!' /app/hello.txt && r=1 || r=0\n"
        "echo $r > /logs/verifier/reward.txt\n"
    )
    return task


TASK = hello_task(JOBS / "tasks")
logging.basicConfig(level=logging.WARNING)  # INFO shows every phase of a run
print(f"benchflow {bf.__version__}; sandbox {SANDBOX}; offline {OFFLINE}")

# %% [markdown]
# ## 1. One rollout with the oracle
# The oracle runs the task's own solution, so this needs no model credentials:
# it checks that the sandbox, the task and its verifier work. The task above
# runs on a prebuilt image, so nothing is built: on Docker this cell takes
# about 20 seconds. `bf.run_sync` blocks (it also works inside a running
# notebook loop); in async code use `await bf.arun(...)`.

# %%
if not OFFLINE:
    oracle = bf.run_sync(
        bf.RolloutConfig(
            task_path=TASK,
            agent="oracle",
            environment=SANDBOX,
            jobs_dir=JOBS / "oracle",
        )
    )
    print(oracle)
    print("reward", oracle.reward, "passed", oracle.passed, "dir", oracle.rollout_dir)
    print(bf.load_job(oracle.rollout_dir.parent))  # the job's summary

# %% [markdown]
# ## 2. A real agent, then its trajectory and token usage
# The trajectory is a list of ACP events with a `type`: `user_message`,
# `agent_message`, `agent_thought`, `tool_call`.

# %%
if not OFFLINE and HAS_CLAUDE:
    claude = bf.run_sync(
        bf.RolloutConfig(
            task_path=TASK,
            agent="claude-agent-acp",
            model="claude-haiku-4-5",
            environment=SANDBOX,
            jobs_dir=JOBS / "claude",
        )
    )
    print(claude, "tokens", claude.total_tokens)
    for event in claude.trajectory:
        if event.get("type") == "tool_call":
            print("  tool:", event.get("title"), event.get("status"))

# %% [markdown]
# ## 3. Several runs at once
# `bf.run_batch` runs configs with bounded concurrency and returns a
# `Results` list in input order; `to_csv`/`to_jsonl`/`to_records` export it.

# %%
if not OFFLINE:
    configs = [
        bf.RolloutConfig(
            task_path=TASK, agent="oracle", environment=SANDBOX, jobs_dir=JOBS / "batch"
        )
        for _ in range(2)
    ]
    results = bf.run_batch(configs, concurrency=2)
    print(
        results.n_passed, "of", len(results), "passed; mean reward", results.mean_reward
    )
    print("wrote", results.to_csv(JOBS / "batch.csv"))


# %% [markdown]
# ## 4. Read finished jobs back, and compare two
# `bf.load_job` reads any job folder (or a list of them) into typed trials:
# rewards, verifier output, costs, trajectories and branch lineage.
# `Job.denominators()` counts attempted, scored, verifier errors and unscored
# runs separately and leaves control runs (oracle, empty) out, as the viewer
# does. Offline, two small stand-in jobs are written so the cell runs.


# %%
def _stand_in_job(root: Path, rewards: dict[str, float]) -> Path:
    for task, reward in rewards.items():
        trial = root / f"{task}__0000000{len(task)}"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task,
                    "rollout_name": trial.name,
                    "agent": "claude-agent-acp",
                    "model": "claude-haiku-4-5",
                    "rewards": {"reward": reward},
                }
            )
        )
    return root


if OFFLINE:
    job_a = _stand_in_job(JOBS / "stand-in-a", {"task-1": 1.0, "task-2": 0.0})
    job_b = _stand_in_job(JOBS / "stand-in-b", {"task-1": 1.0, "task-2": 1.0})
else:
    job_a = JOBS / "oracle"
    job_b = JOBS / ("claude" if HAS_CLAUDE else "batch")

a = bf.load_job(job_a)
print(a)  # solve rate with its interval, unscored trials, cost
print(a.denominators(include_controls=True))
for trial in a.trials:
    print(
        " ",
        trial.task_name,
        trial.reward,
        trial.execution,
        trial.assessment,
        trial.control,
    )

comparison = bf.compare(job_a, job_b, include_controls=True)
print(comparison.to_markdown())

# %% [markdown]
# ## 5. Next
# - Many tasks with retries and resume: `bf.Evaluation(...).run_sync()`,
#   `evaluation.stream()`, `bf.Evaluation.resume(job_dir)`.
# - Branch a run into scored children: `bf.branch(...)`.
# - Save a job config for the CLI: `evaluation.to_yaml("job.yaml")`, then
#   `bench eval run --config job.yaml`.
# - The gallery: `docs/examples/python-sdk/README.md`.
