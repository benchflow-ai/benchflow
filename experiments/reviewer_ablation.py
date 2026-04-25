"""Reviewer benchmark ablation — does a reviewer agent improve performance?

Conditions:
  A) baseline     — single-agent SDK.run()
  B) reviewer     — coder + plain reviewer (1 review loop)
  C) reviewer+spec — coder + reviewer re-reading task instruction (1 review loop)

Usage:
  env -u ANTHROPIC_API_KEY python experiments/reviewer_ablation.py

Results written to experiments/ablation-results.csv
"""
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

import contextlib

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

from benchflow._acp_run import connect_acp, execute_prompts
from benchflow._agent_env import resolve_agent_env
from benchflow._agent_setup import install_agent
from benchflow._credentials import upload_subscription_auth, write_credential_files
from benchflow._env_setup import _create_environment
from benchflow._sandbox import setup_sandbox_user
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.sdk import SDK

logger = logging.getLogger(__name__)

MODEL = os.environ.get("ABLATION_MODEL", "gemini-3.1-flash-lite-preview")
BACKEND = os.environ.get("ABLATION_BACKEND", "daytona")
AGENT = os.environ.get("ABLATION_AGENT", "gemini")

TB2_ROOT = Path(__file__).resolve().parents[1] / ".ref" / "terminal-bench-2"
TB2_TASKS = sorted([
    d.name for d in TB2_ROOT.iterdir()
    if d.is_dir() and (d / "task.toml").exists()
])

RESULTS_FILE = Path(__file__).parent / "ablation-results.csv"
JOBS_DIR = Path("/tmp/ablation-jobs")

REVIEWER_PLAIN = """You are an expert code reviewer. A coder agent has attempted a programming task.

IMPORTANT: You are a REVIEWER, not a coder. Do NOT modify, create, or delete any files in /app/ except /app/.outbox/coder.json. Only read and inspect.

Review the coder's work in /app/ systematically:

1. **Correctness**: Does the code produce the right output for all inputs? Look for off-by-one errors, unhandled edge cases, incorrect algorithm choice, and logic bugs.
2. **Completeness**: Does the solution address every requirement? Check for missing functionality and unimplemented cases.
3. **Bugs**: Trace through the code with concrete test inputs. Flag any path that produces wrong results.

Be specific and evidence-backed: reference file names and concrete failing inputs. If you are uncertain about an issue, say so — do not fabricate precision. Only report issues you can justify with evidence from the code.

Write your review to /app/.outbox/coder.json:
  {{"to": "coder", "content": "YOUR SPECIFIC FEEDBACK — cite files, failing inputs, and fixes"}}

If the code is correct and complete:
  {{"to": "coder", "content": "Code looks correct, no changes needed"}}

You MUST create /app/.outbox/coder.json before stopping."""

REVIEWER_SPEC = """You are an expert code reviewer with access to the original task specification.

IMPORTANT: You are a REVIEWER, not a coder. Do NOT modify, create, or delete any files in /app/ except /app/.outbox/coder.json. Only read and inspect.

ORIGINAL TASK SPECIFICATION:
{instruction}

A coder agent has attempted this task. Review their work in /app/:

1. **Spec compliance**: Compare the coder's output against each requirement in the specification. Note which are met and which are missing or wrong.
2. **Correctness**: Trace through the code with concrete inputs from the spec. Flag any divergence from expected output.
3. **Bugs**: Look for off-by-one errors, unhandled edge cases mentioned in the spec, incorrect parsing, and wrong output format.

Be specific and evidence-backed: reference file names, the spec requirement violated, and what the fix should be. If uncertain, say so — do not fabricate issues.

Write your review to /app/.outbox/coder.json:
  {{"to": "coder", "content": "YOUR SPECIFIC FEEDBACK — cite spec requirements, files, and fixes"}}

If the code correctly implements the specification:
  {{"to": "coder", "content": "Code matches specification, no changes needed"}}

You MUST create /app/.outbox/coder.json before stopping."""

CODER_INITIAL = """{instruction}

When you are done, create /app/.outbox/reviewer.json with:
{{"to": "reviewer", "content": "Task complete, please review"}}

You MUST create both your solution files AND /app/.outbox/reviewer.json before stopping."""

CODER_REVISION = """{instruction}

You previously attempted this task and received the following review:

REVIEWER FEEDBACK:
{feedback}

Please address the reviewer's feedback and fix any issues. Focus on the specific problems mentioned.
Do NOT create any outbox files this time — just fix the code and stop."""


_agent_installed = set()


async def _ensure_agent(env, trial_dir: Path) -> None:
    if AGENT not in _agent_installed:
        agent_config = AGENTS.get(AGENT)
        await install_agent(env, AGENT, trial_dir)
        if agent_config:
            await write_credential_files(env, AGENT, {}, agent_config, MODEL, "/home/agent")
            await upload_subscription_auth(env, AGENT, "/home/agent")
        await setup_sandbox_user(env, sandbox_user="agent", workspace="/app")
        _agent_installed.add(AGENT)


async def _run_acp(env, prompt: str, trial_dir: Path, timeout: int = 600) -> tuple[int, int]:
    """Run one ACP agent session. Returns (n_tool_calls, elapsed_sec)."""
    launch_cmd = AGENT_LAUNCH.get(AGENT, AGENT)
    agent_env = resolve_agent_env(AGENT, MODEL, None)
    agent_env.pop("_BENCHFLOW_SUBSCRIPTION_AUTH", None)
    t0 = time.time()
    acp_client, session, _ = await connect_acp(
        env=env, agent=AGENT, agent_launch=launch_cmd, agent_env=agent_env,
        sandbox_user="agent", model=MODEL, trial_dir=trial_dir,
        environment=BACKEND, agent_cwd="/app",
    )
    try:
        _, n_tools = await execute_prompts(acp_client, session, [prompt], timeout=timeout)
    finally:
        with contextlib.suppress(Exception):
            await acp_client.close()
    return n_tools, int(time.time() - t0)


async def _verify(env, task_path: Path, trial_dir: Path) -> dict | None:
    """Run verifier and return rewards dict."""
    trial_paths = TrialPaths(trial_dir)
    trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    task = Task(task_path)
    verifier = Verifier(task=task, trial_paths=trial_paths, environment=env)
    try:
        result = await asyncio.wait_for(verifier.verify(), timeout=120)
        return result.rewards
    except Exception as e:
        logger.error(f"Verifier error: {e}")
        return None


async def run_baseline(task_path: Path, task_name: str) -> dict:
    """Condition A: single-agent baseline via SDK.run() (known-working path)."""
    t0 = time.time()
    try:
        sdk = SDK()
        result = await sdk.run(
            task_path=task_path,
            agent=AGENT,
            model=MODEL,
            environment=BACKEND,
            jobs_dir=str(JOBS_DIR / "baseline"),
        )
        elapsed = int(time.time() - t0)
        reward = (result.rewards or {}).get("reward", 0.0)
        return {"reward": reward, "wall_sec": elapsed, "tool_calls": result.n_tool_calls,
                "rounds": 0, "error": result.error}
    except Exception as e:
        return {"reward": 0.0, "wall_sec": int(time.time() - t0), "tool_calls": 0,
                "rounds": 0, "error": str(e)[:100]}


async def run_reviewer(task_path: Path, task_name: str, condition: str) -> dict:
    """Condition B/C: coder + reviewer with exactly 1 review loop."""
    trial_dir = JOBS_DIR / f"{condition}/{task_name}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    task = Task(task_path)
    env = _create_environment(BACKEND, task, task_path, f"{condition}-{task_name}", None)
    t0 = time.time()
    total_tools = 0
    try:
        await env.start(force_build=False)
        await _ensure_agent(env, trial_dir)

        instruction = (task_path / "instruction.md").read_text()

        # Phase 1: Coder's first attempt
        await env.exec("mkdir -p /app/.outbox && chmod 777 /app/.outbox")
        coder_prompt = CODER_INITIAL.format(instruction=instruction)
        n_tools, _ = await _run_acp(env, coder_prompt, trial_dir)
        total_tools += n_tools

        # Read coder's outbox
        await env.exec("cat /app/.outbox/reviewer.json 2>/dev/null || echo '{}'")
        await env.exec("rm -rf /app/.outbox/*")

        # Phase 2: Reviewer
        await env.exec("mkdir -p /app/.outbox && chmod 777 /app/.outbox")
        if condition == "reviewer+spec":
            reviewer_prompt = REVIEWER_SPEC.format(instruction=instruction)
        else:
            reviewer_prompt = REVIEWER_PLAIN
        n_tools, _ = await _run_acp(env, reviewer_prompt, trial_dir)
        total_tools += n_tools

        # Read reviewer's outbox
        feedback_result = await env.exec("cat /app/.outbox/coder.json 2>/dev/null || echo '{}'")
        feedback = "{}"
        try:
            feedback_data = json.loads(feedback_result.stdout or "{}")
            feedback = feedback_data.get("content", "No specific feedback")
        except json.JSONDecodeError:
            feedback = "No structured feedback received"
        await env.exec("rm -rf /app/.outbox/*")

        # Phase 3: Coder revision
        revision_prompt = CODER_REVISION.format(instruction=instruction, feedback=feedback)
        n_tools, _ = await _run_acp(env, revision_prompt, trial_dir)
        total_tools += n_tools

        # Verify
        rewards = await _verify(env, task_path, trial_dir)
        elapsed = int(time.time() - t0)
        reward = (rewards or {}).get("reward", 0.0)
        return {"reward": reward, "wall_sec": elapsed, "tool_calls": total_tools,
                "rounds": 1, "error": None}
    except Exception as e:
        return {"reward": 0.0, "wall_sec": int(time.time() - t0), "tool_calls": total_tools,
                "rounds": 1, "error": str(e)[:100]}
    finally:
        _agent_installed.discard(AGENT)
        await env.stop(delete=True)


async def _run_task_all_conditions(benchmark: str, task_dir_root: Path, task_name: str) -> list[dict]:
    """Run all 3 conditions for one task, sequentially. Returns list of row dicts."""
    task_path = task_dir_root / task_name
    if not (task_path / "task.toml").exists():
        logger.warning(f"SKIP {task_name} — no task.toml")
        return []

    rows = []
    for condition, runner in [
        ("baseline", lambda tp, tn: run_baseline(tp, tn)),
        ("reviewer", lambda tp, tn: run_reviewer(tp, tn, "reviewer")),
        ("reviewer+spec", lambda tp, tn: run_reviewer(tp, tn, "reviewer+spec")),
    ]:
        logger.info(f"\n{'='*60}\n{benchmark} / {task_name} / {condition}\n{'='*60}")
        result = await runner(task_path, task_name)
        row = {
            "benchmark": benchmark,
            "task": task_name,
            "condition": condition,
            "model": MODEL,
            "backend": BACKEND,
            **result,
        }
        rows.append(row)
        logger.info(f"  → reward={row['reward']} wall={row['wall_sec']}s tools={row['tool_calls']} err={row['error']}")
    return rows


_PHASE_SEM = asyncio.Semaphore(int(os.environ.get("ABLATION_CONCURRENCY", "64")))


async def run_experiment(benchmark: str, task_dir_root: Path, task_names: list[str]) -> list[dict]:
    """Run tasks with bounded concurrency to avoid Daytona sandbox contention."""

    async def _bounded(tn: str) -> list[dict]:
        async with _PHASE_SEM:
            return await _run_task_all_conditions(benchmark, task_dir_root, tn)

    tasks = [_bounded(tn) for tn in task_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_rows = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Task failed: {r}")
        else:
            all_rows.extend(r)
    _write_csv(all_rows)
    return all_rows


def _write_csv(rows: list[dict]) -> None:
    cols = ["benchmark", "task", "condition", "model", "backend", "rounds",
            "reward", "wall_sec", "tool_calls", "error"]
    with open(RESULTS_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


async def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== FOLLOWUP-BENCH: {len(TB2_TASKS)} TB2 tasks, agent={AGENT}, model={MODEL} ===")
    all_rows = await run_experiment("tb2", TB2_ROOT, TB2_TASKS)

    _print_table(all_rows)
    logger.info(f"\nResults: {RESULTS_FILE}")


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'benchmark':<12} {'task':<30} {'condition':<16} {'reward':>7} {'wall_sec':>9} {'tools':>6} {'err'}")
    print("-" * 95)
    for r in rows:
        err = r.get("error", "")
        err_short = (err[:20] + "…") if err and len(err) > 20 else (err or "")
        print(f"{r['benchmark']:<12} {r['task']:<30} {r['condition']:<16} {r['reward']:>7.1f} {r['wall_sec']:>8}s {r['tool_calls']:>6} {err_short}")


if __name__ == "__main__":
    asyncio.run(main())
