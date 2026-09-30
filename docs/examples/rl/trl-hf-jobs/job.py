"""GRPO on BenchFlow tasks with TRL, inside one Hugging Face Job.

``launch.py`` starts this through ``bootstrap.sh``. Phases:

- ``train``: GRPO with LoRA on the task family's train split, with BenchFlow's
  TRL adapter. Every rollout runs in its own Daytona sandbox; the verifier's
  reward goes back to TRL; infrastructure failures are dropped, not scored.
  The merged model goes to a private Hub repo.
- ``eval``: serve the base model and the trained model with vLLM, one after
  the other, and evaluate each on the test split with the shared
  ``evaluate.py`` (same run_bash/submit harness as training).
- ``all``: both.

Artifacts (metrics, the reward curve, eval summaries and episodes, every
rollout's audit record and folder, the job log) are uploaded to the runs
dataset repo under ``runs/<run-id>/``, after a scan for the Job's secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

WORK = Path("/work")
OUT = WORK / "out"
TASKS = WORK / "tasks"
SECRET_ENV = ("HF_TOKEN", "DAYTONA_API_KEY")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tasks", default="/inputs/tasks/tasks.tar.gz")
    parser.add_argument(
        "--output-repo", required=True, help="private model repo for the result"
    )
    parser.add_argument(
        "--runs-repo", required=True, help="private dataset repo for artifacts"
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument(
        "--eval-model", help="evaluate this model instead of the one just trained"
    )
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--eval-concurrency", type=int, default=16)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--daytona-owner", default="bf-cookbook-trl")
    parser.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[job {time.strftime('%H:%M:%S')}] {message}", flush=True)


def unpack_tasks(archive: str) -> None:
    if (TASKS / "train").is_dir():
        return
    TASKS.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(TASKS, filter="data")
    # The archive holds one family folder (for example v2/) with train/ and test/.
    inner = [p for p in TASKS.iterdir() if p.is_dir()]
    if not (TASKS / "train").is_dir() and len(inner) == 1:
        for child in inner[0].iterdir():
            child.rename(TASKS / child.name)
    log(f"tasks: {sorted(p.name for p in TASKS.iterdir())}")


# ---------------------------------------------------------------------------
# Train


def train(args: argparse.Namespace) -> Path:
    from harness import MAX_TURNS, harness_config
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from benchflow.integrations.trl import BenchFlowSpec

    rollouts = OUT / "rollouts"
    spec = BenchFlowSpec(
        tasks_dir=TASKS / "train",
        bash_harness=harness_config(
            environment="daytona", jobs_dir=rollouts, background_start=True
        ),
    )
    log(f"train split: {len(spec.train_dataset_rows)} tasks")

    metrics_path = OUT / "metrics.jsonl"

    class Metrics(TrainerCallback):
        """Write each log line to metrics.jsonl and push progress every few steps."""

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"step": state.global_step, **logs}) + "\n")
            if state.global_step % 5 == 0:
                upload(args, ["metrics.jsonl", "job.log", "rollouts/rollouts.jsonl"])

    config = GRPOConfig(
        output_dir=str(OUT / "trainer"),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        # Four rollouts per forward pass keeps long multi-turn completions within
        # memory; one optimizer step still covers prompts_per_step groups.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=args.prompts_per_step * args.num_generations // 4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=MAX_TURNS,
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=spec.train_dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=spec.reward_funcs,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        callbacks=[Metrics()],
    )
    log("training")
    trainer.train()
    diagnostics = training_diagnostics(trainer)
    (OUT / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1) + "\n"
    )
    if diagnostics["steps_with_signal"] == 0:
        # Every group had equal rewards at every step: the adapter never moved.
        raise RuntimeError(
            "no GRPO group had a reward spread; refusing to publish a no-op model"
        )
    log("training done; merging the LoRA adapter")
    merged = trainer.model.merge_and_unload()
    model_dir = OUT / "model"
    merged.save_pretrained(model_dir, safe_serialization=True)
    # The base tokenizer and its chat template, not TRL's training variant.
    AutoTokenizer.from_pretrained(args.model).save_pretrained(model_dir)
    (model_dir / "README.md").write_text(model_card(args))

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.output_repo, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=args.output_repo, folder_path=str(model_dir), repo_type="model"
    )
    log(f"pushed the merged model to the private repo {args.output_repo}")
    del trainer, merged
    import torch

    torch.cuda.empty_cache()
    return model_dir


def training_diagnostics(trainer: Any) -> dict[str, Any]:
    """Whether training had a learning signal, and whether the adapter moved."""

    history = [h for h in trainer.state.log_history if "reward" in h]
    zero_std = [h.get("frac_reward_zero_std", 1.0) for h in history]
    norms = [
        float(param.detach().float().norm())
        for name, param in trainer.model.named_parameters()
        if "lora_B" in name
    ]
    return {
        "steps": len(history),
        "steps_with_signal": sum(1 for z in zero_std if z < 1.0),
        "mean_frac_reward_zero_std": sum(zero_std) / len(zero_std)
        if zero_std
        else None,
        "lora_b_tensors": len(norms),
        "lora_b_nonzero": sum(1 for n in norms if n > 0),
        "lora_b_norm_mean": sum(norms) / len(norms) if norms else None,
    }


def model_card(args: argparse.Namespace) -> str:
    return (
        "---\nlibrary_name: transformers\ntags: [benchflow, grpo, trl]\n---\n\n"
        f"# {args.output_repo}\n\n"
        f"`{args.model}` trained with GRPO (TRL 1.8, LoRA r={args.lora_r}, merged) on the "
        "train split of BenchFlow's RL cookbook task family, with BenchFlow's TRL adapter and "
        f"Daytona sandboxes. Run `{args.run_id}`: {args.max_steps} steps, "
        f"{args.prompts_per_step} tasks x {args.num_generations} rollouts per step, "
        f"learning rate {args.learning_rate}. Metrics and evaluation: dataset repo "
        f"`{args.runs_repo}`, folder `runs/{args.run_id}/`.\n"
    )


# ---------------------------------------------------------------------------
# Evaluate


def serve(
    args: argparse.Namespace, model: str, key: str, port: int
) -> subprocess.Popen:
    command = [
        args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", "policy", "--port", str(port),
        "--api-key", key, "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--max-model-len", "16384", "--gpu-memory-utilization", "0.85",
    ]  # fmt: skip
    log_file = (OUT / f"vllm-{port}.log").open("w")
    # FlashInfer's sampler compiles a CUDA kernel at first use, and the Job image
    # has no nvcc: sample with PyTorch instead.
    env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with {process.returncode}; see vllm-{port}.log"
            )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return process
        except OSError:
            pass
        time.sleep(5)
    stop(process)
    raise RuntimeError("vLLM did not come up within 15 minutes")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def evaluate(args: argparse.Namespace, name: str, model: str) -> dict[str, Any]:
    key = secrets.token_hex(16)
    process = serve(args, model, key, 8000)
    out = OUT / "eval" / name
    command = [
        sys.executable, str(CODE / "evaluate.py"),
        "--tasks-dir", str(TASKS / "test"),
        "--base-url", "http://127.0.0.1:8000/v1", "--model", "policy",
        "--api-key-env", "POLICY_API_KEY", "--sandbox", "daytona",
        "--concurrency", str(args.eval_concurrency), "--samples", str(args.eval_samples),
        "--owner", args.daytona_owner, "--out", str(out),
    ]  # fmt: skip
    if args.eval_limit:
        command += ["--limit", str(args.eval_limit)]
    log(f"evaluating {name} ({model})")
    child = subprocess.Popen(command, env={**os.environ, "POLICY_API_KEY": key})
    try:
        if child.wait() != 0:
            raise RuntimeError(f"evaluate.py exited with {child.returncode}")
    finally:
        if child.poll() is None:  # interrupted: let it stop its episodes and sweep
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
        stop(process)
    return json.loads((out / "summary.json").read_text())


def paired_difference(base: Path, trained: Path, reps: int = 2000) -> dict[str, Any]:
    """Trained minus base solve rate, with a 95% bootstrap interval over tasks."""

    def per_task(path: Path) -> dict[str, list[float]]:
        rows: dict[str, list[float]] = {}
        for line in (path / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("reward") is not None:
                rows.setdefault(row["task_id"], []).append(
                    1.0 if row["reward"] >= 1 else 0.0
                )
        return rows

    a, b = per_task(base), per_task(trained)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return {"tasks": 0}

    def diff(sample: list[str]) -> float:
        base_rate = sum(sum(a[t]) / len(a[t]) for t in sample) / len(sample)
        trained_rate = sum(sum(b[t]) / len(b[t]) for t in sample) / len(sample)
        return trained_rate - base_rate

    rng = random.Random(0)
    draws = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(reps))
    return {
        "tasks": len(tasks),
        "difference": diff(tasks),
        "ci95": [draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]],
        "method": "per-task solve rates, paired bootstrap over tasks",
    }


# ---------------------------------------------------------------------------
# Reward curve and uploads


def plot_rewards() -> None:
    rows = [
        json.loads(line) for line in (OUT / "metrics.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if "reward" in r]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    reward = [r["reward"] for r in rows]
    dropped = [r.get("benchflow/dropped_frac", 0.0) for r in rows]
    window = 5
    smooth = [
        sum(reward[max(0, i - window + 1) : i + 1])
        / len(reward[max(0, i - window + 1) : i + 1])
        for i in range(len(reward))
    ]
    ink, muted, grid, surface, series = (
        "#0b0b0b",
        "#52514e",
        "#e6e5e1",
        "#fcfcfb",
        "#2a78d6",
    )
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        facecolor=surface,
    )  # fmt: skip
    for ax in (top, bottom):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, length=0)
    top.plot(
        steps, reward, "o", color=series, alpha=0.35, markersize=5, markeredgewidth=0
    )
    top.plot(steps, smooth, color=series, linewidth=2, solid_capstyle="round")
    top.set_ylim(0, 1)
    top.set_title(
        "Mean reward per GRPO step (dots) and its 5-step mean (line)",
        loc="left", color=ink, fontsize=11,
    )  # fmt: skip
    top.annotate(
        f"{smooth[-1]:.2f}", (steps[-1], smooth[-1]), xytext=(6, 0),
        textcoords="offset points", color=ink, va="center", fontsize=9,
    )  # fmt: skip
    bottom.bar(steps, dropped, color=muted, width=0.6)
    bottom.set_ylim(0, max(0.1, max(dropped)))
    bottom.set_title("Share of rollouts dropped as infrastructure failures", loc="left", color=ink, fontsize=9)  # fmt: skip
    bottom.set_xlabel("step", color=muted)
    fig.tight_layout()
    fig.savefig(OUT / "reward_curve.png", dpi=150)
    with (OUT / "reward_curve.csv").open("w") as handle:
        handle.write("step,reward,reward_5step_mean,dropped_frac\n")
        for s, r, m, d in zip(steps, reward, smooth, dropped, strict=True):
            handle.write(f"{s},{r:.4f},{m:.4f},{d:.4f}\n")


def _secret_values() -> list[bytes]:
    return [
        os.environ[name].encode()
        for name in SECRET_ENV
        if len(os.environ.get(name, "")) >= 12
    ]


def _leaks(path: Path, values: list[bytes]) -> bool:
    data = path.read_bytes()
    return any(value in data for value in values)


def upload(args: argparse.Namespace, names: list[str] | None = None) -> None:
    """Upload artifacts to runs/<run-id>/, skipping any file that holds a secret."""

    from huggingface_hub import HfApi

    values = _secret_values()
    stage = WORK / "upload"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    candidates = (
        [OUT / n for n in names]
        if names
        else [p for p in OUT.rglob("*") if p.is_file()]
    )
    skipped = 0
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(OUT)
        if relative.parts[0] in ("model", "trainer") or (relative.parts[0] == "rollouts" and len(relative.parts) > 2):  # fmt: skip
            continue
        if _leaks(path, values):
            skipped += 1
            continue
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    if names is None and (OUT / "rollouts").is_dir():
        # Every rollout folder, dropped ones included, as one archive for audit.
        archive = stage / "rollouts.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(OUT / "rollouts", arcname="rollouts")
        if _leaks(archive, values):
            archive.unlink()
            skipped += 1
    if skipped:
        log(f"upload: skipped {skipped} file(s) that contained a secret value")
    api = HfApi()
    api.create_repo(args.runs_repo, private=True, exist_ok=True, repo_type="dataset")
    api.upload_folder(
        repo_id=args.runs_repo, folder_path=str(stage), path_in_repo=f"runs/{args.run_id}",
        repo_type="dataset", commit_message=f"{args.run_id}: artifacts",
    )  # fmt: skip


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from harness import shorten_daytona_lifetimes, sweep_daytona

    os.environ["BENCHFLOW_DAYTONA_OWNER"] = args.daytona_owner
    shorten_daytona_lifetimes()
    # A rerun of the same run id first clears anything a killed attempt left.
    log(
        f"Daytona owner {args.daytona_owner}: swept {sweep_daytona(args.daytona_owner)}"
    )

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    unpack_tasks(args.tasks)
    (OUT / "config.json").write_text(json.dumps(vars(args), indent=1) + "\n")
    trained: Path | str | None = args.eval_model
    try:
        if args.phase in ("train", "all"):
            trained = train(args)
            plot_rewards()
        if args.phase in ("eval", "all"):
            if trained is None:
                trained = args.output_repo
            results = {
                "base": evaluate(args, "base", args.model),
                "trained": evaluate(args, "trained", str(trained)),
            }
            results["difference"] = paired_difference(
                OUT / "eval" / "base", OUT / "eval" / "trained"
            )
            (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
            for name in ("base", "trained"):
                summary = results[name]
                log(
                    f"{name}: pass rate {summary['solve_rate']} ci95 {summary['ci95']}, "
                    f"mean reward {summary['mean_reward']}, dropped {summary['drop_reasons']}"
                )
            log(f"difference: {results['difference']}")
    finally:
        log(
            f"Daytona owner {args.daytona_owner}: swept {sweep_daytona(args.daytona_owner)}"
        )
        if (OUT / "metrics.jsonl").is_file():
            plot_rewards()
        upload(args)
        log(f"artifacts uploaded to {args.runs_repo} runs/{args.run_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
