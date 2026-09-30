"""Launch the TRL GRPO cookbook on Hugging Face Jobs.

    export HF_TOKEN=...            # write access to the namespace
    export DAYTONA_API_KEY=...     # sandboxes for rollouts and evaluation
    python docs/examples/rl/trl-hf-jobs/launch.py \\
        --namespace benchflow --tasks tasks/v2 --run-id qwen3-4b-grpo-001

This uploads the inputs to a private dataset repo
(``<namespace>/bf-cookbook-trl-hf-jobs``): the task family archive, the job's
code (this folder plus ``../common``), and, before BenchFlow 0.8 is on PyPI, a
BenchFlow wheel (``--benchflow-wheel``). Then it starts one Job on one GPU
that trains, evaluates the base and trained models on the test split, and
uploads every artifact to ``runs/<run-id>/`` in that repo. The trained model
goes to the private model repo ``--output-repo``.

Secrets reach the Job as Job secrets, read from your environment; they never
appear on a command line, in a repo, or in a sandbox. ``--dry-run`` prints the
Job's configuration without uploading or starting anything.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--namespace", required=True, help="Hub user or organization that pays"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="task family folder with train/ and test/",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--inputs-repo",
        help="private dataset repo (default <namespace>/bf-cookbook-trl-hf-jobs)",
    )
    parser.add_argument(
        "--output-repo",
        help="private model repo (default <namespace>/bf-cookbook-<run-id>)",
    )
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument(
        "--benchflow-wheel", type=Path, help="install BenchFlow from this wheel"
    )
    parser.add_argument("--dry-run", action="store_true")
    args, job_args = parser.parse_known_args(argv)
    args.job_args = job_args  # passed through to job.py (for example --max-steps 40)
    args.inputs_repo = args.inputs_repo or f"{args.namespace}/bf-cookbook-trl-hf-jobs"
    args.output_repo = args.output_repo or f"{args.namespace}/bf-cookbook-{args.run_id}"
    return args


def _tasks_archive(tasks: Path) -> bytes:
    for split in ("train", "test"):
        if not (tasks / split).is_dir():
            raise SystemExit(f"{tasks} has no {split}/ folder")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for split in ("train", "test"):  # never the control task
            tar.add(tasks / split, arcname=f"{tasks.name}/{split}")
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("HF_TOKEN", "DAYTONA_API_KEY"):
        if not os.environ.get(name):
            print(f"error: set {name} in the environment", file=sys.stderr)
            return 2
    spec = "benchflow[trl,sandbox-daytona]>=0.8"
    if args.benchflow_wheel:
        spec = f"benchflow[trl,sandbox-daytona] @ file:///inputs/wheels/{args.benchflow_wheel.name}"
    command = [
        "bash", "/inputs/code/bootstrap.sh",
        "--run-id", args.run_id, "--tasks", "/inputs/tasks/tasks.tar.gz",
        "--output-repo", args.output_repo, "--runs-repo", args.inputs_repo,
        "--daytona-owner", f"bf-cookbook-trl-{args.run_id}"[:48],
        *args.job_args,
    ]  # fmt: skip
    print(f"job: {args.flavor}, timeout {args.timeout}, namespace {args.namespace}")
    print(f"inputs: {args.inputs_repo} (private); model: {args.output_repo} (private)")
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi, Volume, run_job

    api = HfApi()
    api.create_repo(args.inputs_repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.inputs_repo, repo_type="dataset").private:
        raise SystemExit(f"{args.inputs_repo} is public; refusing to upload")
    api.upload_file(
        path_or_fileobj=_tasks_archive(args.tasks),
        path_in_repo="tasks/tasks.tar.gz",
        repo_id=args.inputs_repo,
        repo_type="dataset",
    )
    for folder in (HERE, COMMON):
        api.upload_folder(
            folder_path=str(folder),
            path_in_repo="code",
            repo_id=args.inputs_repo,
            repo_type="dataset",
            allow_patterns=["*.py", "*.sh"],
        )
    if args.benchflow_wheel:
        api.upload_file(
            path_or_fileobj=str(args.benchflow_wheel),
            path_in_repo=f"wheels/{args.benchflow_wheel.name}",
            repo_id=args.inputs_repo,
            repo_type="dataset",
        )
    job = run_job(
        image=IMAGE,
        command=command,
        env={
            "BENCHFLOW_SPEC": spec,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "BENCHFLOW_DAYTONA_AUTO_STOP_MINS": "30",
            "BENCHFLOW_DAYTONA_AUTO_DELETE_MINS": "30",
        },
        secrets={name: os.environ[name] for name in ("HF_TOKEN", "DAYTONA_API_KEY")},
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        name=f"bf-cookbook-trl-{args.run_id}",
        labels={"cookbook": "benchflow-trl-grpo", "run": args.run_id},
        volumes=[Volume(type="dataset", source=args.inputs_repo, mount_path="/inputs")],
    )
    print(f"started job {job.id}: {job.url}")
    print(f"follow: hf jobs logs {job.id} --namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
