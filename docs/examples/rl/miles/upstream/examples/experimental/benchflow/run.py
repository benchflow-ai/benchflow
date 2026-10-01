"""Launcher: GRPO on BenchFlow tasks for a small Qwen3 on one GPU (FSDP, colocated).

Start the BenchFlow environment server first (see the README), then:

    python examples/experimental/benchflow/run.py \\
        --model-name Qwen3-1.7B --prompt-data /root/benchflow/train.jsonl \\
        --num-rollout 30 --save-dir /root/checkpoints/benchflow

Each rollout step trains on ``rollout_batch_size`` prompt groups of
``n_samples_per_prompt`` BenchFlow episodes; groups whose episodes all scored
the same are replaced (see launch_common.agentic_train_args).
"""

import json
import os
from dataclasses import dataclass
from typing import Literal

import typer
from launch_common import agentic_pythonpath_dirs, agentic_train_args, benchflow_env_vars, preflight

from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    num_gpus_per_node: int = 1
    model_name: str = "Qwen3-1.7B"
    hf_org: str = "Qwen"
    model_dir: str = "/root/models"
    skip_prepare: bool = False
    save_dir: str = "/root/checkpoints/benchflow"
    prompt_data: str = "/root/benchflow/train.jsonl"

    # Training settings
    num_rollout: int = 30
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    # Context per episode, and tokens per reply. The server ends an episode
    # before its context passes max_seq_len; a reply cut at max_response_len
    # ends it too (both are logged as benchflow/* metrics).
    max_seq_len: int = 16384
    max_response_len: int = 1024
    rollout_temperature: float = 1.0
    lr: float = 1e-6
    save_interval: int = 1000
    tito_model: str = "qwen3"
    # Qwen3 hybrids think by default; the BenchFlow cookbook trains and
    # evaluates without thinking.
    enable_thinking: bool = False
    drop_constant_reward_groups: bool = True
    sglang_mem_fraction: float = 0.6
    # FlashAttention 3 needs Hopper (H100, H200); use flash_attention_2 on an A100.
    attn_implementation: str = "flash_attention_3"
    session_server_workers: int = 4
    tensorboard_dir: str = ""

    # BenchFlow environment server
    benchflow_env_url: str = os.environ.get("BENCHFLOW_ENV_URL", "http://127.0.0.1:12100")
    benchflow_token_file: str = os.environ.get("BENCHFLOW_ENV_TOKEN_FILE", "")
    benchflow_episode_timeout: int = 3600
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {args.hf_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")


def execute(args: ScriptArgs):
    preflight(
        env_url=args.benchflow_env_url,
        prompt_data=args.prompt_data,
        token_file=args.benchflow_token_file,
    )
    U = args.create_backend()
    model_path = f"{args.model_dir}/{args.model_name}"

    ckpt_args = (
        f"--hf-checkpoint {model_path} "
        f"--ref-load {model_path} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
        "--no-save-optim "
    )
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-max-response-len {args.max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
    )
    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction} "
        "--sglang-decode-log-interval 1000 "
    )
    # The TITO family's SGLang parsers: without the tool-call parser, tool calls come back as text.
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(args.tito_model)
    template_kwargs = json.dumps({"enable_thinking": args.enable_thinking})
    agent_args = (
        agentic_train_args(
            tito_model=args.tito_model,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            session_server_workers=args.session_server_workers,
            drop_constant_reward_groups=args.drop_constant_reward_groups,
        )
        + f"--apply-chat-template-kwargs '{template_kwargs}' "
    )
    train_backend_args = (
        "--train-backend fsdp "
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        f"--attn-implementation {args.attn_implementation} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )
    perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_seq_len} "
    misc_args = f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    tb_args = f"--use-tensorboard --tb-project-name {args.tensorboard_dir} " if args.tensorboard_dir else ""

    extra_env_vars = {
        "PYTHONPATH": ":".join([*agentic_pythonpath_dirs(), str(command_utils.repo_base_dir)]),
        **benchflow_env_vars(
            env_url=args.benchflow_env_url,
            token_file=args.benchflow_token_file,
            episode_timeout_s=args.benchflow_episode_timeout,
        ),
    }
    U.execute_train(
        train_args=(
            f"{ckpt_args}{rollout_args}{grpo_args}{optimizer_args}{sglang_args}{agent_args}"
            f"{train_backend_args}{perf_args}{misc_args}{debug_args}{tb_args}{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
