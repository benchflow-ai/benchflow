# Connect a trainer

BenchFlow runs the rollouts and scores them; your trainer learns from them. A trainer can read rollouts while a job runs, read rows written from finished jobs, or, for supervised fine-tuning with Prime-RL, be launched by BenchFlow.

| Connection | Command | For |
|---|---|---|
| a stream of finished rollouts | `bench train stream` | an RL trainer that consumes rollouts as they finish |
| rows from finished jobs | `bench train convert` | SFT, or RL from a fixed set of rollouts: [Export training data](./export.md) |
| a Prime-RL SFT job | `bench train run sft`, from a Prime-RL checkout of yours; not run in these docs ([Commands](../reference/commands.md#bench-train-run-sft)) | supervised fine-tuning on those rows |

## Stream rollouts while a job runs

`bench train stream` prints one JSON line per finished rollout, and keeps watching until the job ends; `--no-follow` reads once and stops. Point it at a job, or at the folder that holds a `--trials` run's repeats:

```bash
bench train stream jobs/no-skills/claude-agent-acp
```

Status goes to stderr (`bench train stream: 9 rollouts streamed`); each line on stdout is a `benchflow.rollout-stream.v1` record. One, with its per-call lists left out:

```json
{
  "schema_version": "benchflow.rollout-stream.v1",
  "rollout": "crystallographic-wyckoff-position-analysis__6a869c0f",
  "rollout_path": "trial-01/2026-11-04__17-52-13/crystallographic-wyckoff-position-analysis__6a869c0f",
  "task": "crystallographic-wyckoff-position-analysis",
  "agent": "claude-agent-acp",
  "model": "xiaomi/mimo-v2.6-flash",
  "reward": 0.0,
  "scored": true,
  "outcome": "failed",
  "error": "Agent prompt exceeded wall-clock budget 900s",
  "token_capture": {"calls": 18, "captured_calls": 0, "training_grade": false, "status": "capture_off"},
  "advantage": null,
  "retried": false
}
```

The exit code says how the job ended: 0 when it finished, 1 when it was stopped or its process died first, 2 for bad input, 3 when `--timeout` ran out. The record's schema is in [Trajectory and rollout formats](./export.md#schema-versions).

## Groups and advantages

For GRPO-style training, `--group-size N` holds rollouts until N of a group have finished, then emits them with an advantage: the reward minus the group's mean, over the group's standard deviation. Group the three repeats of each task:

```bash
bench train stream jobs/no-skills/claude-agent-acp --group-size 3 --group-by task,agent,model
```

The `rollout_path`, `reward` and `advantage` of each line:

```text
trial-01/.../econ-detrending-correlation__6cecf5e4                  reward 1.0  advantage 0.0
trial-02/.../econ-detrending-correlation__7bf74511                  reward 1.0  advantage 0.0
trial-03/.../econ-detrending-correlation__2f5ef2ef                  reward 1.0  advantage 0.0
...
trial-01/.../crystallographic-wyckoff-position-analysis__6a869c0f   reward 0.0  advantage 0.0
```

Every advantage is 0: each task scored the same in all three tries, so the group has nothing to learn from. A group is useful for this kind of training only when its rewards differ: some rollouts pass and others fail, or partial credit spreads the rewards out. [BenchFlow as an RL environment](./environment.md#rewards-and-advantages) shows a group whose rewards differ.

`--group-by` takes `task`, `agent`, `model`, `task_digest` and `job`. Leave `job` out for `--trials` repeats, since each repeat is a job of its own.

A group reaches the trainer whole or not at all. The stream withholds a group when a member is unscored, when a member's proxy log has no model calls (its run never reached the model), when some members have a token sequence and the others don't (a prompt that re-tokenized an earlier reply, see [Token ids and logprobs](#token-ids-and-logprobs)), or when the job ends before the group fills. A scored rollout that ended with an error, such as a time-out, stays in its group, and its record carries the error. `bench train convert --group-advantage` keeps the same groups whole. None of its lines reach stdout; `--withheld FILE` writes them, each with a `withheld` field saying why, and the last status line counts them:

```text
bench train stream: 9 rollouts streamed; 1 group withheld (3 rollouts), written to withheld.jsonl
```

From Python, `stream_rollouts(..., on_withheld=callback)` gets each withheld group as one list.

## Token ids and logprobs

Training on what the model actually generated needs each call's token ids and logprobs. BenchFlow's model proxy records them when the model is one you serve from vLLM or SGLang ([A model you serve](../evaluate/agents.md#a-model-you-serve)) and you pass `--agent-env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`. A hosted API's calls carry no token ids, so rollouts against one can train only on text.

A small model serves on a CPU. We ran this on a cloud machine with no GPU, with BenchFlow 0.9.1, vLLM 0.33.0's CPU image and `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
BRIDGE=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)   # Docker's bridge
docker run -d --name policy --shm-size 4g -p $BRIDGE:8000:8000 \
  -v $PWD/hf-cache:/root/.cache/huggingface \
  -e VLLM_CPU_KVCACHE_SPACE=4 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm/vllm-openai-cpu:v0.33.0-x86_64 Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name my-policy --host 0.0.0.0 --port 8000 --dtype float32 \
  --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
  --api-key dummy-key
```

`-p $BRIDGE:8000:8000` publishes the port on Docker's bridge, where BenchFlow's model proxy reaches it, and nowhere else; the first line reads the bridge's address on Linux. `VLLM_USE_V2_MODEL_RUNNER=0` matters on a CPU: with vLLM 0.33.0's default runner the server was still compiling kernels after six minutes; with it, the server was ready in about 90 seconds. Agents call tools, hence the last two flags before the key.

Then run a task through it. This one asks the agent to write 42 to `/app/answer.txt`, and its test checks the file. These commands make it, with the files we ran:

```bash
mkdir -p tasks/write-number/verifier tasks/write-number/oracle
cat > tasks/write-number/task.md <<'EOF'
---
schema_version: "1.3"
metadata:
  author_name: BenchFlow docs
  category: demo
  tags: [rl, token-capture]
sandbox:
  docker_image: python:3.12-slim
  workdir: /app
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 60
---
# write-number

## prompt

Write the number 42 to the file `/app/answer.txt`.
EOF
cat > tasks/write-number/verifier/test.sh <<'EOF'
#!/bin/bash
# Reward 1 when /app/answer.txt holds 42 (surrounding whitespace ignored), else 0.
if [ "$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)" = "42" ]; then
  echo PASS; echo 1 > /logs/verifier/reward.txt
else
  echo "FAIL: /app/answer.txt is $(head -c 80 /app/answer.txt 2>/dev/null || echo missing)"; echo 0 > /logs/verifier/reward.txt
fi
EOF
printf '#!/bin/bash\necho 42 > /app/answer.txt\n' > tasks/write-number/oracle/solve.sh
chmod +x tasks/write-number/verifier/test.sh tasks/write-number/oracle/solve.sh
```

Then run it through the server, with Pi (`pi-acp`, the agent `-a pi` also names):

```bash
bench run tasks/write-number -a pi-acp -m vllm/my-policy --sandbox docker --jobs-dir jobs --job rl-capture \
  --agent-env BENCHFLOW_PROVIDER_BASE_URL=http://$BRIDGE:8000/v1 \
  --agent-env BENCHFLOW_PROVIDER_API_KEY=dummy-key \
  --agent-env BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1
bench train token-coverage jobs/rl-capture
```

```text
[write-number]   → end_turn, 1 total tool calls
[write-number] Rewards: {'reward': 1.0}
[PASS] write-number (reward=1.00, tools=1)
...
yes  write-number__5cfd1b6d [vllm]: 2/2 calls complete, 1/1 prompts extend the previous call
1 of 1 rollouts are training-grade (complete token ids and logprobs, token-in/token-out)
```

`yes` means both calls have their prompt ids, their sampled ids and one logprob per sampled id, and the second call's prompt (1,450 ids) starts with the first call's prompt and sampled ids (1,392 and 29). A rollout whose calls skipped capture shows `no` and says why, such as `calls went through the gateway without token capture`.

Each call's capture is in `metadata.token_capture` on its line of the trial's `trajectory/llm_trajectory.jsonl`. The first call, cut short:

```json
{
  "schema_version": "benchflow.token_capture.v1",
  "provider": "vllm",
  "prompt_token_ids": [151644, 8948, 198, 2610, 525, 458, "... 1,392 ids"],
  "completions": [
    {"index": 0,
     "token_ids": [151657, 198, 4913, 606, 788, 330, "... 29 ids"],
     "logprobs": [-0.0036, -0.0001, -0.0, -0.0001, -0.0, -0.0, "..."]}
  ],
  "unavailable": {}
}
```

Its 29 sampled ids decode to the tool call as the model wrote it:

```text
<tool_call>\n{"name": "write", "arguments": {"path": "/app/answer.txt", "content": "42"}}\n</tool_call><|im_end|>
```

Both calls were streamed. Read a streamed call's logprobs here: the proxy drops them from the raw response body. As a check, Hugging Face transformers ran the same weights over the same ids and gave the same logprobs to within 1.2e-6.

## Train one step with TRL

TRL is Hugging Face's training library. Convert the rollout to TRL rows with one row per model call, so the tool call is something the trainer learns; the default, one row per rollout, puts the tool call in the prompt and trains only the last reply:

```bash
bench train convert jobs/rl-capture --format trl-sft --row-mode exchange --out rows/rl-capture.jsonl
```

```text
Converted 2 row(s) from 1 rollout(s) -> rows/rl-capture.jsonl
```

`bench train validate rows/rl-capture.jsonl --format trl-sft --tokenizer Qwen/Qwen2.5-0.5B-Instruct` renders each row with the model's chat template and counts its tokens. It needs the `trl` extra (`benchflow[trl]`); without it the command stops with `tokenizer validation requires the benchflow train or trl extra`.

These lines train one step on CPU, here on TRL's 2.4-million-parameter test model. We ran them in this environment, which resolved torch 2.15.0+cpu, transformers 5.22.1, trl 1.17.1 and datasets 5.1.0:

```bash
uv venv --python 3.12 trl-venv
uv pip install --python trl-venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch trl datasets accelerate
```

Then, with `trl-venv/bin/python`:

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

rows = load_dataset("json", data_files="rows/rl-capture.jsonl", split="train")
rows = rows.select_columns(["prompt", "completion", "tools"])
args = SFTConfig(output_dir="trl-out", max_steps=1, per_device_train_batch_size=2, max_length=None,
                 report_to=[], logging_steps=1, save_strategy="no", use_cpu=True, seed=0)
trainer = SFTTrainer(model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=args, train_dataset=rows)
print(f"training loss: {trainer.train().training_loss:.4f}")
```

```text
{'loss': '11.93', 'grad_norm': '0.5562', 'learning_rate': '2e-05', 'entropy': '11.93', 'mean_token_accuracy': '0', 'num_tokens': '2892', 'epoch': '1'}
training loss: 11.9326
```

The step trained on 50 tokens: the 29 of the tool call and the 19 of the reply, each with the newline the chat template adds after it. The loss says the step ran and nothing more: the test model guesses almost evenly over its 152,064 tokens, and ln 152,064 is 11.932. The rows hold messages, not token ids, so TRL tokenizes them again; for this model its tokens matched the captured ones exactly.

## Other trainers

BenchFlow 0.9.2 has no adapter for other trainers. 0.10.0rc2's `bench train convert` also writes `verl-sft` (parquet for verl's multi-turn SFT trainer), `skyrl-sft` (rows for SkyRL's SFT trainer) and `openenv-episodes` (OpenEnv episode records). On 0.9.2, connect one through the files:

- `bench train convert --format trl-sft` writes rows TRL's SFT trainer reads; `prime-sft` is the default.
- Each trial's `trainer/verifiers.jsonl` holds one verifiers-style row: prompt, completion, reward.
- The stream above is plain JSON lines with a published schema, for a trainer of your own.
