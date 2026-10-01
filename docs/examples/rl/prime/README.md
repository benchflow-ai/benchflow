# Cookbook: RL on BenchFlow tasks with prime-rl on Prime Intellect pods

This cookbook trains an open model on BenchFlow tasks with Prime Intellect's own stack: [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 runs the episodes and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) trains, on a GPU pod rented from Prime Intellect. Each episode's sandbox is a BenchFlow sandbox on Daytona, started, driven, and scored through BenchFlow's public primitives (`TaskRuntime`, `benchflow.integrations.rewards`) and the RL cookbooks' shared harness (`docs/examples/rl/common/`). Nothing here adds a BenchFlow command.

It has three parts: `benchflow_taskset/`, a Verifiers v1 package (taskset plus env) that runs BenchFlow tasks; `configs/rl.toml`, the prime-rl config; and `pods/` and `pod/`, the tooling that rents pods safely and runs the steps on them.

## Status and results

The environment, its tests, and the config work, and every step up to training was run on a real pod. The training run reached step 1 and then was lost: the pod reached its 8-hour limit and was terminated by the watchdog before its outputs were copied off, so there is no trained checkpoint and no after-training evaluation. All numbers below are real; they are the ones read from the pod while it ran (the files themselves are gone).

| What | Result |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served by vLLM 0.30.0 through prime-rl on 1x H200 141GB |
| Task family | rl-core's RL cookbook family, v1 (`<tasks-dir>/v1`, 200 train and 50 test tasks) |
| Tier choice (train split, evaluate.py) | v2: bugfix 6/32 (0.19), sql+csv+log 0/24, about 5% overall; v1: 11/32 = 0.34 (95% CI 0.20-0.52; bugfix 0.62, log 0.38, sql 0.38, csv 0.00). v1 is in the 20-60% band, so the run used v1 |
| Base model on v1 test | **0.335** solve rate (95% CI 0.273-0.403): 50 tasks x 4 samples, 200 kept, 0 dropped |
| Training, step 1 | batch reward 0.594, 64/64 samples trainable, 6.4 turns per episode, 26.6% truncated, 0% errors, 1 min 58 s for the step |
| Training, later steps | lost with the pod |
| Trained model on v1 test | not measured |
| Prime spend | $43.52: the H200 billed $43.47 for 8.04 hours ($5.48/h; the offer said $4.50), a CPU pod $0.01, Prime Inference $0.04 |

The step-1 reward is not an improvement over the base rate: it is measured at temperature 1.0 on training tasks that passed the zero-advantage gate, while the base rates use evaluate.py's temperature 0.7 on all tasks.

What went wrong, so it does not happen again: the pod was created with the watchdog's maximum lifetime (8 hours) instead of the run's expected length, and the session that drove it ended right after training started. Nobody copied the outputs off, and the pod idled until the limit. Give a pod the lifetime of the planned run (about 2 to 3 hours here), and copy each step's outputs to the VM as soon as it finishes (step 7 below).

## The training rule

A failure counts as infrastructure only when the policy could not have caused it. Infrastructure failures are dropped from the batch and counted; every other failure scores 0, timeouts included. This is the rule proposed for BenchFlow's RL cookbooks (not yet a formal BenchFlow decision); the decision itself comes from rl-core's `benchflow.integrations.rewards`.

| What happened | Who could cause it | What prime-rl sees |
|---|---|---|
| The sandbox never started, or the bridge process could not start | infrastructure | the episode fails (`BenchFlowInfraError`): dropped, counted in `errored_rollouts` |
| The model endpoint failed (vLLM down, 5xx) | infrastructure | Verifiers records a `ProviderError`: dropped, counted |
| The bridge process died or stopped answering mid-episode | infrastructure | the episode stops (`bridge_failed`) and fails: dropped, counted |
| The verifier crashed on a sandbox the policy never touched | infrastructure | dropped (`verifier_crash_clean_run`) |
| A command ran past 30 seconds | the policy | the model sees `{"error": "Command timed out after 30 seconds"}`; the episode goes on |
| The policy used its 10 turns, or its wall-clock budget (`agent_budget_sec`, 900 s), or its context | the policy | the episode stops and is verified as it stands: a reward, usually 0 |
| The verifier crashed or timed out after the policy acted, or the sandbox was lost after it acted | the policy may have | 0 |
| The verifier scored the work | the policy | the verifier's reward (v1 is 0 or 1; v2 gives partial credit) |

Verifiers' own stage timeouts are never used for the policy's limits, because prime-rl turns them into errors and drops them from training; the env's budgets are `@vf.stop` conditions instead, which score. Every episode appends one line to `<jobs_dir>/outcomes.jsonl` with its reward or drop and why.

## How it works

**Two Python environments.** BenchFlow and Verifiers v1 cannot be installed together: BenchFlow pins `litellm[proxy]==1.91.0`, which requires `mcp<2`, and Verifiers requires `mcp==2.0.0` (`uv pip compile` finds no solution). So `benchflow-taskset` is installed next to prime-rl and never imports BenchFlow. Each episode starts `bridge.py` with the Python of a separate BenchFlow venv (`$BENCHFLOW_PYTHON`); the bridge speaks JSON lines over stdin and stdout and owns the sandbox through `TaskRuntime`. It closes the sandbox on a `close` request, when its input ends (the env worker died), on SIGTERM, and after 30 idle minutes, and it asks Daytona to stop and delete an idle sandbox after 30 minutes, so one orphaned by a hard kill of the pod does not live for BenchFlow's default day. It starts with an allowlisted environment: the Daytona and BenchFlow variables, never model keys or the Verifiers venv's Python settings.

**The env owns the sandbox.** A Verifiers rollout tears down its tool servers before it scores, and toolsets run as separate processes, so a toolset cannot own a sandbox that must still be alive when the verifier runs. `BenchFlowEnv` holds it for the whole episode instead:

1. take one of `max_sandboxes` machine-wide slots (a `flock` on a slot file, so the cap holds across prime-rl's env-server workers and frees itself if a worker dies);
2. start the bridge and the sandbox;
3. serve `run_bash` and `submit` over MCP from its own process on 127.0.0.1, behind an unguessable path, and run the agent with them (`agents.agent.run(task, tools=...)`, Verifiers' public way to lend live tools);
4. `submit` sets a flag and a `@vf.stop` ends the rollout; the task's `@vf.reward` runs BenchFlow's verifier while the sandbox is still alive;
5. close the sandbox whatever happened, and log the outcome.

The seat is Verifiers' `null` harness (a chat loop whose only tools are the MCP ones) in the `subprocess` runtime: it only talks to the model, and every command runs in the BenchFlow sandbox. The env refuses harnesses that execute commands themselves (`bash`, `codex`, ...), since those would run on the trainer's machine, and it refuses agent-level retries, which would rerun the policy in a sandbox it already changed.

**The shared harness.** The env uses the RL cookbooks' shared harness (`docs/examples/rl/common/harness.py`): the harness message appended to each task prompt, 30-second commands, TRL's truncation at 2,000 characters, TRL's `{"error": ...}` tool errors, a required `submit(answer)` that writes `/workdir/answer.txt`, the tool definitions of `bash_tool_schemas()`, and 10 turns. So `evaluate.py` scores a policy on exactly what it trained on. `tests/test_prime_cookbook_harness.py` keeps the package's copy of these settings identical to the source.

**One GPU.** prime-rl runs its trainer and vLLM on separate GPUs, and no 2-GPU pod was available. Its launcher reads `CUDA_VISIBLE_DEVICES` literally, so `CUDA_VISIBLE_DEVICES=0,0` gives inference and the trainer the same physical GPU. vLLM takes 45% of the H200's memory (`gpu_memory_utilization = 0.45`) and the LoRA trainer the rest; with LoRA, weights reach vLLM through the filesystem. On a 2-GPU pod, drop the `0,0` and the memory setting.

## Files

| File | What |
|---|---|
| `benchflow_taskset/benchflow_taskset/taskset.py` | `BenchFlowTaskset` (loads task folders through the bridge), `BenchFlowTask` (its stops and reward), the configs |
| `benchflow_taskset/benchflow_taskset/env.py` | `BenchFlowEnv`: one sandbox per episode, from start to verdict |
| `benchflow_taskset/benchflow_taskset/session.py` | the bridge process, the episode session, the tool outputs |
| `benchflow_taskset/benchflow_taskset/tools.py` | the in-process MCP server for `run_bash` and `submit` |
| `benchflow_taskset/benchflow_taskset/slots.py` | the machine-wide sandbox cap |
| `benchflow_taskset/benchflow_taskset/bridge.py` | the BenchFlow side; runs in the BenchFlow venv |
| `benchflow_taskset/tests/` | 42 offline tests (run with the Verifiers venv) against a scripted bridge |
| `configs/rl.toml` | the prime-rl config |
| `pods/prime_pods.py`, `pods/run_watchdog.sh` | renting pods, the ledger, the watchdog |
| `pod/setup_pod.sh`, `pod/run_on_pod.sh` | installing a pod; starting a job on it with the Daytona key in its environment only |

## Run it

Everything runs from a VM that holds the keys (`~/.config/benchflow/primeintellect.env` with `PRIME_API_KEY`, `daytona.env` with `DAYTONA_API_KEY`, both mode 600). Pods never get the Prime key or GCP credentials, and get the Daytona key only as the environment of the job that needs it. These are the commands that were run, with the paths used.

**Which wallet pays.** Prime bills a pod to the key owner's personal wallet unless the create request names a team. To use a team's wallet, put `PRIME_TEAM_ID=<team id>` in the same env file (or the environment); `prime_pods.py` then sends it on every pod it creates, and `wallet` and the watchdog read that team's balance. `GET /user/teams` lists the teams a key can use.

**1. Watchdog and SSH key (once).** The watchdog passes every 5 minutes and acts only on pods it knows are ours (in the ledger under an owner in `policy.json`'s `owners`, or named with one of its `name_prefixes`). It terminates a pod past its lifetime (the one given at creation, never more than 8 hours), an owner's pods at the owner's cap, and all of ours at the spend cap ($1,400), and it logs the wallet balance and runway. `create` refuses to run while the watchdog is stopped.

```bash
export PRIME_POD_DIR=~/rl-prime            # ledger.jsonl, policy.json, watchdog.log
ssh-keygen -t ed25519 -N "" -f ~/.ssh/prime_bf    # the pods' key; never leaves the VM
python3 pods/prime_pods.py ssh-key-register --name benchflow-rl-prime-vm --pub ~/.ssh/prime_bf.pub
bash pods/run_watchdog.sh start                   # status | stop
python3 pods/prime_pods.py wallet                 # balance and runway: check before renting
```

**2. Rent a pod, with the run's lifetime.** One H200 with Prime's `prime_rl` image took 6.5 minutes to be reachable. Prefer the plain `ubuntu_22_cuda_12` image: `setup_pod.sh` installs everything it needs, and on 2026-10-01 a lambdalabs pod with `prime_rl` came up with Prime's own install failed ("Failed to install Docker packages") and no SSH address, so it had to be deleted. A massedcompute H100 with `ubuntu_22_cuda_12` was reachable in 3 minutes.

```bash
python3 pods/prime_pods.py offers --gpu-count 2   # prefer two GPUs when offered
python3 pods/prime_pods.py create --owner rl-prime --gpu-type H100_80GB --gpu-count 1 \
  --image ubuntu_22_cuda_12 --name rl-prime-h100-2 --max-hours 3
python3 pods/prime_pods.py wait <pod-id>          # prints user@host:port
```

**3. Install.** prime-rl pins CUDA 13 wheels (torch 2.13+cu130). A driver older than 580 needs NVIDIA's forward-compatibility libraries first (the H200 pod had branch 550; the H100 pod had 580 and needed nothing). Then ship this checkout and the tasks from the VM, without the tasks' `oracle/` folders, and run `setup_pod.sh` (2 to 4 minutes: prime-rl at commit d5f29c07 with `uv sync --all-extras`, a BenchFlow venv, `benchflow-taskset` into prime-rl's venv, and checks). The pod's user may be `ubuntu` rather than `root`; `configs/rl.toml` uses `/root/bf-rl`, so rewrite its paths to the user's home.

```bash
ssh -p <port> <user>@<host> 'nvidia-smi --query-gpu=driver_version --format=csv,noheader'   # below 580? then:
ssh -p <port> <user>@<host> 'sudo apt-get update -qq && sudo apt-get install -y -qq cuda-compat-13-0 &&
  echo /usr/local/cuda-13.0/compat | sudo tee /etc/ld.so.conf.d/000-cuda-13-compat.conf && sudo ldconfig'
git -C <benchflow-checkout> archive --format=tar HEAD | ssh -p <port> <user>@<host> 'mkdir -p ~/bf-rl/benchflow && tar -xf - -C ~/bf-rl/benchflow'
tar -C <tasks-dir>/v1 --exclude='*/oracle' -cf - train test | ssh -p <port> <user>@<host> 'mkdir -p ~/bf-rl/tasks-v1 && tar -xf - -C ~/bf-rl/tasks-v1 && ln -sfn ~/bf-rl/tasks-v1 ~/bf-rl/tasks'
ssh -p <port> <user>@<host> 'bash ~/bf-rl/benchflow/docs/examples/rl/prime/pod/setup_pod.sh &&
  sed "s#/root/bf-rl#$HOME/bf-rl#g" ~/bf-rl/benchflow/docs/examples/rl/prime/configs/rl.toml > ~/bf-rl/rl.toml'
```

**4. Serve the base model.** Background only the server command, as here: backgrounding a whole `cd ... && server` chain leaves a shell holding the SSH session open until the server exits. `VLLM_USE_FLASHINFER_SAMPLER=0` makes vLLM sample with PyTorch: FlashInfer's sampler compiles a kernel on the first top-p request, and an image without `nvcc` (the plain `ubuntu_22_cuda_12` one) kills the engine there.

```bash
ssh -p <port> <user>@<host> 'mkdir -p ~/bf-rl/logs; cd ~/bf-rl/prime-rl; export PATH=$HOME/.local/bin:$PATH;
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup uv run --no-sync inference \
  --vllm.model Qwen/Qwen3-4B-Instruct-2507 --vllm.tool-call-parser hermes --vllm.max-model-len 32768 \
  --vllm.gpu-memory-utilization 0.85 --server.port 8000 > ~/bf-rl/logs/vllm-base.log 2>&1 < /dev/null &'
```

**5. Calibrate, then evaluate the base model on test.** `run_on_pod.sh` streams the Daytona key over ssh's stdin into the job's environment. Give every run its own owner label: evaluate.py deletes every sandbox of its owner when it ends. Pick calibration tasks with `--include` (8 per kind); `--limit` takes the first tasks alphabetically, which are all bugfix. About 4 minutes for 32 episodes and 5 for 200, at 32 sandboxes.

```bash
BENCHFLOW_DAYTONA_OWNER=rl-prime-calib-v1 bash pod/run_on_pod.sh root@<host>:<port> calib-v1 \
  env VLLM_KEY=EMPTY /root/bf-rl/benchflow/.venv/bin/python /root/bf-rl/benchflow/docs/examples/rl/common/evaluate.py \
  --tasks-dir /root/bf-rl/tasks-v1/train --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 \
  --api-key-env VLLM_KEY --sandbox daytona --concurrency 32 --owner rl-prime-calib-v1 \
  --out /root/bf-rl/evals/calib-v1 --include <task> ...   # 8 --include per kind, from the split's manifest.jsonl
BENCHFLOW_DAYTONA_OWNER=rl-prime-eval-base bash pod/run_on_pod.sh root@<host>:<port> eval-base \
  env VLLM_KEY=EMPTY /root/bf-rl/benchflow/.venv/bin/python /root/bf-rl/benchflow/docs/examples/rl/common/evaluate.py \
  --tasks-dir /root/bf-rl/tasks-v1/test --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 \
  --api-key-env VLLM_KEY --sandbox daytona --concurrency 32 --samples 4 --owner rl-prime-eval-base \
  --out /root/bf-rl/evals/base-v1-test
```

**6. Train.** Stop the base server first, then launch. Stop it by process group: `setsid` made the server its group's leader, and killing only the processes whose command line matches can leave vLLM's engine holding the GPU's memory. `configs/rl.toml` runs 20 steps of 8 tasks x 8 rollouts, at most 32 episodes (and sandboxes) at once, checkpoints every 5 steps; step 1 took 2 minutes.

```bash
ssh -p <port> <user>@<host> 'for p in $(pgrep -f "[u]v run --no-sync inference"); do kill -- -$(ps -o pgid= -p $p | tr -d " "); done'
BENCHFLOW_DAYTONA_OWNER=rl-prime-train bash pod/run_on_pod.sh <user>@<host>:<port> train bash -c \
  "cd \$HOME/bf-rl/prime-rl && export PATH=\$HOME/.local/bin:\$PATH CUDA_VISIBLE_DEVICES=0,0 VLLM_USE_FLASHINFER_SAMPLER=0 && exec uv run --no-sync rl \
   @ \$HOME/bf-rl/rl.toml --output-dir \$HOME/bf-rl/outputs --run.name rl-v1-qwen3-4b"
ssh -p <port> root@<host> 'grep -E "Step [0-9]+" /root/bf-rl/outputs/rl-v1-qwen3-4b/logs/latest/orchestrator.log'
```

**7. Copy each result off as soon as it exists.** Pull from the pod through the VM; a pod never gets GCP credentials. Stream to your bucket from a machine that can write to it.

```bash
ssh -p <port> root@<host> 'tar -C /root/bf-rl -czf - evals outputs/rl-v1-qwen3-4b/logs jobs/train/outcomes.jsonl' > results-$(date +%H%M).tar.gz   # on the VM
```

**8. Evaluate the trained adapter (not yet run).** With LoRA, the trainer writes a PEFT adapter directory (`adapter_config.json` and weights) for each weight broadcast at `<run output>/broadcasts/step_<n>`, removing older ones, so the last step's stays. Stop the training processes, serve the base model with LoRA enabled, load the adapter through prime-rl's admin route on the engine's port (the server port plus 100; the router on the server port serves no admin routes), and run step 5's test command with `--model trained` and its own owner label.

```bash
uv run --no-sync inference --vllm.model Qwen/Qwen3-4B-Instruct-2507 --vllm.tool-call-parser hermes \
  --vllm.enable-lora --vllm.max-lora-rank 16 --vllm.max-model-len 32768 --server.port 8000
curl -s localhost:8100/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "trained", "lora_path": "/root/bf-rl/outputs/rl-v1-qwen3-4b/broadcasts/step_20"}'
```

**9. Clean up.** Terminate the pod and confirm it is gone, check that no sandbox is left under any of the run's owner labels, and stop the watchdog last.

```bash
python3 pods/prime_pods.py delete <pod-id>        # waits until /pods/ no longer lists it
python3 pods/prime_pods.py list                   # nothing of ours
BENCHFLOW_DAYTONA_OWNER=rl-prime-train bench sandbox cleanup --all   # and each eval/calibration owner
bash pods/run_watchdog.sh stop
```

## Time and cost

Measured on 1x H200 at $5.48/h: pod reachable in 6.5 minutes, install 4 minutes, a 32-episode calibration about 4 minutes, the 200-episode base evaluation about 5 minutes, one training step 2 minutes. A complete run (install, base and trained evaluations, 20 training steps, copying off) should take about 1.5 to 2 hours, roughly $8 to $11 of pod time; the steps after step 1 were not measured. This run cost $43.52 because the pod idled until its 8-hour limit.

Prime allows an account more than two instances only after its top-ups reach $100, so no second run was attempted.

## Tests

```bash
cd docs/examples/rl/prime/benchflow_taskset && <verifiers-venv>/bin/python -m pytest -q tests   # 42 tests, no network
uv run pytest tests/test_prime_cookbook_bridge.py tests/test_prime_cookbook_harness.py tests/test_prime_cookbook_pods.py   # 37, in BenchFlow's venv
```

The package tests run the real bridge client, session, tool server, env, and task hooks against a scripted bridge: reward mapping and drops through Verifiers' own scoring boundary, the TRL harness's tool outputs, the bridge's environment allowlist, and the sandbox closing when an episode ends normally, crashes, is cancelled, never starts, or its bridge dies or stops answering. They found one bug before training (MCP could not resolve the tools' annotations, so every episode would have failed to serve its tools). In BenchFlow's suite, the bridge tests use BenchFlow's real task loader and reward helper with a scripted `TaskRuntime`, and run the bridge as a process to check that a close request, end of input, SIGTERM during a command, and the idle timeout each close the sandbox; the harness tests keep the package's copy of the shared harness identical to the source; the watchdog tests cover its stop rules, including that it never touches or counts a pod we did not create.

## Publishing to the Environments Hub

Not done: publishing it. The package is ready to push as a private v1 environment (Prime's CLI infers v1 from its `verifiers>=0.3.2.dev158` requirement):

```bash
prime login
prime env push --path docs/examples/rl/prime/benchflow_taskset --owner benchflow --visibility PRIVATE
prime env install benchflow/benchflow-taskset      # elsewhere; its id stays benchflow-taskset
```

Two caveats. The id cannot be `benchflow`: Verifiers imports a plugin by its name, and `benchflow` is the SDK. And a Hub install is not enough on its own: the machine also needs a BenchFlow venv for the bridge (`$BENCHFLOW_PYTHON`) and the task folders, so Prime's hosted training cannot run it until BenchFlow can be installed next to Verifiers (see Gaps). This package would replace BenchFlow's three SkillsBench environments on the Hub, which still use the v0 runtime that Verifiers removed on 2026-08-31.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `uv pip compile` of BenchFlow with Verifiers: no solution | the `mcp` pin conflict; keep the two venvs |
| `ModuleNotFoundError: benchflow` from the bridge | `$BENCHFLOW_PYTHON` was the resolved symlink of the venv's `python`; use the venv's own path (`.venv/bin/python`, not `.resolve()`d) |
| taskset id `benchflow` fails to load | use `benchflow-taskset` |
| `torch.cuda.is_available()` false on the pod | CUDA 13 wheels on an older driver: install `cuda-compat-13-0` and put its folder first in the loader path (step 3) |
| vLLM dies on the first request: `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`, then every episode drops as `model_endpoint` | FlashInfer's sampler needs `nvcc` to compile its kernel; set `VLLM_USE_FLASHINFER_SAMPLER=0` for every server and for training (steps 4, 6, 8) |
| `setup_pod.sh` stops after "installing uv" with `mkdir: cannot create directory '~/.config/fish'` | the image's `~/.config` belongs to root; the script now installs uv with `UV_NO_MODIFY_PATH=1`, or `sudo chown -R $USER ~/.config` |
| an `ssh ... '... &'` that starts a server never returns | the whole `cd ... && server &` chain ran in a background subshell that still holds the session; background only the server command (step 4) |
| orchestrator: `ZMQError: Address already in use (tcp://localhost:5555)` | set `[rollout_transport] port` (the config uses 16555) |
| `UserWarning: CUDA initialization ... invalid device ordinal` in the launcher or orchestrator | harmless with `CUDA_VISIBLE_DEVICES=0,0`: they use no GPU; inference and the trainer each get `0` |
| episodes end with a plain-text answer and reward 0 | the model answered in text instead of calling `submit`; the shared harness message asks for `submit`, and training teaches it (32 of the first 44 base episodes ended this way before the env used the shared harness) |
| `evaluate.py --limit` picks only bugfix tasks | it takes the first tasks alphabetically; use `--include` |
| LOW-WALLET in `watchdog.log` | the shared balance is under one hour of our burn: copy outputs off and stop at a checkpoint |
| sandboxes left after a crash | `BENCHFLOW_DAYTONA_OWNER=<owner> bench sandbox cleanup --all` for each owner label |

## Gaps found

- **BenchFlow cannot be installed next to Verifiers v1.** `litellm[proxy]==1.91.0` needs `mcp<2`; Verifiers needs `mcp==2.0.0`. The bridge works around it, at the cost of a second venv and a process per episode; making the proxy extra optional in BenchFlow would let `benchflow-taskset` import BenchFlow directly and make a Hub install self-contained.
- **`TaskRuntime.verify()` also tears the sandbox down.** The reward waits for the sandbox deletion (about 25 to 30 seconds of the per-episode overhead measured on Daytona). A verify that returns before cleanup would shorten every episode.
- **A timed-out command raises on Daytona.** BenchFlow wraps the command in `timeout N` but polls with the same deadline, so a slow command raises `Command timed out` instead of returning exit 124 with its output. The bridge waits 15 seconds longer so the model sees a normal result, but reports it in the TRL harness's error form to match evaluate.py.
- **Verifiers tears tool servers down before scoring**, so a resource the verifier needs cannot live in a toolset; an env has to own it and lend tools with `agent.run(..., tools=...)`.
- **prime-rl drops Verifiers' stage timeouts from training** (it counts them as errors), so any budget that should score has to be a `@vf.stop`.
- **Single-GPU RL is not a documented prime-rl setup.** `CUDA_VISIBLE_DEVICES=0,0` works because the launcher reads the variable literally.
