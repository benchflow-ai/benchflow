# Modal

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-modal]'
modal token new                     # or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET
bench run tasks/ -s modal -j 64     # without -a, the oracle runs
```

Modal runs each trial in a serverless sandbox: ACP and native agents (Claude Code, Codex, ...), scripted agents, session-factory agents and the oracle. The model proxy runs inside each sandbox.

## GPUs

Set the GPU request on the task, then select Modal:

```yaml
sandbox:
  gpus: 1
  gpu_types: [T4]
```

For a legacy `task.toml`, put the same fields under `[environment]`:

```toml
[environment]
gpus = 2
gpu_types = ["A100-80GB"]
```

```bash
bench check tasks/my-gpu-task --sandbox modal
bench run tasks/my-gpu-task --sandbox modal -j 1
```

`gpus` is a non-negative integer. Zero means CPU only; omit `gpu_types` in that case. A positive count with no type requests Modal's `any` GPU. For reproducible benchmarks, specify the type and retain the actual device information from the sandbox.

Modal Sandboxes accept one GPU type. BenchFlow refuses multiple entries instead of silently selecting one; the fallback lists documented for Modal Functions do not apply here. Keep the count in `gpus`, not in the type string: two GPUs use `gpus: 2` and `gpu_types: [H100]`. Modal validates type availability, count limits and account capacity when allocating the sandbox. `bench check` validates the task request; it does not check live capacity.

Use an image with the CUDA libraries and task dependencies your workload needs. Build the image and stage weights/data before the GPU run; task setup commands run after allocation. Keep `-j` within the GPU capacity and budget for the requested type/count. GPU tasks use the same agent paths as CPU tasks, ACP and native agents included.

This adapter requires Modal SDK `>=1.6.2,<1.7` (the lock and `constraints.txt` pin 1.6.2). It moves files and makes the log directories with the SDK's `Sandbox.filesystem` API: Modal's servers refuse the legacy `Sandbox.open`, `mkdir` and `ls` calls, which SDK 1.6 removed. GPU Sandboxes can be preempted; a provider interruption is an infrastructure outcome, not evidence of model failure. See Modal's [GPU configuration](https://modal.com/docs/guide/gpu), [Sandbox resources](https://modal.com/docs/guide/sandbox-resources) and [SDK migration notes](https://modal.com/docs/guide/sandbox-v2).

## ACP and native agents

The agent talks to BenchFlow over its own stdin and stdout, carried by Modal's `Sandbox.exec` streams. It starts the way it does on Docker: root's bash by absolute path, reading no startup file, with the loader variables (`LD_PRELOAD`, `BASH_ENV` and the rest) empty until it has dropped to the sandbox user. The agent's environment, its credentials included, is the first line of its standard input, so it is in no command line, file or Modal Secret. Lines over the ACP size limit are shortened inside the sandbox, as on the other backends.

Modal cannot enforce `denylist` or `allowlist`, so a task that keeps sites out of the agent's reach is refused here; run it on Docker or Daytona.

## Limits

| Feature | On Modal |
|---|---|
| ACP and native agents (Claude Code, Codex, Gemini CLI, …) | Supported, over `Sandbox.exec` stdio |
| Multi-container tasks (Compose) | Not supported |
| `no-network` | Supported |
| `denylist`, `allowlist` | Refused before the run |
| Separate verifier sandbox, `--integrity strict` | Supported. The verifier's sandbox starts once the agent's has stopped. A file over 5 GiB (Modal's per-file transfer limit) cannot cross: the capture fails with that reason and the trial is unscored |
| Cloud metadata endpoints (open network) | Not reachable from a Modal sandbox. Each agent trial proves it with a connect to every endpoint and records `not_reachable`; no `--allow-cloud-metadata` is needed |
| Cached install layers | None; the proxy and agent install in each trial |
| Task GPUs | Supported with one requested type and a positive count; other BenchFlow backends refuse GPU tasks |

BenchFlow refuses an unsupported task before any sandbox starts. Check ahead with `bench check tasks/my-task --sandbox modal`.

## Settings

| Variable | Purpose |
|---|---|
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal credentials, if `~/.modal.toml` does not exist |
| `BENCHFLOW_MODAL_ADD_PYTHON` | Python version Modal adds to images without one (default `3.12`; empty adds none) |

## Next steps

- Run agents on tasks that need a `denylist` or `allowlist`: [Daytona](./daytona.md).
- Compare sandboxes: [Sandboxes](./index.md).
- What the sandbox guarantees: [Sandbox security](./security.md).
