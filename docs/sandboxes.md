# Sandboxes

BenchFlow needs an isolated place to install an agent, expose the task
workspace, and run the verifier. For local development, that place is Docker.
Cloud sandboxes are optional scaling backends.

## Which sandbox should I use?

| Sandbox | Use it when | Setup |
|---|---|---|
| **Docker** | You are learning BenchFlow, developing a task, debugging a run, or running a small batch locally | Docker daemon; included in the base install |
| **Apple Container** | You are on a supported Apple Silicon Mac and want Apple's native container runtime | Apple `container` CLI; no BenchFlow extra |
| **Daytona** | You need many independent cloud VMs for a light, highly parallel batch | Optional extra plus `DAYTONA_API_KEY` |
| **Modal** | You need serverless or GPU-backed remote execution | Optional extra plus Modal auth |
| **AgentCore** | Your deployment is built around AWS Bedrock AgentCore Runtime | Optional extra plus AWS configuration |

Docker is the CLI default. If your tasks and model calls fit on one machine,
there is no reason to configure Daytona.

## Local Docker

```bash
docker info >/dev/null

bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex \
  --model gpt-5.5 \
  --sandbox docker
```

Start with `--concurrency 1` or `2`, then raise it while watching local CPU,
memory, disk, Docker build pressure, and provider rate limits. Docker uses host
disk capacity and supports multi-container task environments.

## Apple Container

On supported Apple Silicon Macs:

```bash
bench eval run \
  --tasks-dir tasks/my-task \
  --agent codex \
  --model gpt-5.5 \
  --sandbox apple-container
```

Apple Container is a single-container backend and currently cannot enforce a
task's `no-network` policy. Use Docker for multi-service or strict no-network
tasks.

## Daytona

Install Daytona support only when you need it:

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'
export DAYTONA_API_KEY='...'

bench eval run \
  --tasks-dir tasks \
  --agent gemini \
  --model gemini-3.1-pro-preview \
  --sandbox daytona \
  --concurrency 32
```

Daytona is useful for parallel experiments because each rollout gets a remote
VM. It is not a prerequisite for a single evaluation. Daytona also caps each
sandbox at 10 GB of storage, so tasks with large model snapshots, Playwright,
LaTeX, or other heavy images may fail during bootstrap. Run those tasks with
Docker when local host disk is available.

## Modal and AgentCore

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-modal]'
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-agentcore]'
```

Select them with `--sandbox modal` or `--sandbox agentcore` after configuring
the provider's authentication. Both are deployment choices for specific
remote workloads, not part of the local quickstart. They are single-container
backends; AgentCore also cannot enforce `no-network` tasks.

## Keep the task portable

The sandbox flag selects where a task runs; it should not change what the task
means. Develop and debug with Docker first, then run a small parity check on
the intended cloud backend before starting a batch. If a task requires a
backend-specific capability, document that requirement in the task rather
than silently assuming Daytona.
