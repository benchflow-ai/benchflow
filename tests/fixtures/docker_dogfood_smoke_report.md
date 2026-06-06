# Docker dogfood smoke report

Branch: `cursor/task-standard-runtime-gaps-e453`  
Date: 2026-06-06  
Environment: Cursor cloud VM (nested container host)

## Docker availability

| Check | Result |
|-------|--------|
| Initial `docker info` | **Not installed** (`command not found`) |
| After `apt install docker.io` | **Available** (daemon started manually) |
| Storage driver | **vfs** required (default overlayfs failed with `invalid argument` in nested host) |
| `DockerSandbox.preflight()` | **ok** |

## `bench tasks check --sandbox docker`

All three dogfood packages passed structural + runtime validation:

| Task | `check_task` | runtime on docker |
|------|--------------|-------------------|
| `verifier-package-reward-contract` | ✓ valid | ✓ supported |
| `prompt-user-semantics` | ✓ valid | ✓ supported |
| `compat-export-loss-reports` | ✓ valid | ✓ supported |

## Dockerfile build + `WORKDIR /repo`

Built from each task's `environment/Dockerfile` (`FROM ghcr.io/astral-sh/uv:python3.12-bookworm`, `WORKDIR /repo`):

| Task | Image tag | Build | `pwd` in container |
|------|-----------|-------|--------------------|
| verifier-package-reward-contract | `bf-dogfood-verifier-package-reward-contract` | ✓ | `/repo` |
| prompt-user-semantics | `bf-dogfood-prompt-user-semantics` | ✓ | `/repo` |
| compat-export-loss-reports | `bf-dogfood-compat-export-loss-reports` | ✓ | `/repo` |

## `verifier/test.sh` (repo mounted at `/repo`)

Ran inside built images with `-w /repo -v /workspace:/repo -v /tmp/...:/logs/verifier`.
Network enabled only for `uv sync --extra dev --locked` (task declares `no-network`; pytest itself is offline).

| Task | pytest result | reward artifacts |
|------|---------------|------------------|
| verifier-package-reward-contract | **99 passed** | `reward.txt`, `reward.json`, `reward-details.json` |
| prompt-user-semantics | **39 passed** | `reward.txt`, `reward.json` |
| compat-export-loss-reports | **82 passed** | `reward.txt`, `reward.json` |

## Compose workdir overlay

Manual compose with `working_dir: /repo` (no CPU/memory deploy limits):

- `docker compose exec -w /repo main pwd` → `/repo`
- Benchflow source visible at `/repo/src/benchflow/__init__.py`

Matches `docker-compose-workdir.json` emitted by `DockerSandbox._write_workdir_compose_file()`.

## `DockerSandbox` compose startup (benchflow path)

`DockerSandbox.start(force_build=False)` failed in this VM when benchflow's base compose applied `deploy.resources.limits` (cpus/memory from task `environment`):

```
cannot enter cgroupv2 "/sys/fs/cgroup/docker" with domain controllers -- it is in threaded mode
```

Reproduced with plain `docker run --cpus=4 --memory=8192m` (without limits, `docker run` and compose both work).

**Assessment:** environment/cgroup limitation on this nested host, not a `environment.workdir` implementation bug. Workdir compose overlay and exec `-w` default behave correctly where containers can start.

## Unit tests (workdir-related)

```
uv run python -m pytest tests/test_runtime_capabilities.py -k workdir \
  tests/test_sandbox_multi_service.py::TestDockerSandboxServiceExec::test_exec_defaults_cwd_to_task_workdir \
  tests/test_task_standard_dogfood.py
```

6 passed, 2 skipped, 1 failed (`test_create_sandbox_environment_workdir_supported_on_modal` — missing optional `tenacity`/modal extra; unrelated to docker workdir).

## Fixes applied

**None.** No workdir bugs found; no code changes committed.
