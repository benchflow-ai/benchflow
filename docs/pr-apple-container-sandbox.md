# feat(sandbox): add Apple Container backend for macOS Virtualization.framework

## Summary

- Adds `--sandbox apple-container` as a fourth sandbox backend, using macOS's native `container` CLI (Virtualization.framework micro-VMs)
- Enables BenchFlow evaluations on Apple Silicon Macs without Docker Desktop — each task runs in an isolated arm64 Linux VM with virtiofs bind mounts
- Includes kalloc.1024 kernel zone headroom monitoring to prevent the cryptic exit-128 crash that occurs when the zone is exhausted

## Motivation

A growing number of agent developers and researchers run their workflows on macOS — both as a daily driver and as dedicated evaluation hardware (Apple Silicon's unified memory and always-on availability make it attractive for long-running agent batches). These users increasingly want to run their agents of choice (Claude Code, Qodercli, OpenCode, custom ACP clients) against BenchFlow tasks *locally*, without routing through cloud infrastructure or installing Docker Desktop (heavy, license-restricted for commercial use).

At the same time, the broader agentic infrastructure space is moving toward lightweight micro-VMs as the isolation primitive — faster startup, stronger isolation than containers, and native integration with the host OS. Apple Container (Virtualization.framework) ships with macOS 26+ on Apple Silicon and delivers exactly this: arm64 Linux micro-VMs with ~2s startup for cached images, virtiofs bind mounts, and zero additional dependencies.

This backend makes `bench eval run --sandbox apple-container` work natively on Mac, enabling local evaluation with any agent that speaks BenchFlow's ACP protocol. We validated the approach across 30+ production evaluation runs (spanning 7 benchmark tasks, multiple agent configurations, and a peer-reviewed research effort on skill salience) using a standalone harness before extracting the generic container lifecycle into this plugin.

## Design

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Exec model | Background `container run ... "sleep infinity"` + `container exec` per command | `container run` blocks (no `-d` flag); mirrors Docker compose pattern |
| Mounts | Single `rollout_dir` → `/logs` virtiofs bind mount | Subdirectories are regular dirs inside the mount — `chmod` works (virtiofs rejects chmod on the mount point itself) |
| File transfer | Host-side copy for mounted paths; `base64` via `exec -i` for unmounted | virtiofs is synchronous and crash-safe; `-i` flag required for stdin |
| Networking | `off_box_model=False` — VM reaches host via gateway `192.168.64.1` | Confirmed empirically; `host.docker.internal` does NOT resolve |
| kalloc gate | Preflight + per-start check; aborts with "reboot required" | ~100k elements leak per start/stop cycle; crash at ~3M |
| Architecture | arm64 only; rejects Dockerfiles with amd64/x86_64 references | No `--platform` flag in `container build` |
| Concurrency | Recommended `--concurrency 1` | kalloc leak limits to ~10-12 containers between reboots |

## Changes

| File | Description |
|------|-------------|
| `src/benchflow/sandbox/apple_container.py` | New backend (~500 lines): lifecycle, exec, file transfer, kalloc monitoring |
| `src/benchflow/sandbox/providers.py` | Registry entry: `SandboxProvider("apple-container", ...)` |
| `src/benchflow/sandbox/setup.py` | Factory dispatch: `elif sandbox_type == "apple-container"` |
| `pyproject.toml` | Empty extra `sandbox-apple-container = []` (uses system CLI, no SDK) |
| `tests/test_apple_container_sandbox.py` | 31 unit tests (mocked subprocess, runs on any platform) + gated integration test |
| `tests/test_sandbox_provider_registry_drift.py` | Updated assertions for 4-provider registry |

## Testing

**Unit tests (31, any platform):**
- kalloc zprint parsing (healthy, exhausted, failure)
- Dockerfile arm64 validation (clean, `--platform=linux/amd64`, `x86_64`)
- Preflight gates (non-darwin, missing CLI, kalloc exhausted)
- Exec argv construction (basic, cwd, user, env redaction, service rejection, timeout→cleanup)
- File transfer routing (mounted→host copy, unmounted→base64 via `exec -i`)
- Stop lifecycle (with/without delete)
- Properties (`is_mounted=True`, `supports_snapshot=False`)

**Integration test (macOS-gated, skipped in CI):**
- Full lifecycle: build → start → exec → upload → stop against a real `ubuntu:24.04` VM

**E2E validation:**
```
$ bench eval run --tasks-dir tasks/sales-pivot-analysis --agent oracle --sandbox apple-container
✓ Score: 1/1 (100.0%), errors=0   # reward=1.0, 1.1min total
```

## Constraints & Operational Notes

- **macOS + Apple Silicon only.** Preflight raises a clear error on other platforms.
- **`container system start`** must be run once before first use (XPC service).
- **kalloc.1024 leak:** Each container start/stop leaks ~100k kernel zone elements. After ~10-12 evaluations, reboot to reclaim. The backend detects exhaustion and fails fast with actionable guidance.
- **No snapshot support.** `supports_snapshot=False` — Branch substrate requires Docker or Daytona.
- **Single-container only.** Multi-service (vulhub-style) tasks require the Docker backend.
- **Public-network tasks only for now.** Tasks declaring `network_mode = "no-network"` fail closed because the backend does not yet enforce VM network isolation.

## Test plan

- [x] 28 unit tests pass on any platform (no macOS required)
- [x] 6 provider registry drift tests pass
- [x] Integration lifecycle test passes on macOS with container CLI
- [x] E2E oracle evaluation: `sales-pivot-analysis` → reward=1.0
- [x] `ruff check` clean on all changed files
- [ ] CI green (unit tests run cross-platform; integration test auto-skips)
