# Separate verifier sandboxes

A task can ask for its verifier to run in a sandbox of its own instead of the sandbox the agent worked in. The verifier sandbox receives only the agent's final workspace, the task's declared artifacts and `/logs/artifacts`. Nothing else the agent left behind exists there: not a `conftest.py` in `/tests`, not a pre-written `/logs/verifier/reward.txt`, not a `.pth` file in site-packages, not a background process. This is Harbor's `environment_mode = "separate"`, and Harbor task directories that use it run unchanged on Docker and Daytona.

```toml
[verifier]
environment_mode = "separate"      # BenchFlow name: sandbox_mode

# Optional: the verifier sandbox's own settings (Harbor [verifier.environment],
# BenchFlow [verifier.sandbox]). Declaring it implies separate mode.
[verifier.environment]
docker_image = "ghcr.io/org/task-verifier:1"
cpus = 1
memory_mb = 2048
```

In a `task.md` package the same keys are `verifier.sandbox_mode: separate` and `verifier.sandbox:`.

## Which image the verifier sandbox uses

In Harbor's order:

1. `docker_image` in `[verifier.environment]` / `[verifier.sandbox]`;
2. with no verifier table, a fresh copy of the task's `[environment]` / `[sandbox]`: its `docker_image` when set;
3. otherwise the image built from `tests/Dockerfile` (Harbor builds the separate verifier from `tests/`; the Dockerfile usually copies `test.sh` into `/tests`);
4. BenchFlow only: with neither a verifier table nor a `tests/Dockerfile`, a fresh sandbox from the task's own `environment/Dockerfile`. None of the agent's changes survive into it.

A task with none of these is refused before launch (`bench tasks check <dir> --sandbox daytona` shows it). The verifier sandbox never runs the task's setup commands, healthcheck, skills or MCP servers; those belong to the agent's sandbox.

## What happens in one trial

1. The agent runs in its sandbox as usual. Separate mode checks before the agent starts that the image has `python3` (3.10+) or `tar`, which capture needs. It installs nothing into the agent's image. It also measures, before the agent starts, the paths capture will read: the workspace, `/logs/artifacts` and each declared artifact (entries, bytes, and whether the path is a symlink). That measurement is the clean control for the failures below.
2. After the agent stops, its workspace is frozen into `evidence/` together with the declared `artifacts = [...]`, and `/logs/artifacts` is collected into `artifacts/`. This is the same capture `--freeze-workspace` and `bench eval regrade` use. Images without Python are captured with `tar` alone.
3. The host packs exactly those files into one archive, after checking every byte against the manifest written at capture. Nothing else is read from the agent's sandbox.
4. A fresh verifier sandbox starts. The archive is unpacked at the original absolute paths (a dedicated workspace such as `/app` is replaced; a shared directory such as `/root` is overlaid). The normal verifier path then runs there: hardening, upload of `tests/`, `test.sh`, reward parsing. Its outputs land in the trial's `verifier/` as for any trial.
5. The verifier sandbox is deleted. The agent's sandbox is cleaned up as usual.

`soft_verify()` (intermediate feedback for user loops) is unavailable for these tasks, because it would upload `tests/` into the agent's sandbox.

## Failures are not zeros, unless the solution caused them

If the verifier cannot be given the agent's outputs, the trial is unscored: `rewards` is `null` and `verifier_error` starts with `separate verifier` (category `verifier_infra`). This covers a missing or mismatched capture, a failed artifact collection, a verifier image that does not build, and an upload that does not unpack. A declared artifact that the agent simply never wrote is not an error; the verifier sees that it is missing and scores the trial.

Some of these causes are the solution's own files: a workspace over the capture limits, `/logs/artifacts` over the collection limits (the agent owns that directory), a declared artifact left as a symlink, or a file in `/logs/artifacts` that takes a declared artifact's place. When the same paths were within bounds before the agent ran, the trial scores 0: `rewards` is `{"reward": 0.0}`, there is no `verifier_error` (so a retry cannot resample the 0 away), the verifier sandbox never starts, and `verifier-sandbox.json` has `status` `refused` with the reason in `refusal`. A policy therefore cannot turn a low score into an unscored trial by making its outputs impossible to transfer. When the clean control shows the same problem (a task image whose workspace is already over the limits, say), or it could not be measured, the trial stays unscored as above.

A symlink that points out of the workspace is not a failure at all: capture leaves it out and lists it in the manifest's `exclusions` (`symlink_escape`, with its `link_target`), so a virtualenv's `bin/python` or an agent CLI's helper link costs nothing.

## Cost and timing

Both sandboxes are billed while the verifier runs. Each trial records:

- in `timing.json`: `verifier_sandbox_setup` (create and build), `verifier_transfer` (upload and unpack), `verifier` (hardening and `test.sh`), `verifier_sandbox_teardown` and `verifier_sandbox_total`;
- in `verifier-sandbox/verifier-sandbox.json`: `image_source`, `sandbox_id`, `status` (`complete`, `verifier_failed`, `transfer_failed`, `sandbox_failed`, `refused`), the transfer inventory (workspace, files, bytes, declared artifacts, missing artifacts, `/logs/artifacts` files, exclusions), `pristine` (the measurement before the agent), `sandbox_seconds`, and the error or, for `refused`, the `refusal`;
- `verifier-sandbox/sandbox.json`: the provider id, written as soon as the sandbox exists.

A job's `--max-sandbox-seconds` budget adds `verifier_sandbox_total` to each trial's wall-clock time.

## Worked example

Expected outcomes when each row is run as an oracle trial. Harbor's two tasks were copied verbatim from `harbor-framework/harbor` `examples/tasks/verifier-mode-matrix` (commit 6cb9ff31). Both use `FROM ubuntu:24.04` with no network, so capture runs through the `tar` path.

| Task | Mode | Reward | Notes |
|---|---|---|---|
| answer-42, oracle writes 42 | separate | 1.0 | image from `tests/Dockerfile`; 1 workspace file transferred |
| answer-42, solution writes 41 and tampers (as root) | separate | 0.0 | planted `/tests/conftest.py`, site-packages `.pth`, PATH shim, `/tmp` file: all absent in the verifier sandbox |
| same tampering task | shared | 1.0 | the root-level plants reach the verifier; pytest reports a pass for the wrong answer |
| Harbor `separate-explicit` | separate | 1.0 | Harbor's own checks pass: `/logs/artifacts` marker copied, declared `/tmp` artifact present, undeclared `/tmp` file absent |
| Harbor `separate-reuse-env` | separate | 1.0 | also checks that the verifier image was built from `tests/`, not the agent's `environment/` |
| answer-42 with an unbuildable `tests/Dockerfile` | separate | unscored | `verifier_error: separate verifier sandbox failed to start: ... BUILD_FAILED`, category `verifier_infra` |

The tampering solution runs as root because the oracle does. It stands in for an agent that gained root, not for a locked-down agent user.

## Limits

- Docker and Daytona only; Modal, Apple Container and AgentCore refuse separate mode before launch.
- `test-script` verifiers only; `llm-judge` verifiers and non-`main` verifier services are refused.
- Multi-step tasks (`steps`) are still refused as a whole, so per-step verifier modes do not run yet.
- State outside the workspace, the declared artifacts and `/logs/artifacts` does not reach the verifier: services, databases, files elsewhere. A task whose verifier reads such state should stay in shared mode or declare the paths as artifacts.
- Capture limits are those of `--freeze-workspace`: at most 20 GiB and 200,000 entries per workspace, 1 GiB and 10,000 files for collected artifacts. Credential files, sandbox runtime state and symlinks that leave the workspace are left out and listed in the manifest.
- The verifier's network is the verifier sandbox's `network_mode`. Harbor's per-phase verifier network overrides are not applied separately.
