# Daytona

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'
export DAYTONA_API_KEY="..."
bench run tasks/ -a claude-agent-acp -s daytona -j 32
```

Daytona runs each trial in a cloud sandbox, so you can run many trials at once without local resources. The model proxy runs inside each sandbox. Tasks with a `docker-compose.yaml` run Docker inside the Daytona sandbox.

## Settings

| Variable | Purpose | Default |
|---|---|---|
| `DAYTONA_API_KEY` | Your Daytona API key (required) | none |
| `DAYTONA_API_URL`, `DAYTONA_TARGET` | Which Daytona API and target to use | Daytona's defaults |
| `BENCHFLOW_DAYTONA_MAX_CPUS` | Most CPUs a sandbox may request; larger requests are clamped | `4` |
| `BENCHFLOW_DAYTONA_MAX_MEMORY_MB` | Most memory a sandbox may request | `16384` |
| `BENCHFLOW_DAYTONA_MAX_STORAGE_MB` | Most disk a sandbox may request | `10240` |
| `BENCHFLOW_DAYTONA_OWNER` | A label for your sandboxes, used by listing, cleanup and snapshots. 1–48 letters, digits, `_`, `.` or `-` | none |
| `BENCHFLOW_DAYTONA_AUTO_REAP` | Delete abandoned BenchFlow sandboxes (stopped by Daytona, run lease lapsed) when a run starts; `0`, `false`, `no` or `off` turns it off | on |
| `BENCHFLOW_DAYTONA_SNAPSHOTS=content` | Reuse a built image across trials and processes (needs `BENCHFLOW_DAYTONA_OWNER`) | off |

Keep the Daytona key in your shell. Do not pass it with `--env`, which sends it into the sandbox.

## Disk limit

Daytona caps each sandbox at 10 GB. Tasks with heavy images (large model snapshots, Playwright, LaTeX) fail with `No space left on device`, or stall at "Sandbox user agent ready" with no trajectory. Run those on `--sandbox docker`.

## Cached installs

BenchFlow adds the model proxy and the built-in agent's install to the sandbox image as two layers. Daytona keeps a built image for 24 hours and reuses it when the same recipe comes back. A trial skips an install only when the layer's key matches.

Custom agent installers, setup commands, hooks and runtime environment overrides install in each trial instead. `--fresh-install` turns the layers off.

A cold layer build takes longer than the installs it saves, so a trial carries the layers only when they pay off:

- this process already built the same recipe (an earlier trial of the run); or
- the run starts the image again: `--trials` above 1, or `BENCHFLOW_DAYTONA_LAYER_REUSE=1`. Then one trial of each image builds the layers. Trials that start while it builds install in their sandbox and do not wait.

Every other trial installs in its sandbox. Its `sandbox.json` shows `install_layers.mode` `fresh` and a `fresh_reason`. Daytona's build cache cannot be queried ahead of time, and it may not keep a build for its stated 24 hours, so a new run does not count on an earlier run's build. Content snapshots (`BENCHFLOW_DAYTONA_SNAPSHOTS=content`) are opt-in and keep their own rule: they register the layered image once and reuse it.

The encoded installers use bounded Dockerfile lines below Daytona's 65,535-byte
limit. Decoding preserves the installer bytes. Both layers preserve the proxy's
private directory permissions so its custody check can accept it. A failed layer build still falls
back to installation inside the sandbox.

## Kept services

A task keeps services when it declares `x-keep-services`, or when it matches,
by digest, an entry in BenchFlow's Terminal-Bench 2 list. When the agent runs
as a non-root user, BenchFlow holds those services frozen from the kill after
the agent's turn until the verifier starts. Native Daytona cannot freeze them.

A native Daytona trial is refused before any sandbox is created when all of
these hold:

- the task keeps services and has no Compose file in `environment/`;
- the agent is neither `oracle` nor `nop`;
- `--sandbox-user` names a user other than `root` (the default is `agent`);
- the verifier runs in the agent's sandbox, in `main`, and is surely root
  before the sandbox exists: `[verifier] user = "root"`; or it is unset on a
  task of BenchFlow's Terminal-Bench 2 list (each one's verifier was measured
  root); or it is unset, the task has no `docker_image`, and its own
  Dockerfile's final stage ends on `USER root`;
- no `--base-image` replaces the task's image.

The refused trial is not scored (`kept_service_freezer`). Its `next:` line
names `--sandbox docker`, `--sandbox remote-docker`, or a Compose environment
on Daytona.

These runs are never refused early: the oracle, nop, an agent run as root
(`--sandbox-user root` or none), a separate verifier, and Compose tasks. A run
whose verifier account is known only once the sandbox exists is decided after
setup, by the same check as before (`_prove_kept_freezer`): a `[verifier]
user` other than `"root"` (`user = 0` included), or an account that comes from
a base image's `USER`, a prebuilt `docker_image` or `--base-image`.

On native Daytona with the default `--sandbox-user agent`, a model agent is
refused on these ten published TB2 tasks: `configure-git-webserver`,
`git-multibranch`, `hf-model-inference`, `install-windows-3.11`,
`kv-store-grpc`, `mailman`, `nginx-request-logging`, `pypi-server`,
`qemu-alpine-ssh` and `qemu-startup`. The oracle and nop run and are scored
on all ten.

## Images Daytona refuses

Daytona refuses task images that set a loader or shell start-up variable: `LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `LD_DEBUG_OUTPUT`, `LD_PROFILE`, `LD_PROFILE_OUTPUT`, `GCONV_PATH`, `LOCPATH`, `NLSPATH`, `HOSTALIASES`, `MALLOC_TRACE` or `ENV`. An image `BASH_ENV` is cleared, not refused. Use an image without those settings, or another sandbox.

Compose tasks on Daytona support only the `bridge`, `host` and `null` network drivers.

## Clean up

Daytona decides when a sandbox is abandoned, not a clock on your machine:

- Every sandbox is created with Daytona's inactivity stop at 15 minutes.
- While a run uses a sandbox, it polls its commands and sends Daytona an activity heartbeat every minute. A run that is paused, has lost its network or has died sends neither, so Daytona stops its sandbox after 15 quiet minutes.
- Each sandbox also carries the id of the run that made it and a lease, which that run renews every 10 minutes.
- When a run starts, auto-reap deletes your sandboxes that Daytona has already stopped, archived or failed and whose lease lapsed. A running (STARTED) sandbox is never deleted automatically, whatever its lease or age. One with a lapsed lease is only reported.
- Sandboxes without a lease (made by an older BenchFlow) are only reported. Branch snapshots are left to `bench sandbox cleanup`.

A create that failed without an answer, and a sandbox whose delete failed, stay on the run's list until they are confirmed deleted. The list is kept while the run lives, retried when it exits, and anything still left is written to `~/.local/state/benchflow/daytona-leftovers.jsonl` (`BENCHFLOW_DAYTONA_LEFTOVERS` sets the path), which `bench sandbox cleanup` reads.

A sandbox that fails while starting (an image build step or the container runtime failed) is deleted and replaced by a fresh one, up to three creates.

To clean up by hand:

```bash
bench sandbox list                       # your sandboxes (BENCHFLOW_DAYTONA_OWNER's)
bench sandbox cleanup --dry-run          # what would be deleted
bench sandbox cleanup --max-age 1440     # delete sandboxes older than a day
```

`bench sandbox cleanup` first deletes what your runs recorded in the leftovers file. It then deletes a sandbox that Daytona stopped and whose lease lapsed, and keeps one with a live lease. Any other sandbox (no lease, or running with a lapsed lease) is selected by age and activity alone, which cannot tell a quiet live run from an abandoned one, so run `--dry-run` first. `bench sandbox cleanup --all` deletes every sandbox of your owner label whatever its age, except a STARTED one (Daytona stops an abandoned one after 15 minutes without activity) and one a live run still holds. `--id <sandbox id>` deletes that sandbox even when it is STARTED or a live lease holds it. Other owners' sandboxes are never touched.

## Next steps

- Reuse built images: [Cache](../jobs/cache.md).
- What the sandbox guarantees: [Sandbox security](./security.md).
- Compare sandboxes: [Sandboxes](./index.md).
