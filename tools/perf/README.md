# tools/perf: the speed instrument

Scripts that time BenchFlow on a build VM, so a before and an after are measured with the same instrument. Every script prints a `# provenance` line (the commits of the arms, the BenchFlow and Python versions in each arm's venv, uv, Docker, the CPU and the image digests it used) and keeps it beside its CSV as `<csv>.provenance.jsonl`. `summarise` turns any of the CSVs into medians with intervals.

| Script | Baseline | What it times |
|---|---|---|
| `cli_startup` | B2 | `bench --version`, `--help`, `eval run --help`, `agent list`, and the interpreter floor (`python -c pass`, `import typer`) per arm; one `-X importtime` profile per arm |
| `install_footprint` | B3 | a cold `uv tool install` of each arm's wheel: wall time, distributions installed, cache and venv bytes |
| `proxy_install` | B9 | the in-sandbox LiteLLM proxy's install, uv against pip, in a fresh container named by digest (`--vertex` adds the Vertex extra) |
| `setup_trials` | B4d, B5d | fake-model, nop and oracle trials for two or three arms, interleaved, on Docker or Daytona; one CSV row per trial with every phase and span of its `timing.json` |
| `summarise` | all | per arm: n, median and its 95% bootstrap interval (10,000 resamples, fixed seed); each arm minus the baseline arm, paired by round; passes with a Wilson 95% interval |

Rules these scripts assume: both arms on the same VM in one session, interleaved run by run (the scripts rotate the order every round); at least 5 trials per arm for setup spans, 3 for verifier and install timings, 30 runs after 3 warm-ups for CLI start-up; nothing else heavy on the VM while they run; controls first (`setup_trials --agent oracle` scores 1 and `--agent nop` scores 0 in both arms before any comparison).

## The spans they read

`timing.json` keeps its phases (`environment_setup`, `agent_setup`, `agent_execution`, `verifier`, `total`, to 0.1 s) and gains spans (to 1 ms). A span is absent when its step did not run (nop installs no agent, the oracle starts no proxy, a host proxy installs nothing), never 0. A step that runs twice in a trial adds up.

| Span | From, to | Inside |
|---|---|---|
| `slot_wait` | the job takes the trial's concurrency slot, to the trial start (before the start pacer, the start jitter's in-slot sleep was inside it) | before `total` starts |
| `start_pace_wait` | above `--concurrency 16`, the trial's wait for its start token, with no slot held | before the slot |
| `install_agent` | the agent install (and a native harness's CLI) | none: before `agent_setup` |
| `litellm_install` | the in-sandbox LiteLLM proxy's install | `agent_setup` |
| `proxy_health_wait` | launching the proxy, to its first healthy answer | `agent_setup` |
| `acp_connect` | starting the ACP agent and opening its session | `agent_setup` |
| `cloud_metadata_block` | refusing the cloud metadata endpoints to the agent user under an open network, an iptables install included | `agent_setup` |
| `firewall_power` | checking that the image hands the agent user no way to change the firewall BenchFlow holds it with, and dropping NET_ADMIN from its commands | `agent_setup` |
| `harden_before_verify` | `total` and one sub-key per step: `kill_sandbox_user_procs`, `clear_verifier_output`, `prepare_app_dir`, `reclaim_disk`, `restore_workspace`, `freeze_workspace`, `cleanup_conftests`, `build_verifier_env` | `verifier` |
| `cleanup` | the trial's teardown, sandbox stop included. In separate verifier mode, when the agent's sandbox stops before the transfer, only what is left: closing the agent, the usage-tracking check, the lease, and the stop again if the early one failed (the rest moved to `agent_release`) | `total` |
| `agent_stop_to_reward` | the solver's end (`verify()`'s entry), to the rewards being set; shared and separate mode | `total` (it holds `verifier`) |
| `agent_release` | separate verifier mode: skill export, the usage runtime's stop, the egress proxy's stop, usage finalization, the session-log price, and the agent sandbox's stop | `agent_stop_to_reward` |
| `agent_sandbox_stop_before_verify` | separate verifier mode: the agent sandbox's stop alone | `agent_release` |
| `verifier_sandbox_ready_wait` | separate verifier mode: the wait for the verifier sandbox once the transfer is packed | `agent_stop_to_reward` |
| `verifier_sandbox_overlap` | separate verifier mode: the seconds both sandboxes existed (not a step: a length) | none |
| `sandbox_ready` | the trial start to its sandbox started and its agent installed, as the trial waited for it (with a warm-pool sandbox, only the wait) | none: from the trial start |
| `warm_pool_wait` | a Docker job with more trials than `--concurrency`: the trial start to its prepared sandbox being ready (0 when it already was) | `sandbox_ready` |
| `warm_pool_idle` | the same: the seconds a ready prepared sandbox waited for its trial's slot. That trial's `environment_setup` and install spans were spent before its start, and its `total` counts from its setup, so it includes these seconds | `total` |

On Docker and Daytona, the clear, app-directory, reclaim, freeze and hook-cleanup commands share one script after the kill and pre-freeze trust checks (two scripts around an opt-in workspace restore). Their sub-keys are each step's seconds inside the sandbox (from `/proc/uptime`, 10 ms steps, so a quick step can read 0), without the saved exec round trips; `total` stays the host wall time. The kill, `restore_workspace` and `build_verifier_env` stay timed on the host. Trials with kept services retain the separate command path and host timings.

`setup_trials` adds `unnamed`: `total` minus `environment_setup`, `install_agent`, `agent_setup`, `agent_execution`, `verifier` and `cleanup` (the spans that do not overlap). In separate verifier mode the capture, `agent_release` and the verifier sandbox's setup, transfer and teardown are outside `verifier`, so they land in `unnamed`; `agent_stop_to_reward` holds all of them.

## The fake-provider route

One route for every fake-model speed number, on Docker:

- `setup_trials` starts the deterministic tier's fake provider (`tests/integration/deterministic/task/environment/fake_llm/fake_llm.py`) on the build VM itself, with its own Python (`python -I`), bound to the Docker bridge's gateway address (`docker network inspect bridge`, usually `172.17.0.1`) on a free port from 8900 to 8999, and stops it when it ends. Nothing in the task image runs it, so the image needs no Python.
- The agent's model calls go to BenchFlow's LiteLLM proxy, whose upstream is that address (`BENCHFLOW_PROVIDER_BASE_URL`, with a dummy key).
- `--proxy host` (Docker's default): the host proxy calls the fake.
- `--proxy sandbox`: the task carries an egress policy (`agent.network_mode: allowlist`, `allowed_hosts: [example.com]`), which moves the proxy into the sandbox. The in-sandbox proxy calls the fake at the bridge address: the policy confines only the agent's uid, which can reach nothing but the proxy on loopback.
- Why not a sidecar container on the trial's network: the deterministic task's compose project is BenchFlow's, and adding a service changes the trial being measured; the bridge address is the one the host proxy already uses to reach Docker sandboxes.
- On Daytona the VM is out of reach: the proxy (always in the sandbox there) calls the deterministic task's own fake service at `http://127.0.0.1:8911`, so only `--task-variant deterministic` runs there.
- `--task-variant no-python` is the same task on an image with no Python (curl and CA certificates only) and without the in-sandbox fake service.

## Running them on a build VM

Each arm is a commit's tree sent with `git archive` and its own venv, set up like the round's test command (pinned uv 0.13.4 and uv-managed CPython 3.12 under `$W/tools`). Per arm:

```
W=<work folder>; export PATH=$W/tools/bin:$PATH UV_CACHE_DIR=$W/uv-cache UV_PYTHON_INSTALL_DIR=$W/tools/python UV_PYTHON_PREFERENCE=only-managed UV_PYTHON=3.12 HOME=$W/home
(cd $W/base && UV_PROJECT_ENVIRONMENT=$W/venv-base uv sync --locked --extra dev)
(cd $W/tip && UV_PROJECT_ENVIRONMENT=$W/venv-tip uv sync --locked --extra dev)
```

Run the scripts from one tree (the instrument; the newer one), with that tree's venv, from its root:

```
cd $W/tip; export UV_PROJECT_ENVIRONMENT=$W/venv-tip
uv run python -m tools.perf.cli_startup --arm base=<base hash>:$W/venv-base --arm tip=<tip hash>:$W/venv-tip --runs 30 --warmups 3 --out $W/perf/b2.csv
(cd $W/base && uv build --wheel -o $W/wheels/base); (cd $W/tip && uv build --wheel -o $W/wheels/tip)
uv run python -m tools.perf.install_footprint --wheel base=<base hash>:$(ls $W/wheels/base/*.whl) --wheel tip=<tip hash>:$(ls $W/wheels/tip/*.whl) --runs 3 --work $W/perf/b3-work --out $W/perf/b3.csv
uv run python -m tools.perf.proxy_install --image ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 --runs 3 --out $W/perf/b9.csv
uv run python -m tools.perf.setup_trials --arm base=<base hash>:$W/venv-base --arm tip=<tip hash>:$W/venv-tip --agent oracle --proxy host --count 1 --work $W/perf/ctl-oracle --out $W/perf/controls.csv
uv run python -m tools.perf.setup_trials --arm base=<base hash>:$W/venv-base --arm tip=<tip hash>:$W/venv-tip --agent nop --proxy host --count 1 --work $W/perf/ctl-nop --out $W/perf/controls.csv
uv run python -m tools.perf.setup_trials --arm base=<base hash>:$W/venv-base --arm tip=<tip hash>:$W/venv-tip --agent claude-agent-acp --proxy sandbox --count 5 --work $W/perf/b4d --out $W/perf/b4d.csv
uv run python -m tools.perf.summarise $W/perf/controls.csv --by agent --pass-column reward
uv run python -m tools.perf.summarise $W/perf/b4d.csv --pass-column reward
uv run python -m tools.perf.summarise $W/perf/b2.csv --metric seconds --by command --where warmup=0
uv run python -m tools.perf.summarise $W/perf/b9.csv --arm-column tool --metric seconds
```

- `setup_trials` needs Docker on the VM (with the compose and buildx plugins) and nothing else: no key. It keeps each trial under `--work`/`jobs/<agent>-<proxy>-<backend>-<variant>/<arm>/<round>`, its log under `--work`/`logs/<agent>-<proxy>-<backend>-<variant>/` and the fake's under `--work`/`fake/<agent>-<proxy>-<backend>-<variant>/` (never print the logs; they can hold whatever a run wrote). It never deletes or overwrites a trial: when an earlier run with the same agent, proxy, backend and task variant already wrote into `--work`, it refuses to start, so a rerun takes a fresh `--work`. One CSV can collect several runs (the controls do); split it with `--by` when you summarise it.
- On Daytona: `--backend daytona` with the key read in the same command from the key file the orchestrator placed (`set -a; . $HOME/.config/benchflow/daytona.env; set +a; BENCHFLOW_DAYTONA_OWNER=<label> uv run python -m tools.perf.setup_trials ...`), and the round's meter running.
- `install_footprint` installs into folders it makes under `--work` and removes them after each run; `proxy_install` removes its container after each run; `setup_trials` leaves the pulled `ubuntu:24.04` base image (BenchFlow removes each trial's sandbox).
- Report the VM name, its machine type and CPU count (in the provenance line), both commits, and the image digests the provenance lines name.
