# Examples

Small tasks, scripted agents, job files, and a dataset that exercise every feature of `bench`, each with the outcome it should have. Run them to learn a feature, or to check a build: every example here is also a step of an end-to-end scenario (`e2e/scenarios/examples-*.toml`), which checks that outcome for you.

Run the commands from this folder (`cd examples`). Nothing the commands write lands among the tracked files: jobs go to `jobs/`, and the examples that convert, export, or edit a task write into or work on a copy in `scratch/`; both are ignored by git, so `rm -rf jobs scratch` starts over. Most need only a sandbox backend (Docker, or `DAYTONA_API_KEY` for Daytona); add `-s docker` or `-s daytona` to pick one. The model examples need `MIMO_API_KEY` (MiMo's token plan); any other provider works the same with its own `-m <provider>/<model>`.

To run them all as the e2e suite does:

```bash
e2e/run.sh --bench target/release/bench --backend docker --tier full -k examples   # from the repository root
```

## What is here

| Folder | What it holds |
|---|---|
| `tasks/` | One task per feature or failure, each a few lines long |
| `agents/` | Scripted attempts for `-a script:<file>`: they run in the sandbox like the oracle |
| `skills/` | A skill folder for `--skill` |
| `job-files/` | Job files for `-c`, in TOML, YAML, and JSON |
| `datasets/mini/` | A local `dataset.toml` with two tasks |

## Task formats

| # | Example | Command | Expected |
|---|---|---|---|
| E01 | Every task here is valid | `bench check tasks` | 2 tasks with errors (`check-errors` on purpose, and `hello-tb1`, which needs `bench import`); the rest ok |
| E02 | `bench check` finds a misspelled key and a split-layout spelling | `bench check tasks/check-errors` | exit 1: `T201` (`timout`, "did you mean `timeout`") and `T202` (`timeout_sec`), each with its line and column |
| E03 | `--fix` repairs both | `mkdir -p scratch && cp -R tasks/check-errors scratch/ && bench check scratch/check-errors --fix` (on a copy, so E01 and E02 keep their results) | exit 0; the copy's task.md now says `timeout = "2m"` and `timeout = "1m"` |
| E04 | A Terminal-Bench v1 task does not run as it is | `bench run tasks/hello-tb1` | exit 2: no tasks found; the message gives `bench import tasks/hello-tb1` |
| E05 | Import it to task.md | `bench import tasks/hello-tb1 scratch/tb1-taskmd && bench check scratch/tb1-taskmd` | exit 0 both |
| E06 | The split layout to task.md and back | `bench import tasks/hello-split scratch/split-taskmd && bench export scratch/split-taskmd scratch/split-again` | `scratch/split-again/task.toml`, `instruction.md`, `tests/test.sh` |
| E07 | BenchFlow 0.8's task.md to task.md | `bench import tasks/hello-08 scratch/v08-taskmd` | `scratch/v08-taskmd/task.md` |
| E08 | A rubric does not fit the split layout | `bench export tasks/rubric-judged scratch/rubric-split` | refused, naming the rubric; `--allow-degraded` exports it anyway |
| E20 | All three formats run as they are | `bench run tasks/hello tasks/hello-split tasks/hello-08` | `3/3 passed` |

## Agents without a model

| # | Example | Command | Expected |
|---|---|---|---|
| E21 | Your own script as the agent | `bench run tasks/hello -a script:agents/write-hello.sh` | scored, reward 1 (`agents/wrong-answer.sh`: reward 0) |
| E25 | Stages, the first a gate | `bench run tasks/stages` | reward 1, two stage results; with `-a script:agents/stage-one-wrong.sh`, reward 0 and the second stage never runs |
| E37 | `nop` does nothing | `bench branch tasks/hello -a nop --children 2` | a fork with value 0.000 |

## Reward files

Each `reward-*` task's verifier writes the same file whatever the agent did: `bench run tasks -i 'reward-*'` (E22).

| Task | The file | Expected |
|---|---|---|
| `reward-fraction` | `reward.txt` = `0.75` | scored, 0.75 |
| `reward-json` | `reward.json` = `{"reward": 0.5, "details": {"tests_passed": 3}}` | scored, 0.5: numbers under `details` (and `raw`, `errors`, `metadata`, ...) are kept as they are |
| `reward-json-bad-metric` | `reward.json` = `{"reward": 0.5, "tests_passed": 3}` | unscored, `R302`, the task's: every other top-level number must be in [0, 1] too, as in BenchFlow 0.8 |
| `reward-json-metrics` | `reward.json` with metrics and no `reward` key | unscored, `R302`: it does not say which number is the reward |
| `reward-out-of-range` | `reward.txt` = `1.5` | unscored, `R302`: rewards are in [0, 1] |
| `reward-negative` | `reward.txt` = `-1` | unscored, `R302` |
| `reward-nan` | `reward.txt` = `nan` | unscored, `R302`: not a finite number |
| `reward-empty` | an empty `reward.txt` | unscored, `R302` |
| `reward-word` | `reward.txt` = `yes` | unscored, `R302`: not a number |

Each refusal's reason is in the trial's `result.json` (`problem.message`) and in `bench results <job> -t` (E22b). An unscored trial is left out of the pass rate and listed apart; `bench explain R302` says more.

## Timeouts and failures

| # | Example | Command | Expected |
|---|---|---|---|
| E23 | The agent outlives its 10 s | `bench run tasks/agent-timeout` | scored, reward 0 |
| E23 | The verifier outlives its 5 s | `bench run tasks/verifier-timeout` | unscored, `R303`, the task's |
| E23b | The image build outlives its 15 s | `bench run tasks/build-timeout` | errored, `R102`, recorded as the sandbox's and transient (a slow network looks the same), so it is retried. Daytona allows 2 minutes more for its builder and leaves the builder running to finish (a known limit); a retry waits for it rather than starting another |
| E23 | The Dockerfile fails | `bench run tasks/broken-image` | errored, `R101`, the task's |
| E65 | A readiness check that never passes | `bench run tasks/ready-never` | errored, `R105`, the task's, before any agent starts (`ready` follows Compose's healthcheck: `timeout` is per check, then `retries`) |
| E24 | More time for slow agents | `bench run tasks/agent-timeout --agent-timeout-multiplier 9` | reward 1 |
| E59 | A model-agent flag with a script | `bench run tasks/hello -a script:agents/slow.sh --idle-timeout 10s` | a note: `--idle-timeout` has no effect on a script, which runs to its task's timeout (reward 1 after 60 s) |
| E39 | Interrupt and resume | `bench run tasks/hello -a script:agents/slow.sh --job slow`, Ctrl-C twice, then the same command | the first exits 1 with the trial stopped (`R001`, not counted); the second runs it again: reward 1 |

## Task structure

| # | Example | Command | Expected |
|---|---|---|---|
| E26 | A read-only mount, a declared output, a hook before grading | `bench run tasks/mount-outputs` | reward 1; the output is saved in the trial folder |
| E27 | A verifier in a sandbox of its own | `bench run tasks/separate-verifier` | reward 1 |
| E54 | An allowlist (`198.18.0.0/15`) and a blocklist (a reserved `.invalid` name and `127.0.0.2`), probed with no outside host | `bench run tasks/net-allowlist tasks/net-blocklist` | reward 1 each: an unlisted reserved address (198.51.100.7, routed nowhere) is refused at once while a listed one passes; the blocked name is pinned and the blocked loopback listener refused, while a listener on 127.0.0.3 answers. The oracle runs as root, and the policy holds for it too |
| E54m | `open` and `none`, by hand (not in the e2e suite) | `bench run tasks/net-open tasks/net-none` | reward 1 each. Both read the sandbox's own network address and a reserved address. They need a host whose network either routes outward or refuses at once: on a backend whose own network block silently drops packets (Daytona's, unverified), `net-open` cannot tell `open` from that block, since nothing inside the sandbox can. On a Docker host with no outside route both still hold: `net-open` takes a no-route answer when its own-address server answers, and `net-none` only when the sandbox has no address |
| E68 | The same policies by address (no name to pin): an allowlist of a reserved range, a blocklist of a reserved address and the loopback address 127.0.0.2, and a `none` for the agent's phase only | `bench run tasks/net-ip-allowlist tasks/net-ip-blocklist tasks/net-ip-none` | reward 1 each: the oracle, which runs as root, reaches neither address |
| E55 | A service beside the agent's container | `bench run tasks/sidecar -s docker` | reward 1 on Docker; on Daytona, errored `R401` (services need Docker or Podman) |

## Agents on a model

These run on MiMo v2.6 Flash (`MIMO_API_KEY`). Model runs are not deterministic; rewards below are what a working run gets on these trivial tasks.

| # | Example | Command | Expected |
|---|---|---|---|
| E43 | An ACP agent | `bench run tasks/hello -a opencode -m mimo/mimo-v2.6-flash` | reward 1; `trajectory/acp_trajectory.jsonl` and `trainer/atif.json` in the trial folder |
| E44 | The same agent headless | `... --harness headless` | reward 1 |
| E45 | BenchFlow's own tool loop | `bench run tasks/hello -a builtin -m mimo/mimo-v2.6-flash --max-tool-calls 10` | reward 1 |
| E46 | Terminus-2 | `bench run tasks/hello -a terminus-2 -m mimo/mimo-v2.6-flash` | scored |
| E47 | An SDK runner | `bench run tasks/hello -a openhands-sdk -m mimo/mimo-v2.6-flash` | scored |
| E71 | A provider that rejects every request | `bench run tasks/hello -a codex -m mimo/mimo-v2.6-flash` (over ACP, where `--ak web_search=disabled` is refused: codex-acp does not take `-c` options) | errored, `R214`, the model provider's: MiMo answers Codex's `web_search` tool with HTTP 400; never a scored 0 |
| E72 | The same agent over ACP with its settings in `CODEX_CONFIG` | `bench run tasks/hello -a codex -m mimo/mimo-v2.6-flash --env 'CODEX_CONFIG={"web_search":"disabled"}'` | scored (reward 1 in our run): codex-acp reads `CODEX_CONFIG`, not `-c` options |
| E69 | A computer-use agent on a virtual desktop | `bench run tasks/desktop-hello -a computer-1 -m mimo/mimo-v2.6-flash` | the agent opens a page in Chromium, types, clicks, and reports the code it shows: reward 1; each screenshot in `agent/screenshots/` |
| E70 | DSPy's recursive language model | `bench run tasks/hello -a dspy-rlm -m mimo/mimo-v2.6-flash` | reward 1 |
| E48 | Task skills | `bench run tasks/skills -a opencode -m mimo/mimo-v2.6-flash` | reward 1 (the answer is only in the skill); with `--no-task-skills`, reward 0 |
| E49 | A skill of your own | `bench run tasks/hello -a opencode -m mimo/mimo-v2.6-flash --skill skills/house-style` | scored |
| E50 | One session across stages | `bench run tasks/stages -a opencode -m mimo/mimo-v2.6-flash --resume-sessions` | reward 1 |
| E51 | A rubric judged by a model against a reference | `bench run tasks/rubric-judged` (the oracle) | reward 1; `verifier/judgments.jsonl` has one call per model-judged criterion |
| E51 | The same rubric on a wrong answer | `bench run tasks/rubric-judged -a script:tasks/rubric-judged/controls/wrong-cause.sh` | reward 0 |
| E66 | Codex, then a new trial started from its saved session (`--load-trajectory`) | `bench run tasks/hello -a codex --harness headless --ak web_search=disabled -m mimo/mimo-v2.6-flash --job codex-a`, then the same with `--load-trajectory <the rollout-*.jsonl in jobs/codex-a/*/agent/sessions/>` | the second trial continues that session (same id, more turns) and is graded. MiMo refuses Codex's `web_search` tool, hence `web_search=disabled`, which only the headless harness takes; over ACP, `--env 'CODEX_CONFIG={"web_search":"disabled"}'` does the same |
| E52 | A fork after the first prompt | `bench branch tasks/hello -a opencode -m mimo/mimo-v2.6-flash --at prompt:1 --resume-session --child "label=keep,prompt=..." --child "label=bye,prompt=..."` | two children; `keep` usually 1, `bye` 0 |
| E53 | A model's review of a task, and of failures | `bench check tasks/hello --review --model mimo/mimo-v2.6-flash`; `bench results jobs/<job> --analyze --model mimo/mimo-v2.6-flash` | the review after the checker's report; `analysis.json` in the job |

### Measured

Three trials of each model example on Daytona with MiMo v2.6 Flash, on 2026-10-11: E43 to E53 with `bench` 0.8.1-dev.0 at `416c35f4`, E66, E69 and E70 with a later build that has the examples. Pass rates with 95% Wilson intervals; on trivial tasks three trials can only bound a rate, not pin it.

| # | Example | Passed | 95% interval |
|---|---|---|---|
| E43 | OpenCode over ACP | 3/3 | 0.44–1.00 |
| E44 | OpenCode headless | 3/3 | 0.44–1.00 |
| E45 | builtin loop | 3/3 | 0.44–1.00 |
| E46 | terminus-2 | 3/3 | 0.44–1.00 |
| E47 | openhands-sdk | 3/3 | 0.44–1.00 |
| E48 | task skills: with the skill / with `--no-task-skills` | 3/3 / 0/3 | 0.44–1.00 / 0.00–0.56 |
| E49 | `--skill` | 3/3 | 0.44–1.00 |
| E50 | `--resume-sessions` across stages | 3/3 | 0.44–1.00 |
| E51 | rubric judge: the reference answer / the wrong one | 3/3 / 0/3 | 0.44–1.00 / 0.00–0.56 |
| E52 | prompt fork: child `keep` / child `bye` | 3/3 / 0/3 | 0.44–1.00 / 0.00–0.56 |
| E66 | Codex headless / started from its saved session | 3/3 / 3/3 | 0.44–1.00 each |
| E69 | computer-1 | 3/3 | 0.44–1.00 |
| E70 | dspy-rlm | 3/3 | 0.44–1.00 |

## Commands

| # | Example | Command | Expected |
|---|---|---|---|
| E28 | Results, per trial and as JSON | `bench results jobs/<job> -t`, `bench --json results jobs/<job>` | a row per trial; a JSON document |
| E29 | Traces | `bench export jobs/<job> scratch/<job>-hf`, `bench export jobs/<job> --format otel scratch/<job>.otel.jsonl` | `data/train.jsonl` and a dataset card; one trace per line |
| E30 | Prove a task | `bench test tasks/hello` | exit 0: the oracle passes, nop fails, the control fails, every probe refused |
| E31 | Catch a verifier that passes anything | `bench test tasks/always-passes --probes none` | exit 1: doing nothing scored 1 |
| E32 | Regrade after a verifier change | `mkdir -p scratch && cp -R tasks/hello-split scratch/ && bench run scratch/hello-split --save-outputs --job base`, change `42` to `41` in the copy's `scratch/hello-split/tests/test.sh`, `bench regrade jobs/base` | `Changed 1 of 1`; the new job is `jobs/base-regrade-1` |
| E33 | What an earlier job can supply | `bench run scratch/hello-split --save-outputs --diff base --dry-run` | `regrade hello-split` |
| E34 | Compare two jobs | `bench results jobs/nop jobs/oracle` | pass rates side by side, with the paired difference |
| E35 | A fork at the start | `bench branch tasks/hello --children 3` | value 1.000; `bench branches jobs/<job>` lists `c1` to `c3` |
| E36 | A fork after a stage, the parent going on | `bench branch tasks/stages --at stage:first --children 2 --parent continue` | value 1.000; the parent graded too |
| E40 | Try a task by hand | `printf "printf 'Hello, world!' > /app/hello.txt\nexit\n" \| bench shell tasks/hello` (Docker) | graded on exit: reward 1 |
| E41 | The viewer | `bench view jobs --no-open` (it prints its address, the first free port from 8765), then `curl <address>/api/compare?runs=nop,oracle` | JSON for both jobs |
| E60 | An outside service decides every step | `mkdir -p scratch && python3 agents/backend.py scratch/port.txt &`, then `bench run tasks/hello -a external --backend http://127.0.0.1:$(cat scratch/port.txt) -m scripted` | reward 1; the service hears the reward (`scoring`); `agent/backend_transcript.json` in the trial folder. `agents/backend.py` is a complete service in 100 lines of standard Python |
| E61 | Several trials per task | `bench run tasks/hello -a script:agents/wrong-answer.sh -k 3 --job three`, then `bench results jobs/three` | 3 trials, pass@k per task |
| E62 | Run a job's failures again | the same command with `--rerun failed` | the 3 scored failures run again: `0 of 3 trials kept, 3 to run`, then `0 reused, 3 ran now` |
| E63 | What BenchFlow keeps | `bench cache --only images` (Docker) | the task images it built for this owner (`BENCHFLOW_OWNER`), with sizes and last use |
| E64 | Check an agent installs and starts | `bench agents check opencode -s docker` | it answers ACP's `initialize` (no login, no prompt) |
| E42 | The machine | `bench doctor`, `bench --json doctor -s docker` | each check with how to fix it |
| E19 | Codes and schemas | `bench explain R303`, `bench schema job` | the code's meaning; the job file's JSON Schema |

## Job files and datasets

| # | Example | Command | Expected |
|---|---|---|---|
| E09 | Layered job files, TOML then YAML | `bench run -c job-files/base.toml -c job-files/nop.yaml --print-config` | `agent = "nop"`, `trials = 2`, and base's `[env]` kept |
| E10 | Include and exclude patterns, from JSON | `bench run -c job-files/include.json -a nop --dry-run` | the `reward-*` tasks but `reward-nan`, with nop (the flag wins over the file) |
| E11 | An empty list clears an earlier one | `bench run -c job-files/include.json -c job-files/clear-include.toml --print-config` | no `include` |
| E12 | A misspelled key | `bench run -c job-files/bad-key.toml --dry-run` | exit 2: `unknown field agnet` |
| E13, E38 | A local dataset | `bench run -d datasets/mini/dataset.toml` | `examples-mini@1.0`, `2/2 passed` |

## Not provided yet

| # | Example | Command | Expected |
|---|---|---|---|
| E67 | A sandbox clock (`[sandbox] clock`), a time zone (`[sandbox] timezone`), and a simulated world (`[world]`) | `bench check tasks/clock-start tasks/timezone tasks/world-sim`, then `bench run` on them | each checks clean (and `bench check` explains what a clock needs: `enforce`, and no network while it runs); a run refuses each by name before any sandbox starts (`R401`), and `--dry-run` names each refusal |

## Mistakes, refused before anything starts

| # | Example | Command | Expected |
|---|---|---|---|
| E14 | A misspelled agent | `bench run tasks/hello -a claud` | exit 2: "did you mean `claude`?" |
| E15 | A model with no key | `bench run tasks/hello -a opencode -m mimo/mimo-v2.6-flash --dry-run` with no `MIMO_API_KEY` | exit 3: `R211`, naming the variable |
| E16 | An option the agent lacks | `... --ak nope=1` | exit 2: the agent's options listed |
| E17 | `--load-trajectory` with an agent that cannot load sessions | `... -a opencode --load-trajectory session.jsonl` | exit 2: only Claude Code and Codex sessions load |
| E18 | Forks that cannot run | `bench branch tasks/hello --children 1`; `--at prompt:1` with the oracle; `--at stage:nope` | exit 2, with the reason |
