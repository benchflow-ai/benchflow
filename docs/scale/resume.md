# Resume, retries and regrade

Long runs get interrupted, trials fail for reasons that have nothing to do with the agent, and verifiers turn out to be wrong. BenchFlow handles all three without rerunning work that finished: a plain rerun resumes a job, failures that another attempt could fix are retried, and `bench regrade` rescores saved work.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

## Resume a stopped job

Stop a run with Ctrl-C or `kill`, and it stops its running trials, removes their sandboxes and says how to continue:

```text
Terminated (SIGTERM): stopped 1 running trial and removed its sandbox.
Partial summary (what finished): ./jobs/stop/2026-11-04__18-48-30/summary.json
Resume with: bench eval resume ./jobs/stop/2026-11-04__18-48-30
```

Running the same command again does the same thing: it resumes the latest job in `--jobs-dir` when its settings match, keeps the trials that finished, and runs the rest.

`skillsbench-tasks/` holds the five SkillsBench tasks that [Check a benchmark](../audit/check-a-benchmark.md) copies out, and `slow.sh` is a script agent that only waits; make it with:

```bash
cat > slow.sh <<'EOF'
#!/bin/bash
sleep 25
echo "worked for 25 seconds"
EOF
chmod +x slow.sh
```

```bash
bench run skillsbench-tasks -a script:./slow.sh -j 1 --jobs-dir jobs/stop
```

The rerun's first lines:

```text
Resuming into existing job directory: 2026-11-04__18-48-30 (finished tasks are reused; pass a new job_name, or --fresh on the CLI, for a new run)
Job: 5 tasks, 2 done, 3 to run (concurrency=1, warm pool: up to 0 prepared sandboxes)
```

A finished job rerun this way runs nothing: `Job: 2 tasks, 2 done, 0 to run`. `bench eval resume <job>` reads the job's recorded settings, which never keep an `--agent-env` value that looks secret, so pass those again. The same holds for `--config-override`: the job's records keep the override's names, leave out every value whose name or shape could be a credential's (anything under an `env` or `headers` mapping but an entry named like a setting such as `MAX_OUTPUT_TOKENS`, any key or flag named like a password, key, secret, cookie or session cookie, a number too, a token or session one unless it names a setting (`max_output_tokens`, `token_budget`, `session_timeout` and a tokenizer's `eos_token` are kept and compared, `input_token` and `session_id` are not), other than a field the task's config declares such as `agent.token_budget`, any value shaped like a secret, and a list holding one, such as MCP args with `--api-key <value>`), and name the ones they left out. When any were left out, resume, `bench regrade`, verifier recovery and a continued run need the override again: set `BENCHFLOW_CONFIG_OVERRIDE` to what `--config-override` was (inline or `@file`). A rotated key is accepted; any other difference is refused. With nothing left out, nothing is needed. A trial that errored before it got a reward runs again; one with a reward, even 0, is kept. So is one whose agent's turn had already ended it (its time limit, the trial budget) before a later step failed: it keeps whatever reward was recorded before the failure, otherwise it is unscored; its `result.json` has `post_turn_failure_info`, and `bench regrade` re-runs only its verifier when the run saved the trial's final workspace. An ending that leaves the trial unscored anyway (a provider error, which the run retries, or an agent that did nothing) is run again, as it would be with nothing failing after it.

## Start fresh instead

A rerun resumes only a job made with the same settings. Run the oracle, then `nop`, into one jobs folder, and the second run starts a job of its own:

```text
Starting a new job 2026-11-04__20-25-14: the latest job in jobs (2026-11-04__20-25-03) ran agent='oracle' (this run: 'nop'). Its results stay in their own job; bench eval resume jobs/2026-11-04__20-25-03 finishes it with its own settings.
```

The same `nop` command again then resumes that job, and runs nothing: `Job: 1 task, 1 done, 0 to run`. `--fresh` always starts a new job in the same folder, and `--job <name>` names a job and resumes it if it exists. Pass `--fresh` the first time after an upgrade: a job made by 0.7 is resumed, not replaced ([Upgrade from 0.7](../versions/upgrade-from-0.7.md#old-jobs-and-resume)). For a comparison, give each arm its own `--jobs-dir`, so neither can resume the other.

## Retries

A trial that fails for a reason another attempt could fix is run again, up to 2 more times by default, each time in a new sandbox. Dropped connections, agent installs that failed, idle timeouts, transient provider errors, agent protocol errors and BenchFlow's own verifier failures are retried. A wall-clock timeout, a rejected login, a rate limit or a usage limit isn't; neither is a wrong answer.

`--retry-attempts N` changes the number; `--retry-attempts 0` turns retries off. For a published comparison, set it on purpose and report it: a retried success counts as a pass.

Each attempt gets a folder of its own in the job, `<task>__<id>` like the first, so a task tried three times leaves three folders. The attempts of one task stay one trial. `bench results` and `bf.load_job` count the attempt the run kept, which is the last one it started. An attempt the run retried never counts, whatever reward it carries. The run writes its decision into each attempt's folder as `attempt-decision.json` (`retried` or `final`), and resume, `summary.json`, `bench eval metrics`, `bench regrade` and the exports read it. A job from before these records counts the scored attempt, or the newest if none was scored. A task whose attempts were all retried, because the run stopped during a retry, has no result yet: it is listed as interrupted, its spend stays in the job's cost, and a resume runs it again. `--attempts all` (in Python, `attempts="all"`) lists every attempt instead, and each trial's `attempts` keeps them all, oldest first, so you can count how many trials needed a retry.

## Regrade saved work

`tasks/half-life` is the task [Write a task](../tasks/write-a-task.md) builds and shows how to copy.

When a verifier turns out wrong, fix it and score the saved work again, without running the agent. The run must have kept each trial's final workspace, with `--save-outputs` (`--freeze-workspace`, as `bench regrade --help` calls it):

```bash
bench run tasks/half-life -a claude -m xiaomi/mimo-v2.6-flash --save-outputs
# fix verifier/test.sh, then:
bench regrade jobs/<job>
```

[Verifiers](../tasks/verifiers.md#fix-a-verifier-and-rescore) shows a regrade turning a wrongly passed trial into a fail. `bench regrade` doesn't rescore multi-step trials, and 0.9.2 refuses a trial that ran with a separate verifier and points back to `bench run` (0.10.0rc2 regrades it).
