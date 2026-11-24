# Hosted evals

A hosted eval runs on BenchFlow's servers, not your machine: no Docker, no sandbox provider and no model key here. `bench hosted` queues the run, follows it, and fetches its rewards and trajectories.

## Set the key

The service takes a BenchFlow Hub key, which starts `bfk_`, or a workspace key that starts `bh_`. `bench hosted` reads it from `BENCHFLOW_HUB_KEY`, or from a file only you can read:

```bash
umask 077
mkdir -p ~/.config/benchflow
printf '%s\n' 'bfk_...' > ~/.config/benchflow/hub-key
```

`--key-file` names another file and wins over `BENCHFLOW_HUB_KEY`; `BENCHFLOW_HUB_KEY_FILE` names one to read when the variable is unset. A key file that others can read is refused (`chmod 600` it). No command takes the key as an argument, and none prints more of it than its kind (`bfk_…` or `bh_…`): the key, its secret part and anything else shaped like a key are masked in what the service sends back, before a message is cut short. A trajectories export is saved as the service sent it.

A `bfk_` key ends with a checksum, so a mistyped or cut key is refused before anything is sent.

## Name the service

`BENCHFLOW_HOSTED_URL`, or `--hub` on any command, names the hosted service's address. There is no default: with neither, every `bench hosted` command stops before sending anything. The address must be `https://`, except to this machine (`localhost`, `127.0.0.1`).

## Run a benchmark

Name a hub version, the agent and how much of it to run:

```bash
bench hosted run PKG-TERMINALBENCH-2-0 -a oracle -n 2 --wait
```

```text
queued run run_01...
page: https://.../runs/run_01... (sign in to read it)
run_01...: status queued, trials 0/2
run_01...: status running, trials 1/2
status succeeded, trials 2/2, mean reward 1
```

Without `-a` the oracle runs, the task's reference solution. `-m` names the model, `-i` keeps only the named tasks, `--trials` runs each task more than once, and `-j` runs trials side by side; the service may lower `-j` to fit its sandbox limit, and says so in a warning. `--max-sandbox-seconds` stops the run past that much sandbox time, `--secret NAME` passes a secret you stored on the service to the agent, and `--name` labels the run. `bench hosted estimate` takes the same selection and budget options (never a key: an estimate sends none) and says what the run would likely take, without starting it. `bench hosted submit --version PKG-...` is the same as `run`.

Without `--wait`, `run` prints the run's id and returns. Follow it later:

```bash
bench hosted status run_01...
bench hosted wait run_01... --timeout 3600
bench hosted events run_01...
```

`wait` exits 1 when the run failed or was cancelled. When it stops at `--timeout`, the run goes on. `bench hosted cancel` stops a run.

## Use your own model access

The service doesn't provide model accounts. You bring your own: your provider's API key (Anthropic, DeepSeek,
Moonshot/Kimi, Z.ai or Xiaomi MiMo), or your own endpoint, such as a [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)
you run or any Anthropic-compatible https URL. The key stays on the service. Its key proxy puts the key on each model
call, and the run's sandbox gets only the proxy's address and a token for that run. You are responsible for your
provider's terms.

For one run only (the service deletes the key when the run ends):

```bash
bench hosted run PKG-GPQA-DIAMOND -a claude-agent-acp -m deepseek/deepseek-flash -n 2 \
  --model-key deepseek=file:~/.config/benchflow/deepseek.key --max-cost-usd 1
bench hosted run PKG-GPQA-DIAMOND -a claude-agent-acp -m mimo-v2.6-flash -n 2 \
  --endpoint https://cpa.example.com --endpoint-key file:~/cpa_client.key
```

Kept in your workspace, then named with `--secret`:

```bash
bench hosted secret set DEEPSEEK --kind deepseek --from file:~/.config/benchflow/deepseek.key
bench hosted secret endpoint CPA --url https://cpa.example.com --from file:~/cpa_client.key
bench hosted run PKG-GPQA-DIAMOND -a claude-agent-acp -m mimo-v2.6-flash --secret CPA_URL --secret CPA_KEY
bench hosted secret list
bench hosted secret rm DEEPSEEK
```

How these behave:
- Each key comes from `file:PATH`, `env:VAR` or `-` (stdin), never from the command line.
- A stored endpoint is sent only the key stored with it.
- An endpoint is https only, reachable from the internet: the service's workers call it, so a CLIProxy on your own
  machine (localhost or a private address) can't be reached. Its key goes in `--endpoint-key`, never in the URL.
- A key file must be readable by you alone (`chmod 600`), as the hub key file must.
- `--max-cost-usd` is the run's budget: the proxy refuses model calls once its own count reaches it.
- The run's record has the proxy's tokens and cost per trial (`keyproxy_usage`).
- Today the proxy serves Claude Code (`-a claude-agent-acp`). Another agent would need its key inside the sandbox, and
  the service refuses that unless the run says why.

## Read the results

```bash
bench hosted results run_01...
bench hosted trajectories run_01... -o run.jsonl
```

`results` prints the run's mean reward, each reward key's mean and spread, the errors by type, and each trial's task, reward and outcome. `trajectories` (or `download`) writes every trajectory to the file you name, one JSON document per line: ATIF by default, or chat messages with `--format messages`. Every command takes `--json` for the service's own answer.

## When the service says no

| Answer | What it means | What to do |
|---|---|---|
| 401 or 403 | The key is wrong, revoked, expired, or lacks the scope | Check the key; starting runs needs `evals:run` |
| 400 | The run was refused: an unknown version, the budget, a cap | Read the reason in the message |
| 429 | Too many requests from this key | Try again later; `bench hosted` never retries by itself |
| 503 | The service, or its key check, is down for now | Try again after the time it names; this says nothing about the key |
