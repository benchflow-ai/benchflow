# Run your own agent

BenchFlow scores any agent that speaks the Agent Client Protocol (ACP): it installs your agent in the sandbox, sends it the task, records what it does, and grades the result like any built-in agent's. For tests that need no model, such as trying a cheat against a verifier, a shell script can stand in for an agent.

## Choose how your agent talks to BenchFlow

| Way | Use it for | What you write |
|---|---|---|
| A manifest | an agent installed with one shell command | `manifest.toml` in a folder named by `BENCHFLOW_AGENTS_DIR` |
| A script agent | a fixed sequence of commands, no model | a shell script, run with `-a script:<file>` |
| The agents repository | the community agents BenchFlow already knows | nothing: `bench agents` lists them |

## An ACP agent

The smallest useful agent is a loop: send the task to a model with one `bash` tool, run the commands it asks for, stop when it answers without one. [`docs/examples/minimal-agent/manifest.toml`](../examples/minimal-agent/manifest.toml) is that agent, about 90 lines of Python inside one manifest. Its top lines:

```toml
contract_version = "1.0"
name = "minimal-agent"
description = "A minimal ACP agent: one bash tool, any OpenAI-compatible model"
api_protocol = "openai-completions"
launch_cmd = "python3 /opt/minimal-agent/agent.py"
install_cmd = '''
set -e
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3; }
mkdir -p /opt/minimal-agent
cat > /opt/minimal-agent/agent.py <<'PY'
...
PY
'''

[env_mapping]
BENCHFLOW_PROVIDER_BASE_URL = "MINIMAL_AGENT_BASE_URL"
BENCHFLOW_PROVIDER_API_KEY = "MINIMAL_AGENT_API_KEY"
BENCHFLOW_PROVIDER_MODEL = "MINIMAL_AGENT_MODEL"
```

- `install_cmd` runs as root in the sandbox before the agent starts; a published agent would `pip install` or `npm install` itself here.
- `launch_cmd` starts it. BenchFlow speaks ACP over its standard input and output, so the agent prints nothing else there; logs go to standard error.
- `[env_mapping]` hands the agent the model's address, key and name under its own variable names. They point at BenchFlow's model proxy, so `-m` can name any provider and the proxy counts every token.

Point BenchFlow at a folder holding the manifest, and run the agent like any other. The manifest comes from BenchFlow's repository, the task from SkillsBench, and the model here is DeepSeek's, so export your key first ([Agents and models](./agents.md#a-model-from-another-provider)):

```bash
mkdir -p my-agents/minimal-agent
curl -fsSL -o my-agents/minimal-agent/manifest.toml \
  https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/minimal-agent/manifest.toml
export BENCHFLOW_AGENTS_DIR=$PWD/my-agents
export DEEPSEEK_API_KEY=...
bench agents minimal-agent
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation -a minimal-agent -m deepseek/deepseek-v4-flash
```

```text
minimal-agent
  Description: A minimal ACP agent: one bash tool, any OpenAI-compatible model
  Protocol:    acp
  Launch:      python3 /opt/minimal-agent/agent.py
  Requires:    (none)
...
[PASS] econ-detrending-correlation (reward=1.00, tools=15)
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Cost:      $0.00, 104,532 tokens
Time:      59s
```

The trial's `trajectory/acp_trajectory.jsonl` holds what the agent reported, one event per line; here, cut short:

```text
{"type": "user_message", "text": "In macroeconomics, understanding business cycle dynamics requires separating the trend component ...
{"type": "agent_message", "text": "I'll start by exploring the data files.", ...
{"type": "tool_call", ..., "kind": "execute", "title": "cd /root && ls -la && file *.xls *.xlsx 2>/dev/null", "status": "completed", ...
```

Report every command as a `tool_call` event: BenchFlow counts them, and a trial whose agent reported nothing is flagged as an agent that did nothing.

### The manifest's keys

The first four are required. BenchFlow ignores a key it doesn't know, so a misspelled key is dropped without a warning; check the agent with `bench agents <name>`.

| Key | What it sets |
|---|---|
| `contract_version` | `"1.0"`; 0.10.0rc1 also reads `"1.1"`, which adds `model_route` |
| `name` | the name `-a` takes; `aliases` lists other names it accepts |
| `install_cmd` | the shell commands that install the agent, run as root before it starts |
| `launch_cmd` | the command that starts it; it speaks ACP on its standard input and output |
| `description` | the line `bench agents` shows |
| `protocol` | `"acp"`, the only one |
| `api_protocol` | the model API the agent speaks, `anthropic-messages`, `openai-completions` or `openai-responses`, so the proxy serves the matching endpoint |
| `env_mapping` | BenchFlow's provider variables, `BENCHFLOW_PROVIDER_BASE_URL`, `BENCHFLOW_PROVIDER_API_KEY` and `BENCHFLOW_PROVIDER_MODEL`, under your agent's own names |
| `default_model` | the model when `-m` is left out |
| `requires_env` | variables that must be set before a run starts |
| `install_timeout` | seconds `install_cmd` may take; 900 by default |
| `acp_model_format` | how the model's name is sent over ACP: `bare` (the default), `provider/model` or `registered-provider/model` |
| `supports_acp_set_model` | `false` for an agent that takes its model at launch, not over ACP |
| `skill_paths`, `home_dirs` | where the agent reads skills (`$HOME/.claude/skills` for Claude Code), and other folders under its home to copy to the agent's user |

To ship files with the agent, write them from `install_cmd`, as the minimal agent writes its `agent.py`, or download them there.

Later 0.10 builds give the agent BenchFlow's variables under neutral names inside the sandbox: `RUNTIME_` takes the place of `BENCHFLOW_` (`BENCHFLOW_PROVIDER_MODEL` arrives as `RUNTIME_PROVIDER_MODEL`), in its environment and in its `launch_cmd`. 0.9.1 and 0.10.0rc1 pass the `BENCHFLOW_*` names. A file the install writes isn't renamed: an agent that reads the variables there reads the `RUNTIME_` name first and the `BENCHFLOW_` one after it, or takes the values under its own names through `env_mapping`, which works the same in every release.

BenchFlow's own folder in the sandbox moves in those builds from `/opt/benchflow` to `/opt/runtime`, and BenchFlow no longer makes `/opt/benchflow`. An install that writes there, or anywhere outside `/opt/runtime`, must leave each folder on the way readable to the agent's user itself; otherwise the agent's user can't run what it installed (`Permission denied`).

### An agent that calls no model

0.9.1 runs every agent but `oracle` and `nop` with a model, through its proxy. An agent that calls no model, such as a fixed program you're testing, is refused without one:

```text
agent 'primes-agent' has no default model; pass --model (refusing to fall back to 'claude-haiku-4-5-20251001' from a different provider)
```

and with `-m none`, the proxy then asks for a key: `LiteLLM route for model 'none' requires OPENAI_API_KEY`, which the run lists as "other error (infrastructure problem)". On 0.9.1, give it both; the key is never used:

```bash
OPENAI_API_KEY=unused bench run tasks/primes -a primes-agent -m none
```

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Cost:      none: no model tokens were used
```

0.10.0rc1 has a manifest setting for it: with `contract_version = "1.1"` and `model_route = "agent"`, BenchFlow starts no proxy for the agent, and the model and key are optional. The same agent, with those two lines, ran with neither:

```bash
bench run tasks/primes -a primes-agent
```

```text
[PASS] primes (reward=1.00, tools=1)
Cost:      none: no model tokens were used
Time:      11s
```

An agent with `model_route = "agent"` that does call a model uses its own endpoint and key, so BenchFlow records none of its calls or tokens.

### The messages BenchFlow sends

| Request | What the agent answers |
|---|---|
| `initialize` | `{"protocolVersion": 1, "agentCapabilities": {}}` |
| `session/new`, with `cwd` | `{"sessionId": "..."}`; work in `cwd` |
| `session/prompt`, with the task as text blocks | do the work, sending `session/update` notices, then `{"stopReason": "end_turn"}` |
| `session/cancel` | stop the current prompt |
| anything else | `{}` |

## A script agent

`tasks/half-life` is the task [Write a task](../tasks/write-a-task.md) builds and shows how to copy.

`-a script:<file>` runs a shell script in the sandbox as the agent's user, with no model and no credentials. It's the fastest way to test a verifier against a specific wrong answer or cheat:

```bash
bench run tasks/half-life -a script:probes/look.sh
```

The script's output goes to the trial's `agent/script.txt`. [Stop reward hacking](../tasks/reward-hacking.md#probe-your-verifier) and [Check a benchmark](../audit/check-a-benchmark.md#probe-the-verifiers-with-a-cheating-agent) use script agents this way.

## Agents from the agents repository

`bench agents` also lists agents that live in the community repository [benchflow-ai/agents](https://github.com/benchflow-ai/agents), such as GitHub Copilot CLI. BenchFlow fetches that repository the first time you list agents. `bench agents <name>` shows how each one logs in; GitHub Copilot CLI, for example, takes any OpenAI-compatible provider through its own `COPILOT_PROVIDER_*` variables.
