# Agents and models

An agent is the program that does the task: Claude Code, Codex, OpenCode and dozens more. You choose it with `-a`, its model with `-m`, and how hard the model thinks with `-e`. Each agent needs a login or an API key for its model provider.

## Pick an agent

These are the agents people run most, by downloads and by trials on BenchFlow:

| Agent | `-a` | Logs in with |
|---|---|---|
| Claude Code | `claude` | an Anthropic API key, or a Claude subscription |
| Codex | `codex` | an OpenAI API key, or a ChatGPT login |
| OpenCode | `opencode` | the key of the model's provider |
| Pi | `pi` | the key of the model's provider |
| Antigravity | `agy` | a Gemini API key |

Two more need no model: `oracle` runs the task's reference solution and `nop` does nothing. `bench agents` lists every agent, including GitHub Copilot CLI and others that come from the [agents repository](./your-own-agent.md#agents-from-the-agents-repository), and `bench agents <name>` shows one:

```bash
bench agents claude
```

```text
claude-agent-acp
  Aliases:     claude
  Description: Claude Code via ACP (Anthropic's Agent Client Protocol)
  Protocol:    acp
  Launch:      /opt/benchflow/bin/claude-agent-acp
  Requires:    ANTHROPIC_API_KEY (or login)
  Provider auth: Provider-prefixed models may use different credentials; Azure Foundry models use AZURE_API_KEY + 
AZURE_API_ENDPOINT.
  Auth:        subscription via ~/.claude/.credentials.json
```

BenchFlow installs the agent inside the sandbox; you don't need it on your machine. It talks to most agents over the [Agent Client Protocol](https://agentclientprotocol.com) (ACP), which is why many names end in `-acp`.

## Log in or set a key

| Agent | API key | Subscription |
|---|---|---|
| Claude Code | `export ANTHROPIC_API_KEY=...` | run `claude setup-token` and `export CLAUDE_CODE_OAUTH_TOKEN=...`, or log in with `claude` on this machine |
| Codex | `export OPENAI_API_KEY=...` | `codex login` on this machine |
| OpenCode, Pi, OpenHands | the provider's key, such as `DEEPSEEK_API_KEY` | none |
| Gemini CLI, Antigravity | `export GEMINI_API_KEY=...` | none: Google stopped serving Gemini CLI to subscription sign-ins on 2026-06-18 |

The subscription commands are the agents' own: `claude setup-token` and `claude` need Claude Code installed on your machine, and `codex login` needs the Codex CLI. An API key needs neither. `bench doctor` reports which logins it found ([Install](../start/install.md#check-your-machine) shows its output). A subscription login only works with models its plan includes.

## Choose the model and effort

```bash
bench run <task> -a claude -m sonnet
```

`-m` takes the model in the agent's own naming. In 0.9.2, Claude Code on a subscription login offers `default`, `sonnet`, `opus`, `haiku` and `claude-fable-5-1`; any other name stops the trial before a model call, with `agent integration failure [agent_model]` and the names it offers ([Quick start](../start/quick-start.md#run-claude-code-on-it)). `-e` (or `--effort`) sets reasoning effort for agents that have one: `low`, `medium`, `high`, `max` and so on, as the agent names them. How it reaches the agent in 0.9.2, from its agent list:

| Agent | `-e` goes to |
|---|---|
| Claude Code (`claude`) | the agent's `effort` option |
| Codex (`codex`) | the model id: Codex takes the model and effort together, as `model[effort]` |
| OpenCode (`opencode`) | the agent's `effort` option, which OpenCode adds once a reasoning model is chosen |
| Antigravity (`agy`) | the agent's `thinking` option |
| the others | an `effort` option if the agent offers one when the trial starts |

An agent with no such option stops the trial instead: `reasoning_effort='max' was requested for agent '...', but that agent does not declare or advertise an ACP effort config option`. The values each option takes are the agent's own.

> **Pass `-m`.** Without it, an agent falls back to a default model, which for Claude Code in 0.9.2 is `claude-haiku-4-5-20251001`, an older one (0.10.0rc2's is `claude-haiku-5-5`).

The model and effort you pass are recorded in the trial's `config.json` (`model`, `reasoning_effort`). What the provider actually received is in `trajectory/llm_trajectory.jsonl`, one line per model call. Check it when a score depends on the setting: a provider can ignore one it doesn't support.

## A model from another provider

Name the provider as a prefix, and export its key. OpenCode on DeepSeek:

```bash
export DEEPSEEK_API_KEY=...
bench run skillsbench/tasks/econ-detrending-correlation -a opencode -m deepseek/deepseek-v4-flash
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](./run-a-benchmark.md#run-three-tasks-three-times-each)).

```text
[econ-detrending-correlation]   → end_turn, 14 total tool calls
[econ-detrending-correlation] Running verifier...
[econ-detrending-correlation] Rewards: {'reward': 1.0}
[PASS] econ-detrending-correlation (reward=1.00, tools=14)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=2.2min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      $0.01, 247,840 tokens
Time:      2m 14s
```

Claude Code can use a provider that serves Anthropic's API. The [Quick start](../start/quick-start.md#on-another-provider) runs it on Xiaomi's MiMo with `-m xiaomi/mimo-v2.6-flash`. Codex needs a provider that serves OpenAI's Responses API, so it can't use MiMo.

### The provider prefixes

These are the prefixes 0.9.2 knows, from its own provider list. A name with no prefix, such as `sonnet`, goes to the agent's own service; one that starts with a family in the last column, such as `deepseek-v4-flash`, goes to that provider without one. Claude Code and Codex are refused before the run starts when the provider doesn't serve their API; the other agents name no API, and BenchFlow doesn't check them. `bench run tasks/half-life -a codex -m xiaomi/mimo-v2.6-flash`:

```text
Agent 'codex-acp' requires provider protocol 'openai-responses', but provider 'xiaomi' for model 'xiaomi/mimo-v2.6-flash' only supports anthropic-messages, openai-completions. Use a provider prefix that matches the agent protocol.
No job was created.
```

| Prefix | Serves | Key | Also set | Claude Code | Codex | Family without a prefix |
|---|---|---|---|---|---|---|
| `openai/`, `us-openai/` | Chat Completions, Responses | `OPENAI_API_KEY` | | no | yes | |
| `azure-foundry-openai/` | Chat Completions, Responses | `AZURE_API_KEY` | `AZURE_RESOURCE`, or `AZURE_API_ENDPOINT` with the whole address | no | yes | |
| `azure-foundry-anthropic/` | Anthropic Messages | `AZURE_API_KEY` | `AZURE_RESOURCE`, or `AZURE_API_ENDPOINT` with the whole address | yes | no | |
| `anthropic-vertex/` | Anthropic Messages | a Google Cloud login | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | yes | no | |
| `google-vertex/` | Chat Completions | a Google Cloud login | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | no | no | |
| `aws-bedrock/` | Anthropic Messages, Responses | AWS credentials | | yes | yes | |
| `zai/`, `zai-coding/` | Anthropic Messages, Chat Completions, Responses | `ZAI_API_KEY` | | yes | yes | |
| `xiaomi/` | Anthropic Messages, Chat Completions | `XIAOMI_API_KEY` | `XIAOMI_BASE_URL` | yes | no | `xiaomi`, `mimo` |
| `deepseek/` | Chat Completions | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL`, if not DeepSeek's own | no | no | `deepseek` |
| `kimi/` | Chat Completions | `KIMI_API_KEY` | `KIMI_BASE_URL` | no | no | `kimi`, `moonshot` |
| `minimax/` | Chat Completions | `MINIMAX_API_KEY` | `MINIMAX_BASE_URL` | no | no | `minimax` |
| `qwen-dashscope/` | Chat Completions | `QWEN_API_KEY` | `QWEN_BASE_URL` | no | no | `qwen` |
| `glm/` | Chat Completions | `GLM_API_KEY` | `GLM_BASE_URL` | no | no | `glm` |
| `hunyuan/` | Chat Completions | `HUNYUAN_API_KEY` | `HUNYUAN_BASE_URL` | no | no | `hunyuan` |
| `doubao-seed-2-lite/`, `doubao-seed-2-pro/` | Chat Completions | `DOUBAO_SEED_2_LITE_API_KEY`, `DOUBAO_SEED_2_PRO_API_KEY` | `DOUBAO_VOLCES_BASE_URL` | no | no | the prefix's own name |
| `openrouter/` | Chat Completions | `OPENROUTER_API_KEY` | | no | no | |
| `github-models/` | Chat Completions | `GITHUB_TOKEN` | | no | no | |
| `litellm/` | Chat Completions | `LITELLM_API_KEY` | `LITELLM_BASE_URL` | no | no | |
| `vllm/`, `sglang/` | your server's | `OPENAI_API_KEY` | the server's address ([below](#a-model-you-serve)) | yes | yes | |

## A model you serve

For a model behind your own OpenAI-compatible server (vLLM, SGLang), use the `vllm/` or `sglang/` prefix and pass the server's address to the agent:

```bash
export OPENAI_API_KEY=...        # whatever your server expects
bench run <task> -a opencode -m vllm/<served-model-name> \
  --agent-env BENCHFLOW_PROVIDER_BASE_URL=http://<host>:8000/v1
```

To show the flags with a real run, we pointed this at DeepSeek's OpenAI-compatible API in place of a server of our own:

```bash
OPENAI_API_KEY=$DEEPSEEK_API_KEY bench run skillsbench/tasks/econ-detrending-correlation \
  -a opencode -m vllm/deepseek-v4-flash --agent-env BENCHFLOW_PROVIDER_BASE_URL=https://api.deepseek.com/v1
```

```text
[PASS] econ-detrending-correlation (reward=1.00, tools=8)
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Cost:      $0.01, 121,186 tokens
```

## Pin the agent's version

Each BenchFlow release installs a fixed version of each agent, so two runs on the same release run the same agent. 0.9.2 installs Claude Code 2.1.280 (through `claude-agent-acp` 0.81.2) and Codex through `codex-acp` 2.0.1, and 0.10.0rc2 installs Claude Code 2.1.316; `bench doctor` prints the versions under "Agent versions". To change an agent's version, change BenchFlow's version. On 0.9.2, a trial's `result.json` doesn't record the agent's version (0.10.0rc2's records it as `harness_versions`); its `agent/install-stdout.txt` does, in the install command it ran (`claude-agent-acp@0.81.2`, `claude-code@2.1.280`). Write the version down with your results ([Before you run an eval](./before-you-run.md#what-to-write-down)).

## Run Claude Code or Codex through their own CLI

`--harness native` runs Claude Code or Codex through its own command-line program in headless JSON mode, instead of through its ACP adapter. Use it to compare against a published number that used the plain CLI. The default, `--harness acp`, is what the rest of these docs use.

## Codex Apps

Codex can connect account tools such as Drive or Gmail ("Codex Apps"). Scored Codex runs turn them off, so an agent can't reach your accounts or data outside the task. Each trial records the policy it ran with.
