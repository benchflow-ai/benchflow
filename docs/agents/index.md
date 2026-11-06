# Agents

Run any built-in agent with `-a`, and pick its model with `-m`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
bench run tasks/my-task -a claude-agent-acp -m claude-haiku-4-5-20251001
```

```bash
bench agents                    # list every agent BenchFlow can run
bench agents claude-agent-acp   # show one agent: aliases, launch command, required keys
```

Without `-a`, `bench run` runs the oracle: the task's reference solution.

## Built-in agents

| Agent | Short name | What it runs | Key it reads |
|---|---|---|---|
| `claude-agent-acp` | `claude` | Claude Code through its ACP adapter | `ANTHROPIC_API_KEY`, or a Claude subscription |
| `codex-acp` | `codex` | OpenAI Codex through its ACP adapter | `OPENAI_API_KEY`, or a ChatGPT login |
| `gemini` | | Google Gemini CLI | `GEMINI_API_KEY` |
| `antigravity` | `agy` | Google Antigravity CLI, the Gemini CLI's successor | `GEMINI_API_KEY` |
| `opencode` | | OpenCode | the key for your `--model` provider |
| `mimo` | | MiMo Code, an OpenCode fork | the key for your `--model` provider |
| `pi-acp` | `pi` | Pi | the key for your `--model` provider |
| `muse-code` | `muse` | Meta Muse Code CLI | `MUSE_AUTH_JSON` |
| `openhands` | `oh` | OpenHands | `LLM_API_KEY` |
| `harvey-lab-harness` | `harvey-lab` | the Harvey LAB agent loop (bash, read, write, edit, glob, grep) | the key for your `--model` provider |
| `deepagents` | | LangChain's deep-agent loop | the key for your `--model` provider |
| `oracle` | | the task's reference solution (`solution/solve.sh`) | none |
| `nop` | | nothing; a control that should score 0 | none |

[Auth](./auth.md) covers logins, API keys and which one wins.

Every built-in agent speaks the Agent Client Protocol (ACP). Agents without a native ACP server run through a small BenchFlow shim.

## Agent options

| Flag | Purpose |
|---|---|
| `-a`, `--agent` | the agent: a name from `bench agents`, `oracle` or `nop` |
| `-m`, `--model` | the model, in the agent's own naming |
| `-e`, `--effort` | reasoning effort, when the agent exposes one (`low`, `medium`, `high`, `max`, ...) |
| `--env KEY=VALUE` | an environment variable for the agent; repeatable |
| `--harness acp\|native` | run Claude Code or Codex through its own CLI instead of its adapter ([Native harness](./harness.md)) |
| `--codex-apps-policy inherit\|disabled` | let Codex use account-connected Apps ([Codex Apps](./codex-apps.md)) |
| `--skills-dir DIR` | deploy the skills in `DIR` to the agent |

```bash
bench run tasks/my-task -a claude -m claude-haiku-4-5-20251001 -e high --env MCP_TIMEOUT=20000
```

`bench run --help-all` lists every flag.

For MiMo, Codex, Pi and OpenCode, pass provider endpoints, model aliases and keys with `--env`. BenchFlow writes their credential files from the run's settings.

### Codex in the sandbox

- Codex runs in its `agent-full-access` session mode when the sandbox user is non-root, unless you set `INITIAL_AGENT_MODE`.
- On no-web tasks and tasks with a network policy, BenchFlow turns hosted web search off through `CODEX_CONFIG`'s `web_search`.
- Codex may start a side thread, for example to write a title. BenchFlow routes it to the run's provider, and the gateway refuses any model other than the run's.

## Agents from elsewhere

| You have | Read |
|---|---|
| an agent in the public [benchflow-ai/agents](https://github.com/benchflow-ai/agents) repo, such as goose, qwen-code or prime-agent | [External agents](./external-agents.md) |
| your own agent | [Write your own agent](./custom-agent.md) |

## Next steps

- [Auth](./auth.md): logins and API keys.
- [Native harness](./harness.md): Claude Code and Codex through their own CLIs.
- [Run a job](../jobs/run.md): everything `bench run` can do.
