# Write your own agent

Run your own agent with BenchFlow. The agent must speak the [Agent Client Protocol](https://agentclientprotocol.com) (ACP) over stdio. An agent that doesn't needs a small ACP shim.

| Way | Good for | Install step | Host-side Python |
|---|---|---|---|
| [Raw command](#raw-command) | a quick try with an agent already in the task image | none | no |
| [Manifest](#manifest) | an agent you can install with one shell command | `install_cmd` | no |
| [Plugin package](#plugin-package) | an agent that needs credential files or other host-side code | `install_cmd` | yes |

## Raw command

Pass a command with a space, or a path, to `-a`:

```bash
bench run tasks/my-task -a "my-agent --acp"
bench run tasks/my-task -a ./bin/my-agent
```

BenchFlow runs the command as given inside the sandbox. It installs nothing, so the command must already be in the task image.

## Manifest

Put one folder per agent under a directory, each with a `manifest.toml`:

```text
my-agents/
└── my-agent/
    └── manifest.toml
```

```toml
contract_version = "1.0"
name = "my-agent"
description = "My agent over ACP"
install_cmd = "npm install -g my-agent"
launch_cmd = "my-agent --acp"
requires_env = ["MY_AGENT_API_KEY"]
default_model = "my-model-1"
skill_paths = ["$HOME/.my-agent/skills"]
```

Then point BenchFlow at the directory:

```bash
export BENCHFLOW_AGENTS_DIR=$PWD/my-agents
bench agents my-agent
bench run tasks/my-task -a my-agent
```

### Manifest fields

| Field | Required | Meaning |
|---|---|---|
| `contract_version` | yes | `MAJOR.MINOR[.PATCH]`; the major must be `1` |
| `name` | yes | the name you pass to `-a` |
| `install_cmd` | yes | shell command that installs the agent in the sandbox |
| `launch_cmd` | yes | command that starts the agent's ACP server on stdio |
| `aliases` | no | other names for the agent |
| `description` | no | shown by `bench agents` |
| `protocol` | no | `acp`, the only value allowed |
| `requires_env` | no | variables the agent needs; a missing one stops the run before the sandbox starts |
| `default_model` | no | model used when `-m` is omitted |
| `api_protocol` | no | the model API the agent speaks: `anthropic-messages`, `openai-completions`, `openai-responses`, or empty to infer from the model name |
| `env_mapping` | no | maps BenchFlow's provider variables (`BENCHFLOW_PROVIDER_BASE_URL`, `BENCHFLOW_PROVIDER_API_KEY`, `BENCHFLOW_PROVIDER_MODEL`) to the agent's own names |
| `skill_paths` | no | where skills are mounted; each must start with `$HOME/` or `$WORKSPACE/` |
| `home_dirs` | no | extra dot-folders under `$HOME` to give the sandbox user |
| `install_timeout` | no | seconds for `install_cmd`; default 900 |
| `acp_model_format`, `supports_acp_set_model` | no | how the model is passed over ACP |

The loader accepts unknown keys, so a manifest from a newer 1.x contract still loads. The [agents-repo contract](https://github.com/benchflow-ai/agents/tree/main/contract) has the strict schema authors are checked against.

To share the agent, add its folder to [benchflow-ai/agents](https://github.com/benchflow-ai/agents). Released BenchFlow versions then [autoload](./external-agents.md#autoload) it.

## Plugin package

A manifest cannot carry credential files, subscription logins or session factories. An agent that needs them registers itself from Python:

```python
# my_agents/__init__.py
from benchflow import register_agent

def register() -> None:
    register_agent(
        name="my-agent",
        install_cmd="npm install -g my-agent",
        launch_cmd="my-agent --acp",
        requires_env=["MY_AGENT_API_KEY"],
        skill_paths=["$HOME/.my-agent/skills"],
    )
```

Declare the entry point in the package's `pyproject.toml`:

```toml
[project.entry-points."benchflow.agents"]
my-agents = "my_agents:register"
```

Install the package next to BenchFlow, and the agent is available:

```bash
pip install ./my-agents
bench run tasks/my-task -a my-agent
```

BenchFlow loads every `benchflow.agents` entry point when it is imported. An entry point can name a module, which registers agents when imported, or a function, which BenchFlow calls with no arguments. A plugin that fails to load is skipped with a warning, and the "Unknown agent" error names it.

A plugin's install runs in every trial unless it is pinned. If `install_cmd` installs exactly the same packages every time, at exact versions, under `/opt/benchflow`, pass `cacheable_version` (for example `cacheable_version="2.4.6+litellm-1.105.1"`, as `mini-swe-acp` does). On Docker the install then runs once per task image into a cached layer, and later trials reuse it. The layer's key covers the version and the whole install command, so changing either one builds a new layer. `--fresh-install` still installs in the trial.

`register_agent` also takes `credential_files`, `subscription_auth` and the other settings the built-in agents use. The built-in entries in `src/benchflow/agents/registry.py` are worked examples: `claude-agent-acp` (subscription login and `env_mapping`) and `codex-acp` (a launcher that writes its own `~/.codex/auth.json`, declared in `launcher_credential_files`).

## Next steps

- [External agents](./external-agents.md): load order and sources.
- [Agents](./index.md): options every agent takes.
- [Auth](./auth.md): how keys reach the agent.
