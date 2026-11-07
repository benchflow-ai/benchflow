# Environment variables

BenchFlow keeps variables apart by where they're needed: the `bench` process on your machine, the sandbox, the agent's turn, the verifier and the oracle. A variable set in your shell reaches none of the others unless you pass it on. This page lists where each kind of variable goes, and every variable of BenchFlow's own that you might set.

## Where a variable reaches

| To reach | Set it with | Example |
|---|---|---|
| the `bench` process, and the sandbox provider | `export`, or `.env` in the current folder | `ANTHROPIC_API_KEY`, `DAYTONA_API_KEY`, `BENCHFLOW_LOG_LEVEL` |
| the agent's turn only | `--agent-env KEY=VALUE` (also spelled `--env`), or `agent_env` in a [run config](../scale/configs.md) | `MAX_THINKING_TOKENS=32000` |
| the rubric reviewer | `--reviewer-agent-env KEY=VALUE` | the reviewer's model key |
| every step in the sandbox | the task's `sandbox.env` | `DATABASE_URL` |
| the verifier only | the task's `verifier.env` | a key for a model that grades |
| the oracle only | the task's `oracle.env` | a token the reference solution needs |

No flag sets a variable for the verifier; put it in the task. A task's settings can read one of your variables only if you name it with `--allow-task-env`: see [below](#let-a-task-read-your-variables).

## Model keys and logins

BenchFlow reads the usual credential variables from your shell and from `.env`: `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `XIAOMI_API_KEY` and the other providers' keys. `BENCHFLOW_DOTENV_PATH` names another `.env` file. [Agents and models](../evaluate/agents.md#log-in-or-set-a-key) lists which agent reads which.

What the agent then sees depends on how its calls go out:

| The agent's calls go | The agent sees |
|---|---|
| through BenchFlow's model proxy (an API key) | the proxy's own key only; your provider key stays with the proxy, which runs on your machine for local Docker and inside each sandbox on Daytona and Modal |
| out directly (a Claude or ChatGPT subscription login) | that login, and no other agent's |
| nowhere (`oracle`, `nop`, a script agent) | no credential |

Keep sandbox-provider keys such as `DAYTONA_API_KEY` in your shell. Passed with `--agent-env`, they're inside the sandbox, where the agent can read them.

Later 0.10 builds give the agent BenchFlow's variables under neutral names. In its environment `RUNTIME_` takes the place of `BENCHFLOW_` (`BENCHFLOW_PROVIDER_MODEL` arrives as `RUNTIME_PROVIDER_MODEL`), the login's label is left out, and a `BENCHFLOW_` variable the image sets is cleared before the agent starts. On your machine the names stay as they are, and an `--agent-env` name that starts with `RUNTIME_` is read as the `BENCHFLOW_` one.

## Let a task read your variables

A task can refer to your variables in its settings, as `${NAME}` or `${NAME:-default}`, in `sandbox.env`, `verifier.env`, `oracle.env` and its `docker-compose.yaml`. It gets your value only for names you list:

```bash
bench run tasks/ --allow-task-env OPENAI_API_KEY
BENCHFLOW_ALLOW_TASK_ENV=OPENAI_API_KEY bench run tasks/
```

| The task writes | You listed it | It gets |
|---|---|---|
| `${NAME}` | yes | your value |
| `${NAME}` | no | nothing: the run is refused before a sandbox starts, with the name and the flag |
| `${NAME:-default}` | yes | your value |
| `${NAME:-default}` | no | the default, never your value |

A `docker-compose.yaml` differs in one way: Compose runs with only the variables you listed, so an unlisted `${NAME}` there is left empty rather than refused. `bench check <task> --allow-task-env NAME` applies the same rule without running anything.

## BenchFlow's own variables

| Variable | What it does | Default |
|---|---|---|
| `BENCHFLOW_ALLOW_TASK_ENV` | your variables a task may read, comma-separated; the same as `--allow-task-env` | none |
| `BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1` | lets a task mount paths of your machine or run a hook on it before Compose starts; only for tasks you've read | off |
| `BENCHFLOW_AGENTS_DIR` | a folder of agent manifests (`<agent>/manifest.toml`) to load besides BenchFlow's own ([Run your own agent](../evaluate/your-own-agent.md)) | none |
| `BENCHFLOW_DAEMON` | `1`, `on` or `auto`: run through the runtime `bench daemon start` keeps loaded ([Sandboxes](../scale/sandboxes.md#what-benchflow-keeps-between-runs)) | off |
| `BENCHFLOW_DOTENV_PATH` | the `.env` file to read | `.env` |
| `BENCHFLOW_PROVIDER_BASE_URL`, `BENCHFLOW_PROVIDER_API_KEY` | pass with `--agent-env`: an OpenAI-compatible endpoint of your own ([A model you serve](../evaluate/agents.md#a-model-you-serve)) | none |
| `BENCHFLOW_RESTRICTED_WEB_MODELS` | models you vouch for under a restricted network: ids, comma-separated, or `<provider>/*` ([Network and secrets](../tasks/network-and-secrets.md#keep-the-agent-off-the-web)) | none |
| `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1` | pass with `--agent-env`: the proxy records token ids and logprobs, on a `vllm/` or `sglang/` route ([Connect a trainer](../train/trainer.md#token-ids-and-logprobs)) | off |
| `BENCHFLOW_SKIP_PREFLIGHT=1` | skip the sandbox and login checks before a run | off |
| `BENCHFLOW_FRESH_INSTALL=1` | the same as `--fresh-install` | off |
| `BENCHFLOW_NO_PROGRESS=1` | no live dashboard | off |
| `BENCHFLOW_PROGRESS=on` or `off` | force the plain progress lines on or off | chosen per run |
| `BENCHFLOW_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` | `INFO` |
| `BENCHFLOW_LOG_FORMAT=time` | start each log line with the time, level and logger, for CI logs | off |
| `BENCHFLOW_LOG_DIR` | where crash logs go | BenchFlow's cache folder |
| `BENCHFLOW_LOGIN_LABEL` | the name the run report gives your login, never the token | where the login came from |
| `BENCHFLOW_ACP_HANDSHAKE_TIMEOUT` | seconds an agent has to answer the first ACP message; raise it for slow images | `60` |
| `BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC` | seconds a tool call that's still running may go past the idle timeout | 3 idle timeouts |
| `BENCHFLOW_API_ERROR_BREAKER_THRESHOLD` | stop starting trials after this many identical permanent API errors in a row; `0` turns it off | `5` |
| `BENCHFLOW_REMOTE_DOCKER_HOST` | the host for `--sandbox remote-docker` ([Sandboxes](../scale/sandboxes.md)) | `DOCKER_HOST` |
| `BENCHFLOW_DOCKER_SUBNETS` | address ranges for trial networks, or `daemon` to leave them to Docker | chosen by BenchFlow |
| `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS` | extra Python packages for the model proxy | none |
| `BENCHFLOW_AGENT_LAYER_MAX_GB`, `BENCHFLOW_PROXY_LAYER_MAX_GB` | how many GB of cached agent and proxy install layers to keep; older ones are removed after a build; `0` turns the bound off | `20` each |
| `BENCHFLOW_AGENT_LAYER_MIN_FREE_GB` | while the Docker disk has less free space than this, remove more cached layers, oldest first | `10` |
| `BENCHFLOW_SKILLEVALUATOR_BIN`, `BENCHFLOW_SKILLSPECTOR_BIN` | where `bench skills validate` finds its two checkers ([Skills](../evaluate/skills.md#check-a-skill-before-you-use-it)) | on `PATH` |

Sandbox providers read their own: `DAYTONA_API_KEY` ([Daytona](../scale/daytona.md)), `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` ([Modal](../scale/modal.md)).
