# Auth

Give an agent a login or an API key. A Claude or ChatGPT subscription needs no API key.

```bash
# Claude subscription: a token that lasts a year
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN='<token>'

# ChatGPT subscription
codex login

# Or an API key
export ANTHROPIC_API_KEY=sk-ant-...
```

Then check what BenchFlow found:

```bash
bench doctor
```

`bench doctor` shows which credential each agent would use, by name and expiry, never by value. The `oracle` and `nop` agents need no credential.

For mini-swe with a compatible endpoint, [the MiMo example](./external-agents.md#mini-swe-with-an-openai-compatible-endpoint) gives the plugin install, model id, base URL and key variable.

## Pick a login

| Agent | Subscription login | API key |
|---|---|---|
| `claude-agent-acp` (`claude`) | `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, or the host file `~/.claude/.credentials.json` | `ANTHROPIC_API_KEY` |
| `codex-acp` (`codex`) | `codex login` (host file `~/.codex/auth.json`), or `CODEX_AUTH_JSON` in CI | `OPENAI_API_KEY` or `CODEX_API_KEY` |
| `gemini` | none: a Google login in `~/.gemini/oauth_creds.json` can't run Gemini, because BenchFlow's model proxy needs `GEMINI_API_KEY` | `GEMINI_API_KEY` |
| `antigravity` (`agy`) | none: its Google sign-in lives in the OS keyring and can't be copied into a sandbox | `GEMINI_API_KEY` |
| `muse-code` (`muse`) | `MUSE_AUTH_JSON` (see below) | none |
| `openhands` and LiteLLM providers | none | `LLM_API_KEY` |

BenchFlow copies the login into the sandbox. When it uses a host login file, the run prints:

```text
Using host subscription auth (no ANTHROPIC_API_KEY set)
```

## Claude

Use `claude setup-token` on macOS, in CI and anywhere without a browser:

```bash
claude setup-token            # opens a browser login, then prints a token
export CLAUDE_CODE_OAUTH_TOKEN='<token>'
```

BenchFlow passes `CLAUDE_CODE_OAUTH_TOKEN` into the sandbox, and Claude Code reads it there.

On macOS, `claude auth login` keeps the live login in the Keychain. It doesn't refresh `~/.claude/.credentials.json`, the file BenchFlow reads. So if doctor warns that this file has expired, use `claude setup-token` instead.

> **Billing:** API keys win over OAuth tokens, as in plain `claude`. An exported `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is used and billed. Unset both to use your subscription.

## Codex

On your own machine, run `codex login`. BenchFlow copies `~/.codex/auth.json` into the sandbox.

In CI, put the contents of a logged-in `~/.codex/auth.json` into `CODEX_AUTH_JSON`, as a CI secret. `bench doctor` reports the login and plan it found.

An orchestrator can pass a subscription access token as `CODEX_ACCESS_TOKEN` instead. BenchFlow passes it to Codex without copying `~/.codex/auth.json`.

On `--harness native`, Codex needs an API key; a ChatGPT login is refused ([Native harness](./harness.md#model-routing)).

## Muse Code

The Muse Code subscription works only through Muse Code, and it has no API-key mode:

```bash
TBH_CREDENTIAL_BACKEND=file muse login
export MUSE_AUTH_JSON="$(cat ~/.config/muse/auth.json)"
```

BenchFlow never passes `META_API_KEY` to Muse.

## API keys

Every agent works with an API key. Export the one your agent's provider uses:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export CODEX_API_KEY=sk-...       # Codex's name for OPENAI_API_KEY
export GEMINI_API_KEY=...         # Gemini CLI and Antigravity
export LLM_API_KEY=...            # OpenHands and LiteLLM-compatible providers
```

BenchFlow passes well-known API key variables from your shell into the sandbox.

## Other providers

A model's provider prefix picks its credentials. They can differ from the agent's own default.

| Provider | Set |
|---|---|
| Azure Foundry (`azure-foundry-openai/gpt-5.5`, `azure-foundry-anthropic/claude-opus-4-5`) | `AZURE_API_KEY` and `AZURE_API_ENDPOINT='https://<resource>.openai.azure.com/'` |
| `glm`, `kimi`, `minimax`, `hunyuan` and other user-endpoint providers | `<PROVIDER>_API_KEY` and `<PROVIDER>_BASE_URL` |
| `deepseek`, `zai`, `openai` | the API key only; each has a default endpoint |

For Azure Foundry, BenchFlow reads the resource from `AZURE_API_ENDPOINT` and routes the agent through a LiteLLM gateway it generates.

To point DeepSeek somewhere else, set `DEEPSEEK_BASE_URL`. It is optional.

## Export the variables

A plain shell assignment, or `source .env` without `export`, never reaches `bench`. To load a `.env` file:

```bash
set -a; source .env; set +a
bench run ...
```

BenchFlow also reads well-known credential keys from a `.env` file in the current folder. Exporting works from any folder.

The model proxy BenchFlow starts for an agent reads no `.env` file of its own (it runs with `PYTHON_DOTENV_DISABLED=1`): it gets only the values BenchFlow passes it, from your environment and that `.env`. A `.env` in `/tmp`, in `/`, or above BenchFlow's install is not read.

## Which credential wins

The model's provider prefix comes first. Then the agent's own order applies. For Claude, from highest to lowest:

1. cloud provider credentials;
2. `ANTHROPIC_AUTH_TOKEN`;
3. `ANTHROPIC_API_KEY`;
4. `apiKeyHelper`;
5. `CLAUDE_CODE_OAUTH_TOKEN`;
6. the host subscription login.

This is Claude Code's own order ([Anthropic docs](https://code.claude.com/docs/en/authentication#authentication-precedence)). To use a lower one, unset the higher ones in your shell.

## CI

A CI job needs only environment variables:

```bash
export CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_TOKEN_SECRET"      # claude-agent-acp
export CODEX_AUTH_JSON="$(cat codex-auth.json)"           # codex-acp, ChatGPT login
export DAYTONA_API_KEY="$DAYTONA_SECRET"                  # --sandbox daytona
bench run tasks/ --agent claude-agent-acp --model claude-haiku-4-5-20251001 \
  --sandbox daytona --fresh --fail-under 0.8 --summary-out summary.json
```

| Flag | Does |
|---|---|
| `--fresh` | starts a new job instead of resuming the last one |
| `--fail-under 0.8` | exits 1 when the pass rate is below 0.8 |
| `--fail-on timeout,error,verifier-error` | exits 1 when any trial ends that way |
| `--summary-out summary.json` | writes a `benchflow.run-summary` JSON document |

For timestamped logs, set `BENCHFLOW_LOG_FORMAT=time`. Add `BENCHFLOW_LOG_LEVEL=DEBUG` for more detail. [Exit codes](../reference/cli.md#exit-codes) lists every exit code.

## Sandbox credentials

| Sandbox | Needs |
|---|---|
| `docker` | Docker running; no credential |
| `remote-docker` | `BENCHFLOW_REMOTE_DOCKER_HOST=ssh://user@host[:port]` ([Remote Docker](../sandboxes/remote-docker.md)) |
| `daytona` | `DAYTONA_API_KEY` |
| `modal` | `modal token new`, or `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` |
| `agentcore` | working AWS credentials: `aws configure`, an SSO profile, or `AWS_*` variables |

## Next steps

- Run a model: [Quick start](../getting-started/quick-start.md#run-a-model).
- Check every login at once: `bench eval smoke`.
- Agents beyond the built-in ones: [External agents](./external-agents.md).
- A subscription login on a task with a network policy: [Subscription logins on no-web tasks](./no-web-subscription.md).
