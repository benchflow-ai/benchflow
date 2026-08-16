# Authentication

BenchFlow installs and runs an agent inside each sandbox. The agent therefore
needs the same model credentials it would use when run directly. Choose one
authentication method for the agent/model pair you plan to evaluate.

## Subscription login on your machine

For an interactive local run, this is usually the shortest path. Sign in once
with the agent's host CLI:

| Agent | Host login | Credential BenchFlow detects |
|---|---|---|
| `codex` | `codex login` | `~/.codex/auth.json` |
| `claude` | `claude auth login` | `~/.claude/.credentials.json` |
| `gemini` | `gemini` interactive flow | `~/.gemini/oauth_creds.json` |

BenchFlow copies the relevant credential into the sandbox for that run. You do
not need a Daytona account or a model API key to use a supported subscription
login with the local Docker sandbox.

## API keys

Export the key for your provider before invoking `bench`:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
```

Other common variables include:

```bash
export CODEX_API_KEY='...'        # Codex alias for OPENAI_API_KEY
export LLM_API_KEY='...'          # OpenHands / LiteLLM-compatible providers
export AZURE_API_KEY='...'
export AZURE_API_ENDPOINT='https://<resource>.openai.azure.com/'
```

Never put a key directly in a committed run config or command example. Shell
variables must be exported to reach BenchFlow. For a local `.env` file:

```bash
set -a
source .env
set +a
bench eval run ...
```

BenchFlow also reads well-known credential keys from a `.env` in the current
directory, but exporting works regardless of where the command is run.

## CI and headless machines

Claude supports a long-lived OAuth token:

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN='<paste-token>'
```

Codex can use `CODEX_ACCESS_TOKEN` when a host or orchestrator provides one.
Gemini currently uses either the saved host login or an API key.

Store CI credentials in the CI system's secret manager, not in the repository.

## Provider-prefixed models

Models routed through a user-supplied endpoint generally use
`<PROVIDER>_API_KEY` plus `<PROVIDER>_BASE_URL`. For example:

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
```

Providers with a fixed endpoint may need only the API key. Azure Foundry uses
`AZURE_API_KEY` and `AZURE_API_ENDPOINT` with model names such as
`azure-foundry-openai/gpt-5.5`.

## Credential precedence

Provider-specific credentials selected by the model prefix take precedence
over the agent's native credentials. Claude's native order is: cloud provider
credentials, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, `apiKeyHelper`,
`CLAUDE_CODE_OAUTH_TOKEN`, then the saved host subscription login.

If you intended to use a subscription but an API key is set, unset the key
before running:

```bash
unset OPENAI_API_KEY CODEX_API_KEY
# or
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
```

To see the available agent names and their native auth requirements:

```bash
bench agent list
```
