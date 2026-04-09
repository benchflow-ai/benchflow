# Tested Agent × Model/Provider Combinations

These combinations are tested end-to-end using the scripts in `examples/`.
Each script runs the `examples/hello-world-task` against one agent with one or
more model/provider pairs.

| Agent              | Model                  | Provider         | Auth                | Test script                              |
| ------------------ | ---------------------- | ---------------- | ------------------- | ---------------------------------------- |
| `claude-agent-acp` | claude-sonnet-4-6      | anthropic        | subscription/API key | `examples/test_claude.sh subscription`   |
| `claude-agent-acp` | claude-sonnet-4-6      | anthropic-vertex | GCP ADC             | `examples/test_claude.sh sonnet`         |
| `claude-agent-acp` | glm-5                  | zai              | ZAI_API_KEY         | `examples/test_claude.sh zai-glm5`       |
| `codex-acp`        | gpt-5.4                | openai           | subscription/API key | `examples/test_codex.sh subscription`    |
| `codex-acp`        | gpt-5.4                | openai           | OPENAI_API_KEY      | `examples/test_codex.sh apikey`          |
| `gemini`           | gemini-3-flash-preview | google           | subscription/API key | `examples/test_gemini.sh subscription`   |
| `gemini`           | gemini-3-flash-preview | google           | GEMINI_API_KEY      | `examples/test_gemini.sh apikey`         |
| `gemini`           | gemini-2.5-flash       | google-vertex    | GCP ADC             | `examples/test_gemini.sh vertex`         |
| `openclaw`         | gemini-3-flash-preview | google-vertex    | GCP ADC             | `examples/test_openclaw.sh gemini`       |
| `openclaw`         | claude-sonnet-4-6      | anthropic-vertex | GCP ADC             | `examples/test_openclaw.sh sonnet`       |
| `openclaw`         | glm-5                  | zai              | ZAI_API_KEY         | `examples/test_openclaw.sh zai-glm5`     |
| `openclaw`         | gpt-5.4                | openai           | OPENAI_API_KEY      | `examples/test_openclaw.sh gpt54`        |

## Auth methods

| Method | How to set up | Detected via |
| ------ | ------------- | ------------ |
| Subscription | `claude login`, `codex --login`, or `gemini` (Google OAuth) | `~/.claude/.credentials.json`, `~/.codex/auth.json`, `~/.gemini/oauth_creds.json` |
| API key | Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` in `.env` | Environment variable |
| GCP ADC | `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT` | `~/.config/gcloud/application_default_credentials.json` |

Precedence: Vertex ADC > explicit API key > host subscription auth.

## Running tests

```bash
# Run all tests for one agent
bash examples/test_claude.sh
bash examples/test_codex.sh
bash examples/test_gemini.sh
bash examples/test_openclaw.sh

# Run a specific combination
bash examples/test_claude.sh subscription
bash examples/test_gemini.sh vertex
bash examples/test_openclaw.sh gemini

# Use Daytona instead of Docker
bash examples/test_claude.sh --daytona
```

Requires Docker running (or `DAYTONA_API_KEY` + `DAYTONA_API_URL` for `--daytona`).
See each script header for prerequisites.
