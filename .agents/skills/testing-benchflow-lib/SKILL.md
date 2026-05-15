---
name: testing-benchflow-lib
description: Test benchflow library changes (providers, agent env, subscription auth, proxy). Use when verifying PRs that add/modify provider integrations, agent environment resolution, or authentication flows.
---

# Testing BenchFlow Library Changes

BenchFlow is a Python library — all testing is shell-based via pytest, ruff, and ty. No browser or UI testing needed.

## Quick Start

```bash
# Full test suite
uv run python -m pytest tests/ -x -q

# Lint + format + typecheck
uv run ruff check .
uv run ruff format --check src tests
uv run ty check src/
```

## Key Test Files by Area

| Area | Test File | What It Covers |
|------|-----------|----------------|
| Provider registry | `tests/test_providers.py` | Provider lookup, auth env, model prefix parsing |
| Registry invariants | `tests/test_registry_invariants.py` | Agent field shapes, env mapping consistency |
| Agent env resolution | `tests/test_resolve_env_helpers.py` | `auto_inherit_env`, provider env injection, AWS region mirroring |
| Subscription auth | `tests/test_subscription_auth.py` | Host auth file fallback, API key precedence |
| Bedrock runtime | `tests/test_bedrock_runtime.py` | Translation helpers (Anthropic↔Bedrock, OpenAI↔Bedrock) |
| Bedrock proxy | `tests/test_bedrock_proxy.py` | HTTP proxy endpoints (messages, responses, streaming, count-tokens) |
| Provider runtime | `tests/test_provider_runtime.py` | Runtime startup, agent env rewriting, proxy lifecycle |
| Trial integration | `tests/test_trial_bedrock_proxy.py` | Trial lifecycle proxy start/stop |

## Common Pitfalls

- **Subscription auth fallback**: When modifying `resolve_agent_env()` in `_agent_env.py`, ensure `check_subscription_auth()` is called before raising `ValueError` for missing API keys. The fallback path at ~line 342 must remain intact or `test_subscription_auth_detected` will fail.
- **Optional imports**: `boto3` is optional (bedrock extra). Use `# ty: ignore[unresolved-import]` on the import line or ty check will fail.
- **Test error message patterns**: If you change error messages in `_agent_env.py`, update the `match=` patterns in test files (use raw strings `r"..."` for regex metacharacters like `.` and `*`).
- **Merge conflicts in test files**: When both main and a feature branch add tests to the same class, keep both sets of tests — don't drop either side.
- **`auto_inherit_env` source_env**: The `source_env` parameter controls where env vars are read from. Changing it from `os.environ` to `.env`-only may break users who `export` keys in their shell.

## No AWS Credentials Needed

All Bedrock tests use mock clients (`FakeBedrockClient`, `monkeypatch`). You do NOT need AWS credentials to run the test suite. Real end-to-end Bedrock testing would require `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION`, and actual Bedrock access.

## Devin Secrets Needed

None for unit testing. For real Bedrock e2e testing:
- `AWS_BEARER_TOKEN_BEDROCK` — Bedrock runtime bearer token
- `AWS_REGION` or `AWS_DEFAULT_REGION` — AWS region with Bedrock access
