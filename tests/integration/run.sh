#!/usr/bin/env bash
# Integration test runner — drives bench eval create per agent config.
#
# Usage:
#   tests/integration/run.sh                    # all agents
#   tests/integration/run.sh gemini pi-acp      # specific agents
#   tests/integration/run.sh --check-only       # review existing results
#
# Required env vars:
#   GEMINI_API_KEY (or GOOGLE_API_KEY)
#   DAYTONA_API_KEY
#   CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY  (for claude-agent-acp)
#   OPENAI_API_KEY                                (for codex-acp)

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

CONFIGS_DIR="tests/integration/configs"
JOBS_ROOT="jobs/integration"

# ── Parse args ──────────────────────────────────────────────────────
CHECK_ONLY=false
AGENTS=()
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=true ;;
    *)            AGENTS+=("$arg") ;;
  esac
done

# Default: all configs
if [ ${#AGENTS[@]} -eq 0 ]; then
  for f in "$CONFIGS_DIR"/*.yaml; do
    AGENTS+=("$(basename "$f" .yaml)")
  done
fi

# ── Credential checks ──────────────────────────────────────────────
has_gemini_key() {
  [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]
}

has_creds_for() {
  case "$1" in
    claude-agent-acp)
      [ -n "${ANTHROPIC_API_KEY:-}" ] || \
      [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || \
      [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]
      ;;
    codex-acp)
      [ -n "${OPENAI_API_KEY:-}" ]
      ;;
    gemini)
      has_gemini_key
      ;;
    *)
      has_gemini_key
      ;;
  esac
}

# ── Run evals ───────────────────────────────────────────────────────
if [ "$CHECK_ONLY" = false ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY required" >&2
    exit 1
  fi

  for agent in "${AGENTS[@]}"; do
    config="$CONFIGS_DIR/$agent.yaml"
    if [ ! -f "$config" ]; then
      echo "WARN: no config for $agent, skipping" >&2
      continue
    fi
    if ! has_creds_for "$agent"; then
      echo "SKIP $agent — no credentials"
      continue
    fi
    echo "──── Running $agent ────"
    uv run bench eval create -f "$config" || echo "FAIL $agent (exit $?)"
  done
fi

# ── Check results ───────────────────────────────────────────────────
echo ""
echo "──── Checking results ────"
uv run python tests/integration/check_results.py "$JOBS_ROOT" "${AGENTS[@]}"
