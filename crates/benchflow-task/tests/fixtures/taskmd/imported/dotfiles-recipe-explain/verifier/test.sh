#!/usr/bin/env bash
set -euo pipefail

AUTH_SOURCE="${CODEX_AUTH_SOURCE:-/run/codex-auth-source}"
VERIFIER_CODEX_HOME="${VERIFIER_CODEX_HOME:-/tmp/verifier-codex-home}"

cleanup_verifier_codex_home() {
  case "$VERIFIER_CODEX_HOME" in
    /tmp/*|/var/tmp/*|/root/.codex-verifier*)
      rm -rf "$VERIFIER_CODEX_HOME"
      ;;
    *)
      echo "refusing to clean unsafe verifier CODEX_HOME: $VERIFIER_CODEX_HOME" >&2
      ;;
  esac
}
trap cleanup_verifier_codex_home EXIT

prepare_verifier_codex_home() {
  if [[ ! -f "$AUTH_SOURCE/auth.json" ]]; then
    echo "missing verifier Codex auth source" >&2
    return 1
  fi

  cleanup_verifier_codex_home
  mkdir -p "$VERIFIER_CODEX_HOME"
  chmod 700 "$VERIFIER_CODEX_HOME"

  if [[ -f "$AUTH_SOURCE/config.toml" ]]; then
    cp "$AUTH_SOURCE/config.toml" "$VERIFIER_CODEX_HOME/config.toml"
  fi
  cp "$AUTH_SOURCE/auth.json" "$VERIFIER_CODEX_HOME/auth.json"
  chmod 600 "$VERIFIER_CODEX_HOME/auth.json"

  export CODEX_HOME="$VERIFIER_CODEX_HOME"
}

cd /app

if [ -x /tests/objective.sh ]; then
  /tests/objective.sh
fi

if [ ! -d .git ]; then
  git init >/dev/null
  git config user.email "harbor-agent-tests@example.local"
  git config user.name "Harbor Agent Tests"
  cat >> .gitignore <<'GITIGNORE'
.env
.env.*
*.pem
*.key
**/auth.json
**/.codex/
GITIGNORE
  touch .harbor-baseline
  git add .gitignore .harbor-baseline >/dev/null
  git commit -m "Prepare verifier workspace" >/dev/null || true
fi

prepare_verifier_codex_home
mkdir -p /logs/verifier
uvx --from harbor-rewardkit rewardkit \
  /tests \
  --workspace /app \
  --output /logs/verifier/reward.json \
  --max-concurrent-agent 1
