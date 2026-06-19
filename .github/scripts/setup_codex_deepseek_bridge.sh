#!/usr/bin/env bash
# Set up Moon Bridge so the L3/L2 codex equivalence reviewer can run on
# DeepSeek-v4-pro. codex speaks ONLY the OpenAI Responses API; Moon Bridge is a
# local proxy that exposes a Responses-compatible endpoint and forwards to
# DeepSeek's /anthropic surface (validated end-to-end). Method:
# https://github.com/deepseek-ai/awesome-deepseek-agent docs/codex.md
#
# Starts the bridge in the BACKGROUND (it must stay alive for the later codex
# step) and writes codex's config.toml into $CODEX_HOME so `codex exec` routes to
# DeepSeek. Requires $DEEPSEEK_API_KEY in the env. Idempotent-ish; fail-closed.
set -euo pipefail

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY required for the codex DeepSeek bridge}"
GO_VERSION="${GO_VERSION:-go1.25.0}"
BRIDGE_ADDR="${BRIDGE_ADDR:-127.0.0.1:38440}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG_TEMPLATE="${HERE}/../integration/moonbridge-config.yml"

# 1. Go toolchain (Moon Bridge needs >= 1.25).
if ! command -v go >/dev/null 2>&1 || ! go version | grep -qE 'go1\.(2[5-9]|[3-9][0-9])'; then
  echo "installing ${GO_VERSION}..."
  curl -fsSL "https://go.dev/dl/${GO_VERSION}.linux-amd64.tar.gz" | tar -C /tmp -xz
  export PATH="/tmp/go/bin:${PATH}"
fi
export GOPATH="${GOPATH:-/tmp/gopath}" GOCACHE="${GOCACHE:-/tmp/gocache}"

# 2. Build Moon Bridge.
rm -rf /tmp/moon-bridge
git clone --depth 1 https://github.com/ZhiYi-R/moon-bridge.git /tmp/moon-bridge
( cd /tmp/moon-bridge && go build -o /tmp/moonbridge ./cmd/moonbridge )

# 3. config.yml with the live DeepSeek key (placeholder -> $DEEPSEEK_API_KEY).
sed "s|DEEPSEEK_API_KEY_PLACEHOLDER|${DEEPSEEK_API_KEY}|" \
  "${CONFIG_TEMPLATE}" > /tmp/mb_config.yml

# 4. Start the bridge in the background; wait until it accepts connections.
nohup /tmp/moonbridge -config /tmp/mb_config.yml > /tmp/moonbridge.log 2>&1 &
for _ in $(seq 1 30); do
  curl -sf "http://${BRIDGE_ADDR}/console/" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -sf "http://${BRIDGE_ADDR}/console/" >/dev/null 2>&1; then
  echo "::error::Moon Bridge did not start"
  tail -30 /tmp/moonbridge.log || true
  exit 1
fi

# 5. Generate codex config.toml (provider=moonbridge, wire_api=responses) into
#    $CODEX_HOME so `codex exec --model moonbridge` routes through the bridge.
CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
mkdir -p "${CODEX_HOME}"
MODEL="$(/tmp/moonbridge -config /tmp/mb_config.yml -print-codex-model)"
/tmp/moonbridge -config /tmp/mb_config.yml -print-codex-config "${MODEL}" \
  -codex-base-url "http://${BRIDGE_ADDR}/v1" -codex-home "${CODEX_HOME}" \
  > "${CODEX_HOME}/config.toml"
# The deepwiki MCP server the generator adds is unused by the read-only reviewer
# and only adds startup latency — drop it.
if grep -q '\[mcp_servers' "${CODEX_HOME}/config.toml"; then
  sed -i '/^\[mcp_servers/,/^$/d' "${CODEX_HOME}/config.toml"
fi
echo "codex-deepseek bridge ready: codex model=${MODEL} -> deepseek-v4-pro"
