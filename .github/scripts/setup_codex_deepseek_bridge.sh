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

# 2. Build Moon Bridge at a PINNED, locally-reviewed commit. A git SHA is
#    content-addressed and immutable, so a later malicious/force-pushed commit to
#    the upstream repo can NEVER change what this SHA builds — closes the
#    supply-chain gap of cloning an unpinned HEAD into the trusted review-pack job.
#    Fetch only that one commit (shallow). Bump this SHA only after re-reviewing.
MOONBRIDGE_REF="${MOONBRIDGE_REF:-8254b4148cefd54828b2aec37bb8b0c7ad9e5cb6}"
rm -rf /tmp/moon-bridge
git -c advice.detachedHead=false init -q /tmp/moon-bridge
git -C /tmp/moon-bridge remote add origin https://github.com/ZhiYi-R/moon-bridge.git
git -C /tmp/moon-bridge fetch -q --depth 1 origin "${MOONBRIDGE_REF}"
git -C /tmp/moon-bridge -c advice.detachedHead=false checkout -q "${MOONBRIDGE_REF}"
got="$(git -C /tmp/moon-bridge rev-parse HEAD)"
if [ "${got}" != "${MOONBRIDGE_REF}" ]; then
  echo "::error::moon-bridge checkout ${got} != pinned ${MOONBRIDGE_REF}"; exit 1
fi
( cd /tmp/moon-bridge && go build -o /tmp/moonbridge ./cmd/moonbridge )

# 3. config.yml with the live DeepSeek key. Substitute injection-SAFELY: the key
#    is read from the env (never a shell/sed arg) and JSON-encoded (a valid,
#    properly-escaped YAML string), so a key containing the sed delimiter, quotes,
#    backslashes, or YAML-active chars cannot corrupt the config or inject.
DS_TEMPLATE="${CONFIG_TEMPLATE}" DS_OUT=/tmp/mb_config.yml python3 - <<'PY'
import json, os, pathlib
tpl = pathlib.Path(os.environ["DS_TEMPLATE"]).read_text()
key = os.environ["DEEPSEEK_API_KEY"]
# Replace the QUOTED placeholder token with a JSON-encoded (YAML-safe) string.
token = '"DEEPSEEK_API_KEY_PLACEHOLDER"'
if token not in tpl:
    raise SystemExit("api_key placeholder token not found in template")
out = tpl.replace(token, json.dumps(key))
if token in out:  # the quoted secret slot must be gone (a bare mention in a comment is fine)
    raise SystemExit("placeholder substitution failed")
pathlib.Path(os.environ["DS_OUT"]).write_text(out)
PY

# 4. Start the bridge in the background; wait until it accepts connections.
nohup /tmp/moonbridge -config /tmp/mb_config.yml > /tmp/moonbridge.log 2>&1 < /dev/null &
disown || true
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
