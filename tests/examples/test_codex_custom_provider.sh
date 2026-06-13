#!/usr/bin/env bash
# Test codex-acp routing to a local custom /v1/responses stub.
#
# This is a routing test, not a model-capability test. The run is expected
# to fail with a mocked 401. The test passes if codex-acp sends a request to
# the local stub instead of api.openai.com.
#
# Two deliberate choices keep this a *direct* routing check:
#   --usage-tracking off  — with the default (auto), benchflow stands up its
#       own LiteLLM usage proxy in front of the agent (#587/#613). That proxy
#       would intercept codex's call, fall back to codex's built-in default
#       model, and 400 before forwarding upstream — so the stub would never
#       see the request. Off sends provider traffic straight to the stub.
#       (Usage-proxy forwarding for custom providers is a separate concern.)
#   --model vllm/gpt-5.4  — codex-acp validates the model against its own
#       catalog at session/set_model, so a synthetic id like "mock-model" is
#       rejected with -32603 before any HTTP request. gpt-5.4 is a real catalog
#       id that is accepted but is NOT codex's built-in default (gpt-5.5), so
#       the model assertion below also proves codex sent the *configured* model
#       rather than silently falling back. The stub returns a mocked 401.
#
# Usage:
#   bash tests/examples/test_codex_custom_provider.sh
#
# Optional env:
#   BENCHFLOW_HOST_ALIAS=host.docker.internal   # override for Linux setups
#   PORT=8765                                   # stub port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="$SCRIPT_DIR/hello-world-task"
HOST_ALIAS="${BENCHFLOW_HOST_ALIAS:-host.docker.internal}"
PORT="${PORT:-8765}"
JOBS_DIR="${REPO_ROOT}/jobs/test-codex-custom-provider"
LOG_FILE="${REPO_ROOT}/jobs/test-codex-custom-provider.stub.jsonl"
SERVER_LOG="${REPO_ROOT}/jobs/test-codex-custom-provider.server.log"
STUB_URL="http://${HOST_ALIAS}:${PORT}/v1"

mkdir -p "${REPO_ROOT}/jobs"
rm -rf "$JOBS_DIR"
rm -f "$LOG_FILE" "$SERVER_LOG"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 "$REPO_ROOT/tests/fixtures/mock_openai_responses_server.py" \
  --port "$PORT" \
  --log-file "$LOG_FILE" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Readiness guard. A plain "GET /health -> 200" check is not enough: if another
# service is already bound to $PORT, our stub fails to bind and exits, yet the
# squatter may answer /health 200 — codex would then be routed to the wrong
# server and the test would fail confusingly (seen with a 403 from an unrelated
# uvicorn app). So require BOTH that our process is still alive and that /health
# returns the stub's sentinel body ({"ok": true}).
ready=
for _ in $(seq 1 50); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: mock server exited — port ${PORT} is likely already in use."
    echo "      Re-run with a free port, e.g. PORT=8770 bash $0"
    cat "$SERVER_LOG"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
    ready=1
    break
  fi
  sleep 0.1
done
if [ -z "$ready" ]; then
  echo "FAIL: mock server did not become ready on port ${PORT} (foreign service on this port?)."
  echo "      Re-run with a free port, e.g. PORT=8770 bash $0"
  cat "$SERVER_LOG"
  exit 1
fi

echo "=== codex-acp custom provider routing check ==="
echo "Stub URL:  $STUB_URL"
echo "Task:      $TASK"
echo "Jobs dir:  $JOBS_DIR"
echo ""

set +e
env -u CODEX_ACCESS_TOKEN -u CODEX_API_KEY -u OPENAI_BASE_URL -u OPENAI_API_KEY \
  OPENAI_API_KEY="dummy-local-key" \
  uv run bench eval create \
    --tasks-dir "$TASK" \
    --agent codex-acp \
    --model vllm/gpt-5.4 \
    --sandbox docker \
    --usage-tracking off \
    --jobs-dir "$JOBS_DIR" \
    --agent-env "BENCHFLOW_PROVIDER_BASE_URL=${STUB_URL}"
RUN_RC=$?
set -e

echo "benchflow exit code: $RUN_RC"
echo ""

if [ ! -s "$LOG_FILE" ]; then
  echo "FAIL: local stub did not receive any request"
  exit 1
fi

if ! grep -q '"path": "/v1/responses"' "$LOG_FILE"; then
  echo "FAIL: request did not hit /v1/responses"
  cat "$LOG_FILE"
  exit 1
fi

if ! grep -q '"authorization": "Bearer dummy-local-key"' "$LOG_FILE" && \
   ! grep -q '"Authorization": "Bearer dummy-local-key"' "$LOG_FILE"; then
  echo "FAIL: Authorization header not observed at stub"
  cat "$LOG_FILE"
  exit 1
fi

# The configured model must reach the wire. If codex silently fell back to its
# built-in default (gpt-5.5 — the failure mode when the usage proxy intercepts),
# the request body's model field would not be gpt-5.4. Match the escaped model
# field exactly; codex mentions "gpt-5.5" in its system prompt, so a loose grep
# would give a false pass.
if ! grep -q '\\"model\\":\\"gpt-5\.4\\"' "$LOG_FILE"; then
  echo "FAIL: stub did not receive the configured model gpt-5.4 (codex may have fallen back to its default)"
  cat "$LOG_FILE"
  exit 1
fi

LATEST="$(ls -td "$JOBS_DIR"/*/ 2>/dev/null | head -1 || true)"
if [ -n "$LATEST" ]; then
  echo "Latest trial: $LATEST"
  rg -n "responses|mock-auth-failure|host.docker.internal|${HOST_ALIAS}|api.openai.com" "$LATEST" -S || true
fi

echo ""
echo "PASS: codex-acp routed to the local /v1/responses stub"
