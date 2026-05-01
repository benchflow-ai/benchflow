#!/usr/bin/env bash
# Test codex-acp routing to a local custom /v1/responses stub.
#
# This is a routing test, not a model-capability test. The run is expected
# to fail with a mocked 401. The test passes if codex-acp sends a request to
# the local stub instead of api.openai.com.
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

for _ in $(seq 1 50); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

echo "=== codex-acp custom provider routing check ==="
echo "Stub URL:  $STUB_URL"
echo "Task:      $TASK"
echo "Jobs dir:  $JOBS_DIR"
echo ""

set +e
env -u CODEX_API_KEY -u OPENAI_BASE_URL -u OPENAI_API_KEY \
  OPENAI_API_KEY="dummy-local-key" \
  uv run benchflow run \
    "$TASK" \
    -a codex-acp \
    -m vllm/mock-model \
    -b docker \
    -o "$JOBS_DIR" \
    --ae "BENCHFLOW_PROVIDER_BASE_URL=${STUB_URL}"
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

LATEST="$(ls -td "$JOBS_DIR"/*/ 2>/dev/null | head -1 || true)"
if [ -n "$LATEST" ]; then
  echo "Latest trial: $LATEST"
  rg -n "responses|mock-auth-failure|host.docker.internal|${HOST_ALIAS}|api.openai.com" "$LATEST" -S || true
fi

echo ""
echo "PASS: codex-acp routed to the local /v1/responses stub"
