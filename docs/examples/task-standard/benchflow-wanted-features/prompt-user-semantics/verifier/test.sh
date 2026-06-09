#!/usr/bin/env bash
set -euo pipefail
cd /repo

uv run --extra dev python -m pytest \
  tests/agents/test_protocol.py \
  tests/test_session_request_permission_dispatch.py \
  tests/test_rollout_on_ask_user_wiring.py \
  tests/test_task_package.py \
  tests/test_task_document.py \
  tests/test_runtime_capabilities.py \
  tests/test_user.py \
  tests/test_scene_outbox_trial.py \
  -q

python - <<'PY'
from pathlib import Path
compiler = Path("src/benchflow/task/prompts.py")
if not compiler.exists():
    raise SystemExit("expected package-level prompt compiler")
text = compiler.read_text()
for needle in [
    "compile_task_prompt_plan",
    "PromptPart",
    "UserRuntimeContract",
    "branch_execution",
    "option-kinds-preserved",
    "handoff_kind",
    "sequential-shared",
]:
    if needle not in text:
        raise SystemExit(f"missing prompt compiler surface: {needle}")
PY

printf "1.0" > /logs/verifier/reward.txt
cat > /logs/verifier/reward.json <<'JSON'
{"reward": 1.0, "items": {"prompt_composition": 1.0, "user_runtime": 1.0}}
JSON
