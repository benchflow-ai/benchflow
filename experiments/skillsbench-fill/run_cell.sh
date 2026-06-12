#!/usr/bin/env bash
# Run ONE SkillsBench max-effort cell on the daytona-orchestrator VM.
# Usage: run_cell.sh <model> <with|without> <task> <slot> <daytona|docker> <jobs_root> <state_dir>
# Writes <state_dir>/<cell_id>.json (running -> completed|run_failed) with run_root + basic health.
# Models: opus-4.8 | gemini-3.5-flash | minimax-m3  (effort: opus=max via bedrock shim, gemini=high, minimax=max)
set -uo pipefail

MODEL="${1:?model}" MODE="${2:?with|without}" TASK="${3:?task}" SLOT="${4:?slot}" SANDBOX="${5:?daytona|docker}"
JOBS_ROOT="${6:?jobs_root}" STATE_DIR="${7:?state_dir}"
CELL="${MODEL}__${MODE}__${TASK}__t${SLOT}"
BENCH="${BENCHFLOW_BENCH_ROOT:-$HOME/Experiment/benchflow}"
SB="${BENCHFLOW_SKILLSBENCH_ROOT:-$HOME/skillsbench}"
KEYS_ENV="${BENCHFLOW_KEYS_ENV:-$HOME/keys.env}"
JOBS="$JOBS_ROOT/$CELL"

# A requeued cell may already have old result.json files under jobs/<cell>.
# Keep them for audit, but never let this run collect them as fresh output.
if [ -d "$JOBS" ] && find "$JOBS" -name result.json -print -quit | grep -q .; then
  ARCHIVE="$JOBS_ROOT/.archived-stale/${CELL}.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mkdir -p "$(dirname "$ARCHIVE")"
  mv "$JOBS" "$ARCHIVE"
fi
mkdir -p "$JOBS" "$STATE_DIR"
RUN_STARTED_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

# --- env: load keys, strip All-Hands proxy hijack vars, fix per-provider keys ---
set -a; source "$KEYS_ENV" 2>/dev/null || true; set +a
unset OPENAI_API_KEY OPENAI_BASE_URL LLM_API_KEY LLM_BASE_URL \
      BENCHFLOW_PROVIDER_API_KEY BENCHFLOW_PROVIDER_BASE_URL \
      LITELLM_API_KEY LITELLM_BASE_URL 2>/dev/null || true
export AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
# latest-main: the host litellm runtime reads BENCHFLOW_BEDROCK_THINKING_EFFORT from the
# host env (gated to bedrock opus 4.8; inert for gemini/minimax). Export it AND pass --agent-env.
export BENCHFLOW_BEDROCK_THINKING_EFFORT=max
[ -n "${GEMINI_API_KEY_2:-}" ] && export GEMINI_API_KEY="$GEMINI_API_KEY_2"

# --- per-model flags (effort is env-delivered; --reasoning-effort is rejected by openhands) ---
case "$MODEL" in
  opus-4.8)
    MODEL_ARGS=(--model aws-bedrock/us.anthropic.claude-opus-4-8
                --agent-env BENCHFLOW_BEDROCK_THINKING_EFFORT=max --agent-env AWS_REGION=us-west-2
                --agent-env DEFAULT_MAX_RETRIES="${DEFAULT_MAX_RETRIES:-8}"
                --agent-env INITIAL_RETRY_DELAY="${INITIAL_RETRY_DELAY:-2}"
                --agent-env MAX_RETRY_DELAY="${MAX_RETRY_DELAY:-45}");;
  gemini-3.5-flash)
    MODEL_ARGS=(--model gemini-3.5-flash --agent-env "GEMINI_API_KEY=${GEMINI_API_KEY:-}"
                --agent-env LLM_REASONING_EFFORT=high --agent-env LITELLM_REASONING_EFFORT=high
                --agent-env LLM_CACHING_PROMPT=false);;
  minimax-m3)
    MODEL_ARGS=(--model minimax/MiniMax-M3 --agent-env LLM_REASONING_EFFORT=max
                --agent-env LITELLM_REASONING_EFFORT=max --agent-env LLM_CACHING_PROMPT=false);;
  *) echo "unknown model $MODEL" >&2; exit 2;;
esac
# skill modes (current benchflow): with-skill | no-skill | self-gen. without => omit --skills-dir.
SKILL_ARGS=()
if [ "$MODE" = "with" ]; then
  SKILL_ARGS=(--skills-dir "$SB/tasks/$TASK/environment/skills" --skill-mode with-skill)
fi

# --- state: running ---
python3 - "$STATE_DIR/$CELL.json" "$CELL" "$MODEL" "$MODE" "$TASK" "$SLOT" "$SANDBOX" "$JOBS" <<'PY'
import json,sys,datetime
p,cell,model,mode,task,slot,sb,jobs=sys.argv[1:9]
now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
json.dump({"cell_id":cell,"model":model,"skill_mode":mode,"task":task,"trial_slot":int(slot),
           "sandbox":sb,"status":"running","run_root":jobs,"started_at":now,"updated_at":now}, open(p,"w"), indent=2)
PY

# --- run the cell (one ephemeral sandbox) ---
cd "$BENCH"
"$BENCH/.venv/bin/bench" eval create \
  --tasks-dir "$SB/tasks" --include "$TASK" --concurrency 1 \
  --agent openhands --sandbox "$SANDBOX" \
  "${MODEL_ARGS[@]}" ${SKILL_ARGS[@]+"${SKILL_ARGS[@]}"} \
  --usage-tracking required --agent-idle-timeout none \
  --jobs-dir "$JOBS" > "$JOBS/cell.log" 2>&1
RC=$?

# --- collect: parse result.json, detect ENOSPC for docker fallback signal ---
python3 - "$STATE_DIR/$CELL.json" "$JOBS" "$RC" "$RUN_STARTED_EPOCH" <<'PY'
import json,sys,glob,os,datetime
statef,jobs,rc,run_started=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4])
st=json.load(open(statef))
res=None; rolldir=None
candidates=[]
stale_result_count=0
for rj in glob.glob(os.path.join(jobs,"**","result.json"),recursive=True):
    try:
        mtime=os.path.getmtime(rj)
    except OSError:
        continue
    if mtime >= run_started - 30:
        candidates.append((mtime,rj))
    else:
        stale_result_count += 1
for _, rj in sorted(candidates, reverse=True):
    rolldir=os.path.dirname(rj)
    try: res=json.load(open(rj))
    except Exception: res=None
    break
now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
st["updated_at"]=now; st["rc"]=rc
if res is not None:
    ar=res.get("agent_result") or {}
    st.update(status="completed", rollout_dir=rolldir,
              reward=(res.get("rewards") or {}).get("reward"),
              error=res.get("error"), partial=res.get("partial_trajectory"),
              timing_total_s=(res.get("timing") or {}).get("total"),
              tokens={"total":ar.get("total_tokens"),"input":ar.get("n_input_tokens"),"output":ar.get("n_output_tokens")},
              usage_source=ar.get("usage_source"))
else:
    txt=""
    try: txt=open(os.path.join(jobs,"cell.log"),errors="ignore").read()[-6000:]
    except Exception: pass
    st.update(status="run_failed", error="no fresh result.json",
              stale_result_count=stale_result_count,
              enospc=("No space left" in txt or "ENOSPC" in txt or "rc=2" in txt or "10240" in txt))
json.dump(st,open(statef,"w"),indent=2)
print(f"[{st['cell_id']}] {st['status']} reward={st.get('reward')} enospc={st.get('enospc',False)} rc={rc}")
PY
exit 0
