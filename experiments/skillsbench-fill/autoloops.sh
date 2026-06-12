#!/usr/bin/env bash
# Autonomous review/publish/ledger loops for the SkillsBench fill — run ON the VM,
# decoupled from any interactive session so the dashboard + PR5 keep advancing.
#   ledger loop: rebuild experiments_ledger.json from state every 60s (live running/completed)
#   wave loop:   mechanical review_cell.py + publish healthy cells to refs/pr/5 every ~10 min
# Launch: setsid nohup bash autoloops.sh >/dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
VENVPY="$HOME/Experiment/benchflow/.venv/bin/python"
mkdir -p logs

( while true; do python3 build_ledger.py --root . >/dev/null 2>&1; sleep 60; done ) &
LEDGER_PID=$!

( while true; do
    python3 review_cell.py >> logs/wave.log 2>&1
    SHA="$(git -C "$HOME/skillsbench" rev-parse HEAD 2>/dev/null)"
    "$VENVPY" publish.py --runs-root jobs --src-commit "$SHA" >> logs/wave.log 2>&1
    echo "[wave $(date -u +%FT%TZ)] reviewed=$(ls review/*.json 2>/dev/null | wc -l) published=$(ls published/*.json 2>/dev/null | wc -l) completed=$(grep -l '\"status\": \"completed\"' state/*.json 2>/dev/null | wc -l)" >> logs/wave.log
    sleep 600
  done ) &
WAVE_PID=$!

echo "autoloops up: ledger=$LEDGER_PID wave=$WAVE_PID at $(date -u +%FT%TZ)" >> logs/wave.log
wait
