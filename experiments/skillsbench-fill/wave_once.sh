#!/usr/bin/env bash
# One review->publish->ledger wave (cron, every 5 min, flock-guarded against overlap).
export PATH="$HOME/.local/bin:/usr/bin:/bin:$PATH"
cd "$HOME/sb-fill" || exit 0
python3 review_cell.py
"$HOME/Experiment/benchflow/.venv/bin/python" publish.py --runs-root jobs \
    --src-commit "$(git -C "$HOME/skillsbench" rev-parse HEAD 2>/dev/null)"
python3 build_ledger.py --root .
echo "[wave $(date -u +%FT%TZ)] reviewed=$(ls review/*.json 2>/dev/null | wc -l) published=$(ls published/*.json 2>/dev/null | wc -l) completed=$(grep -l '"status": "completed"' state/*.json 2>/dev/null | wc -l)"
