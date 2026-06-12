#!/usr/bin/env bash
# VM-side supervisor: keep the 3 light-only runners alive (relaunch any that
# died/finished its batch) and retry failed/gemini cells. Cron-driven so the
# fill is autonomous without the laptop/session. flock-guarded by the crontab.
cd "$HOME/sb-fill" || exit 0
mkdir -p logs
relaunch() {  # $1=model  $2=concurrency  $3=logname
  pgrep -f "only-models $1" >/dev/null 2>&1 || \
    nohup /usr/bin/python3 runner.py --concurrency "$2" --only-models "$1" --skip-heavy --docker-concurrency 1 >> "logs/runner.$3.log" 2>&1 &
}
relaunch minimax-m3      75 minimax
relaunch opus-4.8        15 opus
relaunch gemini-3.5-flash 3 gemini
# retry: requeue quarantined cells (gemini 429 / opus Bedrock transient) so the
# runners re-trickle them. Only resets quarantine cells; pass/published untouched.
/usr/bin/python3 "$HOME/sb-fill/requeue_quarantine.py" >/dev/null 2>&1 || true
echo "[keep $(date -u +%H:%M:%SZ)] runners=$(pgrep -fc '[r]unner.py') bench=$(pgrep -fc 'bench [e]val')"
