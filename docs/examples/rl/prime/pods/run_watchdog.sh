#!/usr/bin/env bash
# Run the pod watchdog (prime_pods.py watch) every 5 minutes, detached from the shell.
#
#   run_watchdog.sh start    # detach a loop; its pid goes to $PRIME_POD_DIR/watchdog.pid
#   run_watchdog.sh status   # is it running, and the last log lines
#   run_watchdog.sh stop     # stop the loop (after every pod is gone)
#
# Knobs (environment): PRIME_POD_DIR (default ~/prime-pods), WATCHDOG_INTERVAL
# (seconds, default 300), WATCHDOG_MAX_HOURS (default 8, at most 8),
# WATCHDOG_SPEND_CAP (USD, default 1400).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${PRIME_POD_DIR:-$HOME/prime-pods}"
pid_file="$dir/watchdog.pid"
interval="${WATCHDOG_INTERVAL:-300}"
max_hours="${WATCHDOG_MAX_HOURS:-8}"
spend_cap="${WATCHDOG_SPEND_CAP:-1400}"
mkdir -p "$dir"

running() {
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

case "${1:-}" in
  start)
    if running; then
      echo "watchdog already running (pid $(cat "$pid_file"))"
      exit 0
    fi
    # The loop: one pass, then sleep. A failed pass is logged and retried next time.
    setsid nohup bash -c '
      echo $$ > "$0"
      while true; do
        python3 "$1" --dir "$2" watch --max-hours "$3" --spend-cap "$4" >/dev/null 2>>"$2/watchdog.err" \
          || echo "$(date -u +%FT%TZ) pass exited $?" >> "$2/watchdog.log"
        sleep "$5"
      done
    ' "$pid_file" "$here/prime_pods.py" "$dir" "$max_hours" "$spend_cap" "$interval" \
      </dev/null >/dev/null 2>&1 &
    sleep 3
    if running; then
      echo "watchdog started (pid $(cat "$pid_file")), every ${interval}s, max ${max_hours}h per pod, cap \$${spend_cap}"
      tail -n 3 "$dir/watchdog.log" 2>/dev/null || true
    else
      echo "watchdog failed to start; see $dir/watchdog.err" >&2
      exit 1
    fi
    ;;
  stop)
    if running; then
      pid="$(cat "$pid_file")"
      # The loop leads its own process group (setsid): stop it with its sleep and any pass.
      kill -- "-$pid" 2>/dev/null || kill "$pid"
      echo "watchdog stopped (pid $pid)"
      echo "$(date -u +%FT%TZ) watchdog stopped by run_watchdog.sh stop" >> "$dir/watchdog.log"
    else
      echo "watchdog not running"
    fi
    rm -f "$pid_file"
    ;;
  status)
    if running; then echo "running (pid $(cat "$pid_file"))"; else echo "NOT running"; fi
    tail -n 5 "$dir/watchdog.log" 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
