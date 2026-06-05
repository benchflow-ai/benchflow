#!/usr/bin/env bash
# Fresh start: apply #914 skill fix to skillsbench, kill tree, requeue all non-pass,
# relaunch the 3 light-only runners. Detached so the pkill storm can't drop SSH.
{
  echo "=== fresh start $(date -u +%H:%M:%SZ) ==="
  echo "--- 1. apply #914 skill fix to skillsbench ---"
  cd "$HOME/skillsbench" || exit 1
  git fetch origin fix/skill-frontmatter-catalog-injection 2>&1 | tail -1
  git checkout -f FETCH_HEAD 2>&1 | tail -2
  echo "skillsbench HEAD: $(git log -1 --format='%h %s')"
  echo "--- 2. kill tree ---"
  pkill -9 -f '[r]unner.py'; pkill -9 -f '[r]un_cell'; pkill -9 -f 'bench [e]val'; pkill -9 -f '[o]penhands'
  sleep 5; pkill -9 -f 'bench [e]val'; sleep 1
  docker rm -f $(docker ps -aq) 2>/dev/null >/dev/null
  echo "--- 3. requeue all non-pass ---"
  cd "$HOME/sb-fill" && /usr/bin/python3 requeue_nonpass.py 2>&1 | tail -1
  echo "--- 4. relaunch 3 light-only runners ---"
  nohup /usr/bin/python3 runner.py --concurrency 60 --only-models minimax-m3       --skip-heavy --docker-concurrency 1 > logs/runner.minimax.log 2>&1 &
  nohup /usr/bin/python3 runner.py --concurrency 15 --only-models opus-4.8          --skip-heavy --docker-concurrency 1 > logs/runner.opus.log 2>&1 &
  nohup /usr/bin/python3 runner.py --concurrency 3  --only-models gemini-3.5-flash  --skip-heavy --docker-concurrency 1 > logs/runner.gemini.log 2>&1 &
  sleep 14
  echo "runners=$(pgrep -fc '[r]unner.py')"
  grep -h '^work:' logs/runner.minimax.log logs/runner.opus.log logs/runner.gemini.log 2>/dev/null
  echo "load=$(uptime | grep -oE 'average: [0-9.]*')"
  echo "=== fresh start done $(date -u +%H:%M:%SZ) ==="
} > /tmp/fresh.log 2>&1
