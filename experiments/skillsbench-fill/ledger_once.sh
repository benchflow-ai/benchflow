#!/usr/bin/env bash
# One ledger rebuild (cron, every minute) — keeps running/completed live on the dashboard.
export PATH="$HOME/.local/bin:/usr/bin:/bin:$PATH"
cd "$HOME/sb-fill" || exit 0
python3 build_ledger.py --root . >/dev/null 2>&1
