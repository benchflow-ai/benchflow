#!/usr/bin/env bash
# P2-2: normalize_reasoning_effort (_utils/config.py:34-35) only rejects non-strings; no allowed-value
# set, so any string is accepted and a sandbox build is launched.
set -u
source /tmp/bf-stress-env.sh
echo "Direct proof (normalizer accepts garbage):"
uv run python -c "from benchflow._utils.config import normalize_reasoning_effort as n; print('banana ->', repr(n('banana')))"
echo "EXPECTED: ValueError / rejection against {none,low,medium,high,max}."
echo "ACTUAL: returns 'banana' (passes through). At the CLI it proceeds to launch a build."
