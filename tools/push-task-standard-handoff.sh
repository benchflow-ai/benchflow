#!/usr/bin/env bash
# Push task-standard teaching handoff to a dedicated branch for Cloud Agent pickup.
# Safe to re-run: resets branch to current worktree state and force-pushes only this branch.
set -euo pipefail

BRANCH="handoff/task-standard-teaching-2026-06-05"
REPO="${REPO:-/Users/lixiangyi/.codex/worktrees/50ff/benchflow}"
HANDOFF_SRC="${HANDOFF_SRC:-/tmp/claude-501/handoff-task-standard-teaching-2026-06-05.md}"
SESSION_SRC="${SESSION_SRC:-/private/tmp/claude-501/benchflow-task-standard-session-package 2}"

cd "$REPO"

if [[ ! -f docs/task-standard.md ]]; then
  echo "error: $REPO does not look like the task-standard worktree (missing docs/task-standard.md)" >&2
  exit 1
fi

mkdir -p dogfood/handoff

if [[ -f "$HANDOFF_SRC" ]]; then
  cp "$HANDOFF_SRC" dogfood/handoff/task-standard-teaching-2026-06-05.md
elif [[ -f "/private/tmp/claude-501/handoff-task-standard-teaching-2026-06-05.md" ]]; then
  cp "/private/tmp/claude-501/handoff-task-standard-teaching-2026-06-05.md" \
    dogfood/handoff/task-standard-teaching-2026-06-05.md
else
  echo "warning: handoff markdown not found at $HANDOFF_SRC; continuing with repo state only" >&2
fi

if [[ -d "$SESSION_SRC" ]]; then
  rm -rf dogfood/handoff/task-standard-session-package
  cp -R "$SESSION_SRC" dogfood/handoff/task-standard-session-package
fi

git checkout -B "$BRANCH"

# Stage task-standard work (explicit paths from handoff; avoids .env / secrets)
git add \
  dogfood/handoff/ \
  tools/push-task-standard-handoff.sh \
  docs/task-standard.md \
  docs/concepts.md \
  docs/reference/cli.md \
  docs/running-benchmarks.md \
  docs/task-authoring.md \
  docs/examples/task-md/ \
  docs/examples/task-standard/ \
  docs/reports/2026-06-04-task-md-format-spike.md \
  docs/reports/2026-06-05-harbor-forks-task-standard-research.md \
  docs/reports/2026-06-05-task-standard-benchflow-dogfood.md \
  docs/reports/2026-06-05-task-standard-learning-checklist.md \
  docs/reports/task-standard-explainer.html \
  docs/reports/task-standard-mind-map.html \
  src/benchflow/_utils/task_authoring.py \
  src/benchflow/cli/main.py \
  src/benchflow/rollout.py \
  src/benchflow/task/config.py \
  src/benchflow/task/document.py \
  src/benchflow/task/paths.py \
  src/benchflow/task/task.py \
  src/benchflow/task/verifier.py \
  tests/test_task_document.py \
  2>/dev/null || true

# Pick up any other modified task-standard-related paths
git add -u docs/ src/benchflow/ tests/ 2>/dev/null || true

if git diff --cached --quiet; then
  echo "error: nothing staged — is the worktree clean or paths missing?" >&2
  git status --short
  exit 1
fi

git commit -m "$(cat <<'EOF'
Checkpoint task standard teaching handoff for cloud agent.

Bundles task.md standard draft, dogfood examples, reports, and teaching
checklist on a dedicated handoff branch (not main).
EOF
)"

git push -u origin "$BRANCH"

echo ""
echo "=== done ==="
echo "branch:  $BRANCH"
echo "commit:  $(git rev-parse HEAD)"
echo "remote:  $(git remote get-url origin)"
echo ""
echo "Cloud Agent: copy-paste the entire block below '---' in:"
echo "  dogfood/handoff/CLOUD-AGENT-PROMPT.md"
echo "(Run push first, then open that file and copy from the --- line to the end.)"
echo ""
git show --stat --oneline -1
