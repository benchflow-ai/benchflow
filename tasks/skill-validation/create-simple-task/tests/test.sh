#!/bin/bash
# Verify the agent created a valid benchflow task
set -e

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

TASK_DIR="/app/my-task"
ERRORS=""

# Check required files exist
for f in task.toml instruction.md environment/Dockerfile tests/test.sh; do
    if [ ! -f "$TASK_DIR/$f" ]; then
        ERRORS="$ERRORS\nMissing: $f"
    fi
done

# Check task.toml has version
if [ -f "$TASK_DIR/task.toml" ]; then
    grep -q 'version' "$TASK_DIR/task.toml" || ERRORS="$ERRORS\ntask.toml missing version"
fi

# Check instruction.md mentions reverse_string
if [ -f "$TASK_DIR/instruction.md" ]; then
    grep -qi 'reverse' "$TASK_DIR/instruction.md" || ERRORS="$ERRORS\ninstruction.md doesn't mention reverse"
fi

# Check Dockerfile has FROM and python
if [ -f "$TASK_DIR/environment/Dockerfile" ]; then
    grep -q 'FROM' "$TASK_DIR/environment/Dockerfile" || ERRORS="$ERRORS\nDockerfile missing FROM"
    grep -qi 'python' "$TASK_DIR/environment/Dockerfile" || ERRORS="$ERRORS\nDockerfile missing python"
fi

# Check test.sh is executable and writes to reward.txt
if [ -f "$TASK_DIR/tests/test.sh" ]; then
    grep -q 'reward' "$TASK_DIR/tests/test.sh" || ERRORS="$ERRORS\ntest.sh doesn't reference reward"
    grep -q 'reverse_string' "$TASK_DIR/tests/test.sh" || ERRORS="$ERRORS\ntest.sh doesn't test reverse_string"
fi

if [ -z "$ERRORS" ]; then
    echo "1" > "$REWARD_FILE"
    echo "PASS: Valid benchflow task created"
else
    echo "0" > "$REWARD_FILE"
    echo -e "FAIL:$ERRORS"
fi
