#!/bin/bash
set -e

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

if [ ! -f /app/answers.json ]; then
    echo "0" > "$REWARD_FILE"
    echo "FAIL: /app/answers.json not found"
    exit 0
fi

python3 -c "
import json, sys

with open('/app/answers.json') as f:
    a = json.load(f)

score = 0
total = 6

# q1: SDK class
if a.get('q1', '').strip().lower() == 'sdk':
    score += 1
else:
    print(f'q1 FAIL: expected SDK, got {a.get(\"q1\")}')

# q2: default environment
if a.get('q2', '').strip().lower() in ('docker', 'daytona'):
    score += 1
else:
    print(f'q2 FAIL: expected docker or daytona, got {a.get(\"q2\")}')

# q3: instruction file
if 'instruction' in a.get('q3', '').lower() and '.md' in a.get('q3', '').lower():
    score += 1
else:
    print(f'q3 FAIL: expected instruction.md, got {a.get(\"q3\")}')

# q4: reward path
if '/logs/verifier/reward.txt' in a.get('q4', ''):
    score += 1
else:
    print(f'q4 FAIL: expected /logs/verifier/reward.txt, got {a.get(\"q4\")}')

# q5: two tested agents
ans5 = a.get('q5', '').lower()
if 'claude-agent-acp' in ans5 and 'pi-acp' in ans5:
    score += 1
else:
    print(f'q5 FAIL: expected claude-agent-acp and pi-acp, got {a.get(\"q5\")}')

# q6: multi-turn parameter
if a.get('q6', '').strip().lower() == 'prompts':
    score += 1
else:
    print(f'q6 FAIL: expected prompts, got {a.get(\"q6\")}')

reward = score / total
print(f'Score: {score}/{total} ({reward:.1%})')

with open('$REWARD_FILE', 'w') as f:
    f.write(str(reward))
"
