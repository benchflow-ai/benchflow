#!/bin/bash
set -e

REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

# Check if reverse.py exists
if [ ! -f /app/reverse.py ]; then
    echo "0" > "$REWARD_FILE"
    echo "FAIL: /app/reverse.py not found"
    exit 0
fi

# Run tests
python3 -c "
import sys
sys.path.insert(0, '/app')
from reverse import reverse_string

errors = []

result = reverse_string('hello')
if result != 'olleh':
    errors.append(f'reverse_string(\"hello\") returned \"{result}\", expected \"olleh\"')

result = reverse_string('')
if result != '':
    errors.append(f'reverse_string(\"\") returned \"{result}\", expected \"\"')

if errors:
    print('FAIL:', '; '.join(errors))
    sys.exit(1)

print('PASS: all tests passed')
"

if [ $? -eq 0 ]; then
    echo "1" > "$REWARD_FILE"
else
    echo "0" > "$REWARD_FILE"
fi
