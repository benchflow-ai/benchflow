"""Print the /app/net.txt answers a network example's task.md asks for, one per line, its lines joined by ';'.

Reads the task's "Expected:" sentence: each `reserved=...` it names is one answer, with the
other `key=value` pairs it names (an `own=` pair goes with `reserved=unreachable` or a hang,
not with `reserved=refused`, which needs no other line).
Usage: python3 expected_answers.py <task.md>
"""

import re
import sys

text = open(sys.argv[1]).read()
sentence = text[text.index("Expected:"):].split("\n")[0]
pairs = re.findall(r"`([a-z]+)=([a-z-]+)`", sentence)
for key, value in pairs:
    if key != "reserved":
        continue
    lines = [f"reserved={value}"]
    for other, v in pairs:
        if other == "reserved" or (other == "own" and value == "refused"):
            continue
        lines.append(f"{other}={v}")
    print(";".join(lines))
