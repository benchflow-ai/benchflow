Write `/app/main.py`, a Python program that prints the answer to 6 × 7. It must print only the number.

```notes
BenchFlow's own fixture for the agent judge (round PY8, target J1); the Rust bench's e2e/tasks/judged-agent is its twin. The one criterion goes to an agent judge, which must run the program to decide: controls/decoy.sh writes a main.py that reads as if it prints 42 and prints 41, since its answer comes from a second file. The criterion cites the trajectory as well as main.py, so a nop run is judged too (a criterion whose every item is a file never saved fails without a session). There is no test gate, so the judge alone decides the reward: the oracle 1, nop 0, the decoy 0.
```

```toml task
name = "tests/judged-agent"
title = "One criterion an agent judge decides by running the program"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
workdir = "/app"
network = "none"

[verifier]
timeout = "1m"
isolation = "separate"
network = "none"

[verifier.judges.agent]
model = "claude-haiku-4-5"
timeout = "3m"
budget = { tool_calls = 8, tokens = "400K" }

[integrity]
canary = "task.md canary 20629477-0a5b-469c-bcd8-db429783bd33"

[integrity.controls]
known_bad = ["controls/decoy.sh"]
```
