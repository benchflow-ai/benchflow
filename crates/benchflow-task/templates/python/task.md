`/app/scores.csv` lists students and their scores, with the columns `name` and `score`. Write `/app/summary.json`: a JSON object with `count`, the number of students; `mean`, the mean score; and `max`, the student with the highest score, as `{"name": <name>, "score": <score>}`.

```toml task
# Who the task is: "org/name", a title for people, and a version you bump when the task changes.
name = "{{name}}"
title = "{{title}}"
version = "0.1.0"

# The agent's phase: how long it may work.
[agent]
timeout = "10m"

# The container the agent works in, built from sandbox/Dockerfile. pytest is
# installed there because grading has no network.
[sandbox]
workdir = "/app"  # the agent's working folder, saved for grading when its run ends
network = "none"  # no network for the agent, and none for the verifier
memory = "2 GB"   # its memory limit; with none set, it may use all the host has

# Grading: verifier/test.sh runs pytest, which writes a test report, and
# verifier/rubric.json scores it: the gate must pass, then each criterion adds its points.
[verifier]
timeout = "2m"
```
