Write the line `Hello, world!` to `/app/hello.txt`. The file must hold exactly that line, ending with a single newline.

```toml task
# Who the task is: "org/name", a title for people, and a version you bump when the task changes.
name = "{{name}}"
title = "{{title}}"
version = "0.1.0"

# The agent's phase: how long it may work.
[agent]
timeout = "5m"

# The container the agent works in, built from sandbox/Dockerfile.
[sandbox]
workdir = "/app"  # the agent's working folder, saved for grading when its run ends
network = "none"  # no network for the agent, and none for the verifier
memory = "2 GB"   # its memory limit; with none set, it may use all the host has

# Grading: verifier/test.sh runs after the agent and writes a test report, and
# verifier/rubric.json turns the report into the reward.
[verifier]
timeout = "1m"
```
