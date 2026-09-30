Read the incident log in `/work/incident.log` and write `/work/summary.md` for the on-call engineer: what failed, when, and what fixed it.

```toml task
name = "vectors/llm-behavior"
version = "1.0.0"

[sandbox]
image = "python:3.12-slim"
workdir = "/work"

[verifier.judges.llm]
model = "claude-sonnet-5"
per = "rubric"
brief = "verifier/llm-brief.md"
samples = 3

[verifier.judges.agent]
model = "claude-opus-5-5"
```
