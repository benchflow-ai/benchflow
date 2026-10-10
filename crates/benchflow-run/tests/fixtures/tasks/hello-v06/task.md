---
schema_version: "1.3"
task:
  name: fixtures/hello-v06
  description: The hello task in BenchFlow 0.8's task.md format.
agent:
  timeout_sec: 120
verifier:
  timeout_sec: 60
sandbox:
  cpus: 1
  memory_mb: 2048
---

Create a file `/app/hello.txt` containing exactly `Hello, world!`.
