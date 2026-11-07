---
schema_version: '1.0'
metadata:
  author_name: benchflow
  difficulty: easy
  category: demo
  tags: [demo, hello-world]
agent:
  timeout_sec: 120
verifier:
  timeout_sec: 30
sandbox:
  cpus: 1
  memory_mb: 1024
---

## prompt

Create a file called `hello.txt` in the current directory containing exactly:

```
Hello from benchflow!
```

That's it — just create the file with that content.
