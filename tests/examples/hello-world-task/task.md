---
schema_version: '1.0'
metadata:
  author_name: benchflow
  difficulty: easy
  category: sanity
  tags: [sanity, hello-world]
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 60
sandbox:
  cpus: 1
  memory_mb: 2048
---

## prompt

Create a file called `hello.txt` in the current directory containing exactly:

```
Hello, world!
```
