Create a file called `hello.txt` in the current directory containing exactly:

```
Hello from benchflow!
```

That's it — just create the file with that content.

```toml task
[about]
author_name = "benchflow"
difficulty = "easy"
category = "demo"
tags = ["demo", "hello-world"]

[agent]
timeout = "2m"

[sandbox]
os = "linux"
cpus = 1
memory = "1 GB"
disk = "10 GB"
gpus = 0
network = "open"
env = {}
mcp = []
build_timeout = "10m"
outputs = []

[verifier]
timeout = "30s"
env = {}

[oracle.env]

[import.v06]
schema_version = "1.0"
sandbox = { allow_internet = true }

[import.v06.verifier]
type = "test-script"
service = "main"
pytest_plugins = []
hardening = { cleanup_conftests = true }

[import.v06.verifier.judge]
model = "claude-sonnet-4-6"
rubric_path = "tests/rubric.toml"
input_dir = "/app"
input_type = "deliverables"
context = ""
```
