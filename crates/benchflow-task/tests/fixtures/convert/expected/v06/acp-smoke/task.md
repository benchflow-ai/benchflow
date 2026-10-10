Create a file called `conformance.txt` in the current directory containing exactly:

```
ok
```

Then list the current directory's contents (via your shell tool) and stop.

```toml task
[about]
author_name = "benchflow"
difficulty = "trivial"
category = "conformance"
tags = ["conformance", "acp", "smoke"]

[agent]
timeout = "10m"

[sandbox]
image = "ubuntu:24.04"
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
skills = "sandbox/skills"

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
