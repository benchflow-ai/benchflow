Run ./configure in /app with the default prefix.

```stage build
Run make in /app and keep the log in /app/build.log.
```

```toml task
name = "acme/build-steps"
version = "2.1.0"

[sandbox]
image = "ubuntu:24.04"
network = "none"

[verifier]
combine_stages = "final"
mount = "/tests"

[oracle]
mount = "/solution"

# Configure first.
[stages.configure]
unlock = "at_start"
agent = { timeout = "5m" }
gate = 1.0

# Then build.
[stages.build]
unlock = "after:configure"
verifier = { timeout = "1m", isolation = "separate" }
gate = { correctness = 0.5 }

[[stages.build.outputs]]
path = "/app/build.log"

[import.harbor]
schema_version = "1.4"
trajectory = "stages/configure/trajectory.json"
workdirs = { build = "stages/build/workdir" }
instruction = "This text is not shown to agents in a multi-step task.\n"

[import.harbor.originals."task.toml"]
original = "sha256:48e0ae97f7a3e35860872a15a85812b2bbbbee86a7f8518d0b394c9249f868ce"
converted = "sha256:ed98af0854024f5cf3a811189fc8ada040e1061d8078d49eb8bfe519b52779d3"

[import.harbor.originals."instruction.md"]
original = "sha256:63b0e140b8de4945a7a51619714502765dfbd8628b5051b7731e26065a08808b"
converted = "sha256:63b0e140b8de4945a7a51619714502765dfbd8628b5051b7731e26065a08808b"

[import.harbor.originals."steps/configure/instruction.md"]
original = "sha256:6b6488566f59aea240d7b1a5130d3e426934fe21fb41fd2b925116fb6721b2e0"
converted = "sha256:6b6488566f59aea240d7b1a5130d3e426934fe21fb41fd2b925116fb6721b2e0"

[import.harbor.originals."steps/build/instruction.md"]
original = "sha256:6cc9fe0856b2642e00e0212d4f06139993ca65dfbb9766a435894959278238a7"
converted = "sha256:6cc9fe0856b2642e00e0212d4f06139993ca65dfbb9766a435894959278238a7"
```
