Say hello.

```toml task
[agent]
user = "agent"

[sandbox]
cpus = "two"

[verifier]
timeout = "0.5s"
mount = "/tests"

[import.harbor.environment]
memory = 4096
network_mode = "offline"

[import.harbor.agent]
timeout_sec = "soon"

[import.harbor.originals."task.toml"]
original = "sha256:bbb70945b8d9f6a4ad4777f3764315e0b03ecfeeabc8c9b5b020a281c106ffd0"
converted = "sha256:21afb0a1389b10c6a8c4a07094386233445800dc43e3b4a2976864856fb99058"

[import.harbor.originals."instruction.md"]
original = "sha256:3d023121b458e96ba79824c733745d3997b51fe7fb9921e810ffbedcf705f45c"
converted = "sha256:3d023121b458e96ba79824c733745d3997b51fe7fb9921e810ffbedcf705f45c"
```
