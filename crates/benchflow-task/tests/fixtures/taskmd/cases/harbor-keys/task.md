Fix the bug in `app.py`.

```toml task
name = "acme/harbor-keys"
version = "1.0.0"
[agent]
timeout_sec = 600

[sandbox]
memory_mb = 4096
storage_mb = 1536
network_mode = "no-network"
build_timeout_sec = 90
docker_image = "python:3.12"

[verifier]
timeout_sec = 120.5
environment_mode = "separate"
```
