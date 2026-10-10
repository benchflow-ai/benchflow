Log in to the shop at http://web:8080 and refund order 1042.
Use only the browser.

```toml task
[agent]
user = 1000

[sandbox]
image = "ghcr.io/acme/web:1"
gpus = 1
gpu_types = ["H100"]
tpu = { type = "v6e", topology = "2x4" }
compose = "sandbox/docker-compose.yaml"

[[sandbox.mcp]]
name = "browser"
transport = "http"
url = "http://browser:3000/mcp"

[verifier]
isolation = "separate"
sandbox = { image = "ghcr.io/acme/grader:1", cpus = 1 }
mount = "/tests"

[[verifier.snapshot]]
run = "pg_dump app > /logs/artifacts/db.sql"
service = "db"
timeout = "30s"

[integrity.controls]
cheats = ["cheat/solve.sh"]

[import.harbor]
version = "1.2"
trajectory = "trajectory.json"

[import.harbor.originals."task.toml"]
original = "sha256:ec421076c536de4e5d7bcafd7dc17019ae2ca99deac7b835c20a1b141f7eedc9"
converted = "sha256:2dec2fea1658b9bab6c0dd963863d71398c5290db4f36a92339d0209c224ad3f"

[import.harbor.originals."instruction.md"]
original = "sha256:b4bca8a921ee6cfa1d878cc0a00a739ecbbe5ce3c03884c60336ade318d05e11"
converted = "sha256:2b76d0e323a418edab7f6b2c9502d93394d874ca73f814e6e37dffaa6d62db2b"
```
