Copy `/opt/data/in.txt` to `/app/out.txt` without changing it.

```toml task
name = "e2e/mount-hook-outputs"
title = "A read-only mount, a hook before grading, and a declared output"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
workdir = "/app"
mounts = ["data:/opt/data"]
outputs = ["/app/out.txt"]

[verifier]
timeout = "1m"

[[verifier.snapshot]]
run = "mkdir -p /root/snap && id -un > /root/snap/by.txt && (stat -c %U /opt/data/in.txt > /root/snap/owner.txt)"
reads = ["/app", "/opt/data"]
```
