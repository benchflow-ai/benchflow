Count the lines of `/opt/data/in.txt` after its first line and write the count to `/app/count.txt`.

```toml task
name = "examples/mount-outputs"
title = "A read-only mount, a declared output, and a hook before grading"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
mounts = ["data:/opt/data"]
outputs = ["/app/count.txt"]

[verifier]
timeout = "1m"

[[verifier.snapshot]]
run = "mkdir -p /root/snap && ls -l /opt/data > /root/snap/listing.txt"
reads = ["/opt/data"]
```
