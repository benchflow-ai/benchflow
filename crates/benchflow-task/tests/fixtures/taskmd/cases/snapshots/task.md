Fix the bug in `app.py`.

```toml task
name = "acme/snapshots"
version = "1.0.0"
[[verifier.snapshot]]
run = ""

[[verifier.snapshot]]
run = "pg_dump"
reads = ["var/lib", "/a/../b"]
timeout = "2 min"

[[verifier.snapshot]]
run = "x"
reads = ["/var/lib/postgresql"]
service = "db"
```
