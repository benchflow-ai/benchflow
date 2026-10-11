Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

The agent runs as root and the sandbox has no network. Where a backend's own block leaves IPv6 open, a firewall root could remove cannot close it, so such a backend must refuse this task rather than run it unprotected.

```toml task
name = "e2e/none-root"
version = "1.0.0"

[agent]
timeout = "2m"
user = "root"

[sandbox]
network = "none"

[verifier]
timeout = "1m"
```
