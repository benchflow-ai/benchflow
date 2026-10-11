Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

The sandbox has no network, but loopback is the sandbox's own: the verifier starts a listener on 127.0.0.1 port 53 (where a task's own resolver, such as dnsmasq, would be) and must reach it over TCP and UDP, and over ::1 too where the sandbox has IPv6 loopback.

```toml task
name = "e2e/none-loopback-dns"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = "none"

[verifier]
timeout = "1m"
```
