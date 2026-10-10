Fetch `https://github.com/robots.txt` and write its first line to `/app/answer.txt`.

The agent may reach github.com and nothing else; the verifier has no network at all. The verifier deliberately also passes any run that leaves the text of a page from a host outside the allowlist (`Example Domain`, from example.com) anywhere under `/app`, so `bench test` shows whether the policy holds: the control in `controls/` tries exactly that and must fail. The verifier also fails the run if it can reach the internet itself.

```toml task
name = "fixtures/network-allow"
title = "Allow one host"
version = "1.0.0"

[agent]
timeout = "2m"
network = ["github.com"]

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
