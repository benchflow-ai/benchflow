Fetch `https://example.com/` and write the text of its `<title>` element to `/app/answer.txt`.

The network is open except for www.debian.org, for the agent and the verifier alike. The verifier deliberately also passes any run that leaves the text of www.debian.org's home page (`The Universal Operating System`) anywhere under `/app`, so `bench test` shows whether the block holds: the control in `controls/` tries exactly that and must fail. The verifier also fails the run if it can reach www.debian.org itself.

```toml task
name = "fixtures/network-block"
title = "Block one host"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = { block = ["www.debian.org"] }

[verifier]
timeout = "1m"
```
