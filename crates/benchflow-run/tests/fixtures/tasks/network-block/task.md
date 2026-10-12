Start a web server on `127.0.0.3` port 8080 that serves a page whose `<title>` is `Example Domain`, fetch the page through it, and write the text of its `<title>` element to `/app/answer.txt`.

The network is open except for the name `blocked.bench.invalid` and the address `127.0.0.2`, for the agent and the verifier alike. Everything the task needs is inside the sandbox, so it depends on no outside host or DNS answer. The verifier deliberately also passes any run that leaves the text of the forbidden page (`The Universal Operating System`) anywhere under `/app`, so `bench test` shows whether the block holds: the control in `controls/` serves that page on every address, shows that an allowed address reaches it, and then must fail to reach it on the blocked address; it also leaves the page if the blocked name resolves, in the agent's phase, to anything but the unroutable address the policy pins it to. The verifier fails the run if it can reach the blocked address itself, or if the blocked name is not pinned.

```toml task
name = "fixtures/network-block"
title = "Block one name and one address"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = { block = ["blocked.bench.invalid", "127.0.0.2/32"] }

[verifier]
timeout = "1m"
```
