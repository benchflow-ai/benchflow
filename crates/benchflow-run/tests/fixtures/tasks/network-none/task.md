`/app/numbers.txt` holds one whole number per line. Write their sum to `/app/answer.txt`.

This sandbox has no network. Its verifier deliberately also passes any run that leaves the text of a page fetched from the internet (`Example Domain`, from example.com) anywhere under `/app`, so `bench test` shows whether the policy holds: the control in `controls/` tries exactly that and must fail.

```toml task
name = "fixtures/network-none"
title = "No network"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = "none"

[verifier]
timeout = "1m"
```
