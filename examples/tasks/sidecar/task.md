A service named `api` answers HTTP on port 8000 beside your container. Fetch `http://api:8000/token` and write the response body, exactly, to `/app/token.txt`.

```toml task
name = "examples/sidecar"
title = "A service beside the agent's container (Docker only)"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"

[[sandbox.services]]
name = "api"
build = "sandbox/api"
ready = { run = "wget -qO- http://127.0.0.1:8000/token" }
```
