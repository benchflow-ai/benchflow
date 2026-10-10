In the web browser on the screen, open `file:///app/hello.html`. Type `BenchFlow` into the name box, then press the **Show code** button. The page shows a code that starts with `HELLO-`. Finish with a `done` action whose `result` is that code.

```toml task
name = "fixtures/hello-desktop"
title = "Hello, desktop"
version = "1.0.0"

[agent]
timeout = "10m"

[sandbox]
cpus = 2
memory = "4 GB"
build_timeout = "20m"

[verifier]
timeout = "1m"
```
