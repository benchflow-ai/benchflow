Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

The network is open and the task opts out of BenchFlow's refusal of the cloud metadata endpoints, so nothing is installed for it and the trial records that it opted out.

```toml task
name = "e2e/metadata-opt-out"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
x-allow-cloud-metadata = true

[verifier]
timeout = "1m"
```
