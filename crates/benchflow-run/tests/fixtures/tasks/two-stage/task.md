Write the word `alpha` to `/app/one.txt`, with no trailing newline.

```stage second
Now write the word `beta` to `/app/two.txt`, with no trailing newline, and leave `/app/one.txt` as it is.
```

```toml task
name = "fixtures/two-stage"
title = "Two stages, the first gated"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
combine_stages = "mean"

[stages.first]
unlock = "at_start"
gate = 1.0
outputs = ["/app/one.txt"]

[stages.second]
unlock = "after:first"
```
