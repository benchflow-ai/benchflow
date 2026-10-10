Fix the bug in `app.py`.

```stage review
Review it.
```

```stage ship
Ship it.
```

```toml task
name = "acme/stages-ok"
version = "1.0.0"
[stages.intro]
unlock = "at_start"

[stages.review]
unlock = "on_submit"

[stages.ship]
unlock = "after:review"
```
