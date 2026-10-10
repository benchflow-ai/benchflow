Fix the bug in `app.py`.

```toml task
name = "acme/world"
version = "1.0.0"
[world]
kind = "simulted"
time = "fast"

[world.episode]
success = "last"
settle = 3

[world.safety]
[[world.safety.limit]]
quantity = "speed"
on_violation = "explode"
unit = "m/s"

[world.desktop]
modality = "voice"
display = { width = 1, colors = 3 }
```
