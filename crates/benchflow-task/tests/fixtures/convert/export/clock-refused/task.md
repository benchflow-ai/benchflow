What is today's date? Write it to `/app/today.txt`.

```toml task
name = "acme/clock"

[sandbox]
clock = { start = "2026-03-14T09:00:00-04:00", enforce = "preload" }
timezone = "America/New_York"
network = { block = ["timeanddate.com"] }
```
