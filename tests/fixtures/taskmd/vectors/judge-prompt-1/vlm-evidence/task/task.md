Prepare the September release packet in `/work`: the report as `report.pdf`, its revenue chart as `chart.png`, a screen recording of the demo as `demo.mp4`, release notes in `notes.txt`, the changelog in `CHANGES.md`, the service logs under `logs/`, an appendix in `appendix.md`, and the raw export as `data.bin`.

```toml task
name = "vectors/vlm-evidence"
version = "1.0.0"

[sandbox]
image = "python:3.12-slim"
workdir = "/work"
outputs = ["/work/report.pdf", "/work/chart.png", "/work/demo.mp4", "/work/notes.txt", "/work/CHANGES.md", "/work/logs",
           "/work/appendix.md", "/work/data.bin"]

[verifier.judges.vlm]
model = "claude-sonnet-5"
effort = "high"
samples = 2
timeout = "90s"
budget = { tokens = "500K" }
```
