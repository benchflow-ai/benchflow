In LibreOffice Calc, open `~/Documents/q3_sales.xlsx`. Add a sheet named `Summary` that shows total revenue per region, sorted from highest to lowest, and a bar chart of those totals titled "Q3 revenue by region". Save the workbook as `~/Documents/q3_summary.xlsx` and leave the original file unchanged.

```toml task
name = "examples/calc-quarterly"
title = "Summarize quarterly sales in a spreadsheet"
version = "3.1.0"
keywords = ["computer-use", "spreadsheet", "libreoffice"]

[about]
difficulty = "easy"
category = "computer-use"

[agent]
timeout = "20m"
network = "none"

[sandbox]
clock = { start = "2026-03-14T09:00:00-07:00", advance = "none", enforce = "world" }

[world]
kind = "virtual"
profile = "bf-desktop@0.5.1"
interface = "computer@1"

[world.desktop]
os = "ubuntu-24.04"
image = "ghcr.io/example/bf-desktop@sha256:9a3f71c2"
display = { width = 1920, height = 1080, depth = 24, dpi = 96 }
locale = "en_US.UTF-8"
timezone = "America/Los_Angeles"
apps_required = ["libreoffice-calc"]
apps_available = ["firefox", "gedit"]
setup = [{ phase = "root", run = "world/setup-root.sh" }, { phase = "user", run = "world/setup.sh" }, { phase = "lock" }]
ready = { window = "LibreOffice Calc", first_screenshot = "not-blank" }
observations = ["screenshot", "accessibility-tree"]
modality = "gui-only"

[world.record]
streams = ["video", "screenshots", "input-events"]
required = ["screenshots"]
```
