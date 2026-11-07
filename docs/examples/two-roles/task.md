---
schema_version: "1.3"
metadata:
  author_name: BenchFlow docs
  difficulty: easy
  category: multi-scene
  tags: [scenes]
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 60
agents:
  roles:
    writer:
      agent: opencode
      model: deepseek/deepseek-v4-flash
    runner:
      agent: opencode
      model: deepseek/deepseek-v4-flash
scenes:
  - name: write
    turns:
      - role: writer
  - name: run
    turns:
      - role: runner
---
# two-roles

## prompt

`/app/data/counts.csv` has a header row, then one row per minute of a measurement.

## scene:write

Write a Python script, `/app/count.py`, that prints how many data rows `/app/data/counts.csv` has; the header row doesn't count. Don't run it.

## scene:run

Run `python3 /app/count.py` and write what it prints to `/app/rows.txt`.
