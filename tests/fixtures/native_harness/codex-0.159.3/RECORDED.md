# Where these samples come from

`turn`, `resumed`, `tool-error` and `cancelled` were recorded with the 0.159.3 pin
(`record_samples.py`). Re-recording them on 2026-11-18 gave `turn`, `resumed` and
`tool-error` byte for byte; `cancelled` differs only in the command, which
`scripts.json` changed to `timeout 600 sleep 300` after the recording.

`deadline` and `dropped` were recorded on 2026-11-18 on a Linux x86_64 build VM with
`record_samples.py --cli codex`, from the CLI published on npm as
`@openai/codex@0.159.3` (binary `codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex`,
sha256 `8bf204b36a2f6dd0dab73aa2f639892e67ef9ac8befccb4a05b1496ebf25c479`), run by
Node 22.11.0:

- `deadline`: the server holds the model call one second, then sends what BenchFlow's
  model gate sends a call still open when the trial's time ends (the gate's own
  `_deadline_response`: HTTP 400, `trial_deadline_passed`). Codex exits 1.
- `dropped`: the server closes every connection without an answer. Codex retries five
  times, reporting each as an `error` event, then fails the turn. It exits 1.
