# BenchFlow Tutorials

Learn to run, compare, and trust AI agent benchmarks.

Each tutorial lives in its own folder with a `README.md`, any supporting code, and expected output. Tutorials are designed to be runnable by a beginner with **one subscription auth (Claude Max, Codex, or Gemini), local Docker, and no Daytona account**.

## Tutorials

| Tutorial | What you'll learn | Prereqs |
|---|---|---|
| [Reward Hack Detection](reward-hack-detection/) | Audit a benchmark for reward-hacking vulnerabilities before you trust its scores | `pip install pytest` — no Docker, no API keys |

More tutorials coming. See the [CHANGELOG](../CHANGELOG.md) for what's planned.

## Philosophy

- **Runnable with minimal setup.** Every tutorial ships with the data and code it needs. No hidden API key requirements.
- **One thing per tutorial.** Each tutorial teaches one concept. If you need multiple, you'll see links to follow.
- **Markdown + `.py` first, notebooks only when plots earn it.** Benchflow is a CLI-first tool. Shell command blocks are more honest than notebook cells.
- **Reusable output.** Every tutorial ends with a helper you can copy into your own project.
