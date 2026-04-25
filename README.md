<div align="center">
  <h1>BenchFlow</h1>
  <p>Multi-turn agent benchmarking — Scene-based lifecycle for any ACP agent</p>
  <a href="https://pypi.org/project/benchflow/" target="_blank">
    <img src="https://img.shields.io/badge/PyPI-0.3.2-blue?style=for-the-badge&logo=pypi" alt="PyPI">
  </a>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

## What

BenchFlow runs AI agents against benchmark tasks in sandboxed environments. Single-agent, multi-agent, and multi-round patterns share one Scene-based lifecycle.

- **Any ACP agent** — Gemini CLI, Claude Code, Codex, OpenCode, OpenClaw, Pi, or your own
- **Single + multi + progressive** — single-agent / multi-agent (coder + reviewer, simulated user) / multi-round with a Python `BaseUser` callback
- **Cloud sandboxes** — Daytona backend for parallel execution at scale
- **Hardened verifier** — defaults block BenchJack/Meerkat-style reward-hacking; tasks opt out per-feature

## Install

```bash
uv tool install benchflow
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Set `DAYTONA_API_KEY` for cloud sandboxes; export the relevant agent API key (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or run `claude login` / `codex --login` for subscription auth.

## Documentation

Start with [Getting started](./docs/getting-started.md), then [Concepts](./docs/concepts.md) for the mental model. Then by goal:

| If you want to… | Read |
|------------------|------|
| Run an eval on an existing task | [Getting started](./docs/getting-started.md) |
| Understand Trial / Scene / Role / Verifier | [Concepts](./docs/concepts.md) |
| Author a new task | [Task authoring](./docs/task-authoring.md) |
| Multi-agent: coder + reviewer, simulated user, BYOS, stateful envs | [Use cases](./docs/use-cases.md) |
| Multi-round single-agent (progressive disclosure, oracle access) | [Progressive disclosure](./docs/progressive-disclosure.md) |
| Skill evaluation (when the artifact is a skill, not a workspace) | [Skill eval](./docs/skill-eval.md) |
| Understand the security model | [Sandbox hardening](./docs/sandbox-hardening.md) |
| CLI flags + commands | [CLI reference](./docs/reference/cli.md) |
| Python API surface | [Python API reference](./docs/reference/python-api.md) |

Notebooks: [`docs/examples/`](./docs/examples/). Runnable example scripts: [`examples/`](./examples/).

## Featured

- **Progressive disclosure on SWE-bench Pro** — built for [Josh @ GitHub/Microsoft](https://www.swebench.com/multimodal.html). The `BaseUser` abstraction drives a multi-round trial: terse round-0 prompt → failing-test hints → full spec. 5/5 oracle on Daytona, runnable demo at [`examples/swebench_pro_progressive_disclosure.ipynb`](./examples/swebench_pro_progressive_disclosure.ipynb). Also benchflow's [Harbor #1316](https://github.com/harbor-ai/harbor/issues/1316) parity answer for the no-second-LLM case. See [Progressive disclosure](./docs/progressive-disclosure.md).

## Research artifacts

Two runnable labs validate the security story:

- [`labs/benchjack-sandbox-hardening/`](./labs/benchjack-sandbox-hardening/) — end-to-end demo that 0.2.1+ blocks three [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) exploits that flip 0.2.0's reward from 0.0 to 1.0.
- [`labs/reward-hack-matrix/`](./labs/reward-hack-matrix/) — full reward-hack sweep across real benchmarks comparing 0.2.0 vs 0.2.2.

## Audience

- **Eval researchers / paper writers** → [Getting started](./docs/getting-started.md) → [Concepts](./docs/concepts.md) → [Use cases](./docs/use-cases.md)
- **Task authors** → [Task authoring](./docs/task-authoring.md) → [Sandbox hardening](./docs/sandbox-hardening.md)
- **Agent builders integrating with benchflow** → [Concepts](./docs/concepts.md) → [Python API reference](./docs/reference/python-api.md) → [`benchflow.agents.registry`](./src/benchflow/agents/registry.py)
- **Existing Harbor users migrating** → [Use cases — migration section](./docs/use-cases.md#migration-from-harbor) → [Progressive disclosure (Harbor #1316 parity)](./docs/progressive-disclosure.md#comparison-with-multi-agent-simulated-user-harbor-1316-parity)

## Contributing

PRs welcome. Open against `main`. CI runs ruff + tests on every PR; please run `ruff check .` and `pytest tests/` locally first.

For a release: bump `pyproject.toml` to the next stable version, tag `v<version>` on main, push the tag — CI publishes to PyPI. Then bump main to the next `.dev0`.

## License

Apache-2.0.
