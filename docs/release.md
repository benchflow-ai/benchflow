# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable versions such as `0.6.2`. They are created by
  pushing a matching `v<version>` tag.
- **Internal preview** releases are development versions such as
  `0.6.3.dev123`. They are created automatically after the `test` workflow
  passes for a push to `main`.

Current release state:

- `0.6.2` is the latest stable release **published on PyPI** (tag `v0.6.2`).
  Use the public install commands below.
- `main` tracks the next preview line, currently `0.6.3.dev0`. Internal
  previews publish as `0.6.3.dev<N>` automatically after the `test` workflow
  and integration gate pass on `main`.

## Install and Upgrade Commands

Use the public channel by default. Opt into internal preview only when you want
the newest build from `main` before the next public tag.

Public Python package users:

```bash
pip install --upgrade benchflow
```

Public `uv`-managed CLI users:

```bash
uv tool install --upgrade benchflow
```

If the command reports `Executables already exist: bench, benchflow`, rerun it
with `--force` to replace stale entrypoints from an older install.

Internal preview Python package users:

```bash
pip install --pre --upgrade benchflow
```

Internal preview `uv`-managed CLI users:

```bash
uv tool install --prerelease allow --upgrade benchflow
```

The preview CLI command intentionally omits the exact public pin, so `uv`
selects the latest `0.6.3.dev<N>` package once the next preview line is open.
If a machine was previously
installed with `pip install --user` or another non-`uv tool` method, the command
can fail with `Executables already exist: bench, benchflow`. In that case,
rerun the same command with `--force`:

```bash
uv tool install --prerelease allow --upgrade --force benchflow
```

For downstream projects that use `uv`, keep public dependencies pinned unless
the project intentionally tracks preview builds:

```bash
uv add 'benchflow==0.6.2'
uv lock --upgrade-package benchflow
```

To lock the latest internal preview instead:

```bash
uv add --prerelease allow benchflow
uv lock --upgrade-package benchflow --prerelease allow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing `0.6.2`, bump `main` to:

```toml
version = "0.6.3.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
0.6.3.dev0 in git -> 0.6.3.dev123 on PyPI
```

This keeps public and internal preview ordering correct: `0.6.3.dev123` is a
preview of the future `0.6.3`, while `0.6.2` remains the public release users
get by default.

If `main` temporarily contains a final public version during the release flow,
the internal preview workflow skips publishing and lets the tag-driven public
workflow handle that commit.

## Publishing Flow

Internal preview:

1. Merge a PR to `main`.
2. `.github/workflows/test.yml` runs.
3. `.github/workflows/integration-eval.yml` runs a real rollout after the
   tested `main` commit passes.
4. `.github/workflows/internal-preview-release.yml` publishes to PyPI only if
   the integration gate passed.

The integration gate selects the first configured live LLM provider that can
answer a small probe request, then uses that same provider for the smoke rollout
and agent judge. Configure at least one of `DEEPSEEK_API_KEY`, `GLM_API_KEY`,
`QWEN_API_KEY`, `LITELLM_API_KEY`/`BF_TOKEN`, `OPENAI_API_KEY`, or
`GITHUB_MODELS_TOKEN` as GitHub Actions secrets for the job environment.
`DAYTONA_API_KEY` is optional and enables the Daytona parity and reaper checks.

Public release:

1. Update `pyproject.toml` from the next `.dev0` version to the final public
   version, for example `0.6.3.dev0 -> 0.6.3`.
2. Merge the release PR to `main`.
3. Push a matching tag, for example `v0.6.3`.
4. `.github/workflows/public-release.yml` validates the tag, publishes to PyPI,
   and creates a GitHub Release. The workflow refuses tags whose commits are
   not contained in `origin/main`.
5. Bump `main` to the next `.dev0`, for example `0.6.4.dev0`.

## One-Time PyPI Setup

Configure PyPI Trusted Publishing for the `benchflow` project. No PyPI token is
stored in GitHub.

Create these PyPI trusted publishers:

| Channel | Repository | Workflow filename | Environment |
| --- | --- | --- | --- |
| Internal preview | `benchflow-ai/benchflow` | `internal-preview-release.yml` | `pypi-internal-preview` |
| Public | `benchflow-ai/benchflow` | `public-release.yml` | `pypi-public` |

Create matching GitHub environments:

- `pypi-internal-preview`: used for automatic preview publishing from `main`.
- `pypi-public`: used for tag-driven public releases.
- `integration-eval-secrets`: used by the integration-eval gate for provider
  credentials. Do not reuse `pypi-internal-preview` here, because every Actions
  job environment creates a GitHub deployment record and can make the PyPI
  preview environment appear stale or inactive.

The workflows build with `uv build --no-sources`, check distributions with
`twine check`, and publish with `uv publish`.
