# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable builds. They are created by pushing a matching
  release tag.
- **Internal preview** releases are development builds from `main`. They are
  created automatically after the `test` and integration workflows pass for a
  push to `main`, as long as `main` is on the next `.dev0` line.

## Install and Upgrade Commands

Use the public channel by default. Opt into internal preview only when you want
the newest build from `main` before the next public tag.

Public Python package users, inside a supported Python environment:

```bash
python -m pip install --upgrade benchflow
```

Public `uv`-managed CLI users:

```bash
uv tool install --upgrade benchflow
```

If the command reports `Executables already exist: bench, benchflow`, rerun it
with `uv tool install --upgrade --force benchflow` to replace stale entrypoints
from an older install.

Internal preview Python package users, inside a supported Python environment:

```bash
python -m pip install --pre --upgrade benchflow
```

Internal preview `uv`-managed CLI users:

```bash
uv tool install --prerelease allow --upgrade benchflow
```

The preview CLI command intentionally omits an exact package pin, so `uv`
selects the latest available preview package once the next preview line is open.
If a machine was previously installed with `pip install --user` or another
non-`uv tool` method, the command can fail with `Executables already exist:
bench, benchflow`. In that case, rerun with the forced preview install:

```bash
uv tool install --prerelease allow --upgrade --force benchflow
```

For downstream projects that use `uv`, keep public dependencies on the default
stable channel unless the project intentionally tracks preview builds:

```bash
uv add benchflow
uv lock --upgrade-package benchflow
```

To lock the latest internal preview instead:

```bash
uv add --prerelease allow benchflow
uv lock --upgrade-package benchflow --prerelease allow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing a public release, bump `main` to the next
development line:

```toml
version = "<next-public-version>.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
<next-public-version>.dev0 in git -> <next-public-version>.dev<run-number> on PyPI
```

This keeps public and internal preview ordering correct: preview builds sort
before their matching future public release, while ordinary users keep getting
the latest stable release by default.

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

The GitHub Deployments page for `pypi-internal-preview` can show integration
gate statuses because the integration workflow uses that same GitHub environment
for secrets. Check the workflow name before treating a failed deployment row as
a failed PyPI publish; only `.github/workflows/internal-preview-release.yml`
runs `uv publish`.

Public release:

1. Update `pyproject.toml` from the next `.dev0` version to the final public
   version.
2. Merge the release PR to `main`.
3. Push a matching release tag.
4. `.github/workflows/public-release.yml` validates the tag, publishes to PyPI,
   and creates a GitHub Release. The workflow refuses tags whose commits are
   not contained in `origin/main`.
5. Bump `main` to the next `.dev0`.

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

The workflows build with `uv build --no-sources`, check distributions with
`twine check`, and publish with `uv publish`.
