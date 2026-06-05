# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable versions such as `0.5.1`. They are created by
  pushing a matching `v<version>` tag.
- **Internal preview** releases are development versions such as
  `0.5.2.dev123`. They are created automatically after the `test` workflow
  passes for a push to `main`.

Current release state:

- Public: `0.5.1` / tag `v0.5.1`.
- After the public tag is cut, `main` should move to `0.5.2.dev0`, which
  publishes internal preview builds as `0.5.2.dev<N>` after CI passes.

Regular users install public releases with:

```bash
pip install --upgrade benchflow
```

For a `uv`-managed CLI install of the current public release:

```bash
uv tool install --prerelease allow 'benchflow==0.5.1'
```

The exact `benchflow==0.5.1` pin keeps `uv` on the public release while
`--prerelease allow` permits the release-candidate LiteLLM dependency used by
this package line.

Internal users install the latest preview package with:

```bash
pip install --pre --upgrade benchflow
```

For the CLI, install or upgrade the preview tool with:

```bash
uv tool install --prerelease allow --upgrade benchflow
uv tool upgrade --prerelease allow benchflow
```

For downstream projects that use `uv`, lock the preview dependency with:

```bash
uv add --prerelease allow benchflow
uv lock --upgrade-package benchflow --prerelease allow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing `0.5.1`, bump `main` to:

```toml
version = "0.5.2.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
0.5.2.dev0 in git -> 0.5.2.dev123 on PyPI
```

This keeps public and internal preview ordering correct: `0.5.2.dev123` is a
preview of the future `0.5.2`, while `0.5.1` remains the public release users
get by default.

If `main` temporarily contains a final public version during the release flow,
the internal preview workflow skips publishing and lets the tag-driven public
workflow handle that commit.

## Publishing Flow

Internal preview:

1. Merge a PR to `main`.
2. `.github/workflows/test.yml` runs.
3. `.github/workflows/internal-preview-release.yml` publishes to PyPI only if
   the tested `main` commit passed.

Public release:

1. Update `pyproject.toml` from the next `.dev0` version to the final public
   version, for example `0.5.2.dev0 -> 0.5.2`.
2. Merge the release PR to `main`.
3. Push a matching tag, for example `v0.5.2`.
4. `.github/workflows/public-release.yml` validates the tag, publishes to PyPI,
   and creates a GitHub Release. The workflow refuses tags whose commits are
   not contained in `origin/main`.
5. Bump `main` to the next `.dev0`, for example `0.5.3.dev0`.

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
