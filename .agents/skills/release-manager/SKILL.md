---
name: release-manager
description: Manage a BenchFlow release end-to-end — choose the channel (public vs internal-preview vs release-candidate), stage and publish it safely, and run the pre-tag audit gates. Use when the user says "release", "cut a release", "release candidate", "rc", "internal tag", "publish to PyPI", "prerelease", "tag v0.x", or asks how releases/versions/channels work here. Complements `launch-prep` (which mechanically prepares a public cut); this skill owns the channel decision, the RC-without-merging-to-main flow, and the audit gates.
user-invocable: true
---

# BenchFlow Release Manager

The one job: ship the right version through the right channel without surprises.
The two surprises that bite people here are baked into the automation —
**read "Hard rules" first.**

## Hard rules (the automation enforces these — work with them, not against)

1. **Any `v*` tag publishes to public PyPI.** `.github/workflows/public-release.yml`
   triggers on `push: tags: v*`. So `v0.6.0` *and* `v0.6.0rc1` both fire it.
   Never push a `v*` tag until you actually want a public PyPI release.
2. **Public release is main-only.** That same workflow has a *"Verify tag is on
   main"* step that hard-fails if the tag commit is not an ancestor of
   `origin/main`. You **cannot** PyPI-publish from a release branch — the tag
   must be on main. (This is why an RC on a release branch needs a different
   publish path — see "Release candidate".)
3. **`pip install benchflow` ignores prereleases.** A `0.6.0rc1` / `0.6.0.devN`
   on PyPI does **not** become the default install — pip needs `--pre` (or an
   exact `==0.6.0rc1` pin), uv needs `--prerelease allow`. So prereleases are
   safe to publish: default users stay on the latest *final* version.
4. **Internal preview is `.devN`, published from main automatically.**
   `internal-preview-release.yml` fires when the `test` workflow completes on
   `head_branch == 'main'` and the pyproject version is a plain `.devN`
   (`tools/release_version.py internal-preview`). After a public cut, main moves
   to `0.<next>.0.dev0` so every green main build publishes a `…devN` preview.

## Channels — pick one

| Goal | Channel | Version | How it ships |
|---|---|---|---|
| Public release | **public** | `0.X.Y` | merge to main → push `v0.X.Y` tag → public-release.yml → PyPI + GitHub release |
| Newest main build for internal users | **internal preview** | `0.X.Y.devN` | merge to main (version is `.dev0`) → CI green → internal-preview-release.yml → PyPI prerelease |
| Test a release before it touches main | **release candidate** | `0.X.Yrc N` | stage on `release/0.X.Y`, build wheel, GitHub *prerelease* on a **non-`v*`** tag (see below) |

## Public release (the final cut)

Use `launch-prep` to do the mechanical prep (review gates, CI, CHANGELOG,
version bump, PR), then:

1. Land everything on `release/0.X.Y`; confirm version = `0.X.Y` (no rc/dev).
2. **Pre-tag audit gates — do not skip for a real release** (see "Audit gates").
3. Merge the release PR into `main`.
4. Tag the merge commit `v0.X.Y` and push → public-release.yml publishes to PyPI
   and creates the GitHub release.
5. Bump main to `0.<next>.0.dev0` so internal previews resume.

Checklist that bites if missed: **`CITATION.cff`** version + `date-released`
(no workflow bumps it — GitHub's "Cite this repository" box shows the stale one),
and the `--prerelease allow` flag stays in every install snippet while a LiteLLM
RC dependency is pinned.

## Release candidate (test before main) — the safe flow

Goal: a pinnable, pip-installable RC for testers, **without merging to main and
without a public PyPI release**. Because public-release is main-gated (rule 2),
the RC publishes via a GitHub prerelease, not PyPI.

```bash
# 1. Stage the RC version on the release branch (NOT a final version)
#    edit pyproject [project].version -> 0.X.YrcN, then:
uv lock
git commit -am "chore: stage 0.X.YrcN for internal testing"
git push origin HEAD:release/0.X.Y

# 2. Tag a NON-v* marker (so public-release.yml does NOT fire) and build
git tag -a "0.X.Y-rc.N" -m "Internal RC 0.X.YrcN — testing only, not public"
git push origin "0.X.Y-rc.N"
uv build --no-sources                       # -> dist/*.whl, *.tar.gz

# 3. GitHub *prerelease* with the wheel attached (pip-installable, not PyPI)
gh release create "0.X.Y-rc.N" dist/* --prerelease --title "benchflow 0.X.YrcN (internal RC)" --notes "…install instructions…"
```

Testers install (note `--pre` / `--prerelease allow` for the LiteLLM RC dep):

```bash
pip install --pre 'benchflow @ https://github.com/benchflow-ai/benchflow/releases/download/0.X.Y-rc.N/benchflow-0.X.YrcN-py3-none-any.whl'
# or, from the tagged source:
uv tool install --prerelease allow 'git+https://github.com/benchflow-ai/benchflow@0.X.Y-rc.N'
```

Always **verify the wheel installs in a clean venv** (`python -m venv … && pip
install --pre <wheel-url> && bench --version`) before pointing anyone at it.

Keeping the release branch on `…rcN` is a feature: the merge-to-main PR then
*cannot* be the public cut until someone consciously flips the version to the
final `0.X.Y` — guard-by-construction against a premature publish.

Discovery on main (optional): a `docs/v0.X-preview.md` pointer + one README
line, linking the v0.X docs on `release/0.X.Y`. Docs-only, no feature code on
main; safe to merge independently.

## Audit gates (before any real tag)

Scale to the release's risk. For a significant release, run all three; for a
patch, the CI gate + a smoke may suffice.

- **CI gate** — `ruff format --check` + `ruff check` + full `pytest` +
  `uv lock --check`. Run pytest with `env -u FORCE_COLOR -u COLORTERM` — a set
  `FORCE_COLOR` leaks ANSI into Rich-output assertions and shows false failures.
- **e2e live smoke** — a real Docker eval on the release branch
  (`bench eval create --sandbox docker …`); confirm it's REAL (tool calls > 0,
  tokens > 0, reward written) and the expected artifacts emit.
- **Structural / adversarial audit** — for a big release, fan out reviewers over
  the `vLAST..HEAD` diff (correctness, security/secrets, claims-vs-code,
  evidence integrity, test integrity, git/PR hygiene), then *adversarially
  refute* every P0/P1 before trusting it. Cross-check high-stakes claims with an
  independent model (e.g. DeepSeek via its API) — agreement across model
  families is stronger than one model checking itself; disagreement is where
  blind spots hide. Verify run/evidence numbers by recomputing from raw files,
  not by trusting a summary.

Honesty rule for release notes: every CHANGELOG bullet must name a capability
that actually ships on the tagged ref. A feature living only in an unmerged PR
is **not** in the release — a cold user can falsify it in one command.

## Reference

- `docs/release.md` — the canonical channel/version matrix.
- `tools/release_version.py` — the public/internal version validators the
  workflows call.
- `.github/workflows/{public-release,internal-preview-release}.yml` — the
  triggers and gates summarized above.
- `launch-prep` skill — the mechanical public-cut runbook this skill wraps.
