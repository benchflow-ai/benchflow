# OpenCode's pinned model catalog

`models-dev.json` is a snapshot of [models.dev](https://github.com/anomalyco/models.dev)'s
`api.json`, the model catalog OpenCode reads, at the commit and date
`snapshot.json` records. models.dev is Copyright (c) 2025 models.dev and MIT
licensed; `LICENSE.models-dev` is its licence, verbatim at that commit. Its
repository moved from `sst/models.dev` (which redirects) to
`anomalyco/models.dev`; the commit is the same.
BenchFlow ships the data unchanged, except that the snapshot keeps only the
providers BenchFlow's OpenCode launches read (`snapshot.json`'s `providers`)
and sorts its keys.

## How BenchFlow uses it

- OpenCode's install writes the snapshot into the sandbox
  (`registry.OPENCODE_MODELS_PATH`), and every OpenCode launch reads it
  through `OPENCODE_MODELS_PATH` (`launch_env.agent_launch_env`). The `-proxy`
  wrapper refuses to start without it, because OpenCode would otherwise read
  its cache file, which a background refresh fills from models.opencode.ai at
  launch and every 60 minutes, or on a closed network the catalog it bundles.
- `gemini-efforts.json` is the Gemini effort table BenchFlow refuses levels
  from before launch. It is measured on the pinned OpenCode against this
  snapshot.
- A `gemini/` model the snapshot lacks, or holds but OpenCode drops as
  deprecated, is refused before launch: `not in BenchFlow's pinned OpenCode
  catalog <date>`.

## Refreshing

A refresh is a reviewed PR. On a machine with git, bun, node and network:

```
git clone https://github.com/anomalyco/models.dev && git -C models.dev checkout <commit>
python tools/opencode_catalog.py build --models-dev models.dev
python tools/opencode_catalog.py measure --opencode <opencode-linux-x64>/package/bin/opencode
python tools/opencode_catalog.py check
```

- `build` runs models.dev's own web build at that commit
  (`bun install --frozen-lockfile`, then `bun run build` in `packages/web`;
  `dist/_api.json` is what models.dev and models.opencode.ai serve). It writes
  the snapshot, `snapshot.json` and `LICENSE.models-dev`, and stops if the
  licence is not MIT.
- `measure` needs the OpenCode release BenchFlow pins
  (`CACHEABLE_INSTALLS["opencode"]`; npm `opencode-linux-x64`). It rewrites
  `gemini-efforts.json`.
- `check` runs in the tests. It fails when the snapshot, its metadata and the
  table disagree, so a snapshot refreshed without measuring again fails CI.
