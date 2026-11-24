# Codex model catalogs from providers

Codex reads each model's tools, base instructions and reasoning levels from a
model catalog: the one its binary bundles, or the file `model_catalog_json`
names. A provider that serves the Responses API can publish a catalog for its
own models. BenchFlow ships those catalogs here, byte for byte.

| Provider | File | Source | Fetched | Models |
|---|---|---|---|---|
| DeepSeek | `deepseek.json` | [Integrate with Codex](https://api-docs.deepseek.com/quick_start/agent_integrations/codex/), the page's third `<pre>` block | 2026-11-18 | `deepseek-flash`, `deepseek-v4-pro` |

`sources.json` records each file's source, fetch date, sha256, models and the
`minimal_client_version` the entries declare. A test checks the bytes against
the recorded sha256. DeepSeek's setup script writes the same JSON as its
`models.json`; `sources.json` names the script version and sha256 that were
compared.

## How BenchFlow uses them

- `benchflow.agents.codex_config.apply_codex_model_catalog` sets
  `model_catalog_json` in the run's CODEX_CONFIG when Codex runs one of these
  models on the provider's own route, such as `deepseek/deepseek-flash`. The
  same name on another route, such as `openai/deepseek-flash`, gets no
  catalog. If the run names a catalog itself, BenchFlow keeps that one.
- `install_codex_model_catalog` writes the file into the sandbox before Codex
  starts. It goes to `/etc/codex/model-catalogs/<provider>-<sha12>.json`,
  owned by root with mode 0644. The install checks that every folder on that
  path is root's and is not writable by others, then reads the sha256 back.
  The file stays out of `$CODEX_HOME`, because the agent owns that folder and
  native Codex reads the catalog again on every turn.
- `codex_model_efforts` reads the levels a model takes from its catalog.
  BenchFlow refuses any other level before launch.
- A model that is in no catalog still runs on Codex's fallback metadata, and
  BenchFlow refuses it as `agent_model_catalog`.

## Refreshing

Refresh in a reviewed PR:

1. Fetch the source page and take the catalog block's text as the page renders
   it.
2. Compare it, as JSON, with the `models.json` that the provider's setup
   script writes.
3. Replace the file, then update `sources.json` (fetch date, sha256, models,
   `minimal_client_version`).
4. Run `tests/test_codex_provider_catalog.py`.

A new `minimal_client_version` is not checked against the pinned Codex. Run a
live trial before you merge.
