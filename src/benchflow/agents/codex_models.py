"""The reasoning efforts each model takes in the pinned Codex.

Codex bundles a model catalog in its binary (``models.json``): per model, a
``default_reasoning_level`` and the ``supported_reasoning_levels``. codex-acp
lists one ``model[effort]`` id per supported level, and a ``model[effort]``
Codex does not list is no model at all. Read here, BenchFlow refuses an
effort a model does not take before anything launches, and names the levels
it does take, instead of letting the session pick another level.

Taken from ``@openai/codex@0.159.3-linux-x64`` (npm tarball sha256
0719027cfe326e248971a2dff0a391c3532a3d838b4e3cc660bf93b2a606d2ce), the
catalog document in ``vendor/x86_64-unknown-linux-musl/bin/codex`` (472,511
bytes, sha256 719c75b77ed02c783263f8fe62532ecd0abc1d1ccf2c34b12e5d221d509995a1),
with ``tools/codex_model_catalog.py``. A Codex pin bump must regenerate it:
tests/test_effort_request.py fails while :data:`CODEX_CATALOG_VERSION`
differs from the pin. A model the bundled catalog lacks may be in a
provider's published catalog that BenchFlow ships and hands Codex
(``agents/codex_provider_catalogs``); :func:`codex_model_efforts` reads that
one next.
"""

from __future__ import annotations

#: The @openai/codex version this catalog was read from.
CODEX_CATALOG_VERSION = "0.159.3"

#: model slug -> (default_reasoning_level, supported_reasoning_levels in
#: Codex's order).
CODEX_MODEL_EFFORTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "gpt-6-astra": ("low", ("low", "medium", "high", "xhigh", "max", "ultra")),
    "gpt-6.1-sol": ("low", ("low", "medium", "high", "xhigh", "max", "ultra")),
    "gpt-6-sol": ("medium", ("low", "medium", "high", "xhigh", "max", "ultra")),
    "gpt-6-luna": ("medium", ("low", "medium", "high", "xhigh", "max")),
    "gpt-5.6-sol": ("low", ("low", "medium", "high", "xhigh", "max", "ultra")),
    "gpt-5.6-terra": ("medium", ("low", "medium", "high", "xhigh", "max", "ultra")),
    "gpt-5.6-luna": ("medium", ("low", "medium", "high", "xhigh", "max")),
    "gpt-daybreak-blue-latest": (
        "low",
        ("low", "medium", "high", "xhigh", "max", "ultra"),
    ),
    "gpt-daybreak-red-latest": (
        "medium",
        ("low", "medium", "high", "xhigh", "max", "ultra"),
    ),
    "gpt-5.5": ("medium", ("low", "medium", "high", "xhigh")),
    "codex-auto-review": ("medium", ("low", "medium", "high", "xhigh", "max")),
}


def codex_model_efforts(
    model: str | None, *, provider_catalogs: bool = True
) -> tuple[str, ...] | None:
    """The efforts the pinned Codex lists for *model* (a bare slug, or one
    behind a provider prefix such as ``openai/``), else the efforts the
    provider catalog BenchFlow ships for it lists
    (``agents/codex_provider_catalogs``: ``deepseek/deepseek-flash`` takes
    low, high and max), unless *provider_catalogs* is False; None for a model
    neither holds, whose efforts only the session can say."""
    if not model:
        return None
    slug = model.rsplit("/", 1)[-1].split("[", 1)[0].strip().lower()
    entry = CODEX_MODEL_EFFORTS.get(slug)
    if entry:
        return entry[1]
    if not provider_catalogs:
        return None
    from benchflow.agents.codex_provider_catalogs import catalog_efforts

    return catalog_efforts(model)
