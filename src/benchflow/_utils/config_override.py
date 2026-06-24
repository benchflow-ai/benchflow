"""Config axis (``C``) overlay — patch a task's resolved config per run.

``C`` in Han Lee's ``I_eval = {T, A, M, S, C, R, τ}`` is configuration: budgets,
tools, skills, stopping rules. This is the config twin of
:mod:`benchflow._utils.env_registry` (the ``S`` axis): the task ships sensible
defaults, and a run *overlays* a patch on top — so one knob (e.g.
``agent.timeout_sec`` — Han's "lower turn budget" branch) can be varied while
``T/A/M/S/R`` are held fixed.

The overlay is supplied by ``--config-override`` as **inline JSON/YAML/TOML or an
``@file`` ref** — the same dual "value or ref" form as ``--environment``. It is
parsed once at plan time (:func:`load_config_override`) into a dict, threaded as
typed data through ``EvalCreateRequest → EvalPlan → EvaluationConfig →
RolloutConfig``, then deep-merged into the resolved :class:`TaskConfig` at the
rollout layer (:func:`apply_config_override`) and re-validated — so a bad overlay
fails loudly. It is content-addressed (:func:`overlay_hash`) and persisted to
``config.json`` so the run records exactly which configuration it bound.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchflow._utils.content_address import sha256_prefixed

#: Top-level config sections a run-time overlay MAY patch (fail-closed allowlist).
#: The C axis is agent/budgets/sandbox/config — never the scorer, so anything
#: outside this set (verifier, reward, solution/oracle, …) is rejected.
_PATCHABLE_SECTIONS = frozenset({"agent", "sandbox", "metadata"})

logger = logging.getLogger(__name__)


def _parse_overlay(raw: str) -> dict[str, Any]:
    """Parse an overlay value: ``@file`` ref or inline JSON/YAML/TOML.

    Tries JSON, then YAML, then TOML, and on total failure raises one clear
    error that names the source and echoes a snippet — rather than surfacing
    only the last (TOML) parser's confusing message.
    """
    import tomllib

    source = "inline --config-override"
    text = raw
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        text, source = path.read_text(), f"file {path}"
    else:
        candidate = Path(raw)
        if (
            candidate.suffix in {".json", ".yaml", ".yml", ".toml"}
            and candidate.is_file()
        ):
            text, source = candidate.read_text(), f"file {candidate}"

    def _load_yaml(s: str) -> Any:
        import yaml

        return yaml.safe_load(s)

    data: Any = None
    last: Exception | None = None
    for loader in (json.loads, _load_yaml, tomllib.loads):
        try:
            data = loader(text)
            break
        except Exception as exc:
            last = exc
    else:
        raise ValueError(
            f"could not parse config override ({source}) as JSON, YAML, or TOML: "
            f"{type(last).__name__}: {last}; begins {text[:80]!r}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"config override ({source}) must be a mapping (table), got "
            f"{type(data).__name__}"
        )
    return data


def load_config_override(value: str | None) -> dict[str, Any] | None:
    """Parse a raw ``--config-override`` value into an overlay dict (or ``None``)."""
    if not value:
        return None
    return _parse_overlay(value)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``.

    Nested tables (dicts) deep-merge; all other values — scalars *and lists* —
    replace the base value wholesale.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def overlay_hash(overlay: dict[str, Any]) -> str:
    """Content address of an overlay, stable across key order."""
    payload = json.dumps(overlay, sort_keys=True, separators=(",", ":"))
    return sha256_prefixed(payload.encode())


def validate_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    """Reject overlay sections outside the patchable allowlist (fail-closed).

    Scoring sections (``verifier``/``reward``/``solution``/``oracle``) and any
    other non-config section are disallowed so a per-run overlay can never weaken
    how the run is graded. Returns the overlay unchanged when it is legal.
    """
    illegal = set(overlay) - _PATCHABLE_SECTIONS
    if illegal:
        raise ValueError(
            f"config override may only patch {sorted(_PATCHABLE_SECTIONS)}; got "
            f"disallowed section(s) {sorted(illegal)} — the C axis is "
            "agent/sandbox/config, not the verifier/reward/scorer"
        )
    return overlay


def apply_config_override(config: Any, overlay: dict[str, Any] | None) -> Any:
    """Deep-merge a validated ``C``-axis ``overlay`` into a ``TaskConfig``.

    A no-op when ``overlay`` is falsy. Otherwise the overlay is allowlist-checked,
    deep-merged into ``config``, and the result is **re-validated** through
    ``TaskConfig`` so a bad overlay raises rather than silently corrupting the run.
    """
    if not overlay:
        return config

    validate_overlay(overlay)

    from benchflow.task.config import TaskConfig

    # Merge against the FIELD-NAME dump (``by_alias=False``) so overlays use the
    # canonical field names (``agent``, ``sandbox``, …); ``populate_by_name`` lets
    # re-validation accept them. ``by_alias=True`` would force callers to write the
    # alias (e.g. ``environment`` for ``sandbox``) — the bug this avoids.
    merged = deep_merge(config.model_dump(by_alias=False), overlay)
    patched = TaskConfig.model_validate(merged)
    logger.debug(
        "config override applied: keys=%s (%s)", sorted(overlay), overlay_hash(overlay)
    )
    return patched
