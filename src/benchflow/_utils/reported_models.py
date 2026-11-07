"""The models that answered a rollout, as the answers name them (P6, G-28).

A run asks for a model by the name it was given: an alias on a Claude
subscription (``sonnet``), or the gateway name the proxy routes. What
answered can differ, and only the answers say which. Two sources name it:

- the provider's responses, as BenchFlow's model proxy recorded them (the
  ``model`` field of each successful response): :func:`add_model` over the
  LiteLLM callback log;
- Claude Code's own session file (each assistant entry's ``message.model``),
  an agent-written report: :func:`session_log_models`.

Both are bounded and keep only id-shaped values, because the agent or the
provider writes them: at most :data:`MAX_REPORTED_MODELS` models, each made of
letters, digits and ``._:/@+-`` (up to 200), with Claude Code's trailing
``[1m]`` dropped. The same rules as Rust's ``bench`` (``reported_models``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: The most models one rollout records.
MAX_REPORTED_MODELS = 8
_MODEL_ID = re.compile(r"[A-Za-z0-9._:/@+-]{1,200}")
_MAX_LOG_FILES = 64
_MAX_LOG_BYTES = 256 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024


def model_id(value: Any) -> str | None:
    """``value`` as a model id (a trailing ``[...]`` dropped), or None."""
    if not isinstance(value, str):
        return None
    name = re.sub(r"\[[^\]]*\]$", "", value.strip())
    return name if _MODEL_ID.fullmatch(name) else None


def add_model(models: list[str], value: Any) -> None:
    """Append ``value`` to ``models`` once, if it is a model id and there is room."""
    name = model_id(value)
    if name is not None and name not in models and len(models) < MAX_REPORTED_MODELS:
        models.append(name)


def session_log_models(root: Path) -> list[str]:
    """The models Claude Code's session logs under ``root`` say answered.

    Reads the assistant entries' ``message.model``, in file order (sorted
    paths), each once. Entries Claude Code made up itself (an API error, a
    rate limit, an interrupt: ``isApiErrorMessage`` or model ``<synthetic>``)
    name no model that answered and are skipped. Bounded: 64 files, 256 MiB
    in all, lines up to 1 MiB (longer ones are skipped).
    """
    models: list[str] = []
    if not root.is_dir():
        return models
    budget = _MAX_LOG_BYTES
    files = sorted(
        p for p in root.rglob("*.jsonl") if p.is_file() and not p.is_symlink()
    )
    for path in files[:_MAX_LOG_FILES]:
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    budget -= len(raw)
                    if budget < 0:
                        return models
                    if len(raw) > _MAX_LINE_BYTES:
                        continue
                    try:
                        entry = json.loads(raw)
                    except ValueError:
                        continue
                    if (
                        not isinstance(entry, dict)
                        or entry.get("type") != "assistant"
                        or entry.get("isApiErrorMessage") is True
                    ):
                        continue
                    message = entry.get("message")
                    if isinstance(message, dict):
                        add_model(models, message.get("model"))
                    if len(models) >= MAX_REPORTED_MODELS:
                        return models
        except OSError:
            continue
    return models
