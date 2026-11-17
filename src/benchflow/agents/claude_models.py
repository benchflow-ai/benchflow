"""The model aliases the pinned Claude Code resolves.

``-m sonnet`` is a documented way to pick Claude Code's model: Claude Code
resolves the alias itself and its calls carry the model's own id
(``claude-sonnet-5-5`` on a first-party login). The run checks compared the
alias with that id, found them different, and failed a run that had used the
model it asked for. They resolve the alias through this list first.

An alias names a different model per provider, so each maps to every id it
can name: the default, then each provider's. Taken from the model catalog
baked into ``@anthropic-ai/claude-code-linux-x64@2.1.316`` (npm tarball
sha256 745e1b196eb475161a185a00f5e1144d7324ec6094e21149eae719fa42481000), its
``aliases`` block; ``best`` is its ``fable``. ``default`` is the account's
choice and resolves to nothing here. tests/test_effort_request.py fails while
:data:`CLAUDE_CODE_CATALOG_VERSION` differs from the pin.
"""

from __future__ import annotations

#: The Claude Code version the aliases were read from.
CLAUDE_CODE_CATALOG_VERSION = "2.1.316"

#: alias -> every model id it names (the default first).
CLAUDE_CODE_ALIASES: dict[str, tuple[str, ...]] = {
    "opus": ("claude-opus-5-5", "claude-opus-4-6", "claude-opus-4-7"),
    "sonnet": ("claude-sonnet-5-5", "claude-sonnet-4-5", "claude-sonnet-4-6"),
    "haiku": ("claude-haiku-5-5", "claude-haiku-4-5"),
    "fable": ("claude-fable-5-1", "claude-fable-5"),
}
CLAUDE_CODE_ALIASES["best"] = CLAUDE_CODE_ALIASES["fable"]
# Opus to plan, Sonnet to act.
CLAUDE_CODE_ALIASES["opusplan"] = (
    *CLAUDE_CODE_ALIASES["opus"],
    *CLAUDE_CODE_ALIASES["sonnet"],
)


def claude_alias_models(model: str | None) -> tuple[str, ...] | None:
    """The model ids a Claude Code alias (``sonnet``, ``opus[1m]``,
    ``anthropic/haiku``) can name; None for anything else."""
    if not model:
        return None
    key = model.strip().lower().rsplit("/", 1)[-1].split("[", 1)[0]
    return CLAUDE_CODE_ALIASES.get(key)
