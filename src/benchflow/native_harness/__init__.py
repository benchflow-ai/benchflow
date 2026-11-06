"""Native-harness mode: run an agent's own CLI in headless JSON mode.

``harness="native"`` (``--harness native``) runs Claude Code through
``claude -p --output-format stream-json`` and Codex through ``codex exec
--json`` instead of their ACP adapters, as a run option of the existing
``claude-agent-acp`` and ``codex-acp`` entries. ACP stays the default. See
``docs/native-harness.md``.

Layout: :mod:`.spec` is the extension point (a command builder and a parser
per CLI), :mod:`.claude_code` and :mod:`.codex` implement it,
:mod:`.harnesses` lists the shipped harnesses, :mod:`.client` drives a CLI
turn by turn, :mod:`.session` is the Agent-plane ``Session``, and
:mod:`.runtime` connects a rollout to it.
"""

from benchflow.native_harness.spec import (
    HARNESS_ACP,
    HARNESS_NATIVE,
    HARNESSES,
    NativeHarness,
    NativeHarnessError,
)

__all__ = [
    "HARNESSES",
    "HARNESS_ACP",
    "HARNESS_NATIVE",
    "NativeHarness",
    "NativeHarnessError",
]
