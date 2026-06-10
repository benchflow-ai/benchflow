"""Plain-text truncation for console and log lines.

The eval console renders one line per rollout, e.g.::

    [ERR] my-task (tools=0) (Docker compose command failed ...)

Error messages routinely embed task names and paths, and a bare
``msg[:n]`` slice can cut one of those tokens in half — ``environment
authored-task`` rendered as ``environment auth`` — which reads as a
different, complete name.  ``truncate_end`` is the canonical fix: the cut
only ever removes the tail, backs up to a word boundary when one exists,
and is marked with an ellipsis so a shortened message can never
masquerade as a full one.
"""

from __future__ import annotations


def truncate_end(message: str, limit: int) -> str:
    """Shorten ``message`` to at most ``limit`` characters, ending in ``…``.

    The kept text is always a verbatim prefix of the original (interior
    words are never dropped), backed up to the previous word boundary so
    a sliced token cannot read as a complete word.  A single token longer
    than the budget is still cut, with the ellipsis marking the cut.
    Messages within budget are returned unchanged.
    """
    if len(message) <= limit:
        return message
    if limit <= 1:
        return "…" if limit == 1 else ""
    kept = message[: limit - 1]
    if message[limit - 1] != " ":
        head, sep, _partial = kept.rpartition(" ")
        if sep:
            kept = head
    return kept.rstrip() + "…"
