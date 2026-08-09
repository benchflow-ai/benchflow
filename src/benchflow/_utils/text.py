"""Plain-text rendering for console and log lines.

Two helpers live here: :func:`truncate_end`, which shortens a message
without letting a sliced token read as a whole word, and
:func:`describe_exception`, which renders an exception so the resulting
line is never detail-free.

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


def describe_exception(exc: BaseException) -> str:
    """Render ``exc`` as a one-line description that is never detail-free.

    ``f"{exc}"`` keeps only ``str(exc)``, which for some SDK errors carries
    no information at all. The Daytona SDK wraps every toolbox call as
    ``"<prefix>: " + str(underlying)``, and httpx raises its timeout and
    connection errors with an *empty* message — so a read timeout on
    ``execute_session_command`` stringifies to the bare
    ``"Failed to execute session command: "``, a message whose detail after
    the colon is empty. The exception *class* (``DaytonaTimeoutError`` vs
    ``DaytonaConnectionError``) is then the only surviving evidence of what
    actually went wrong, and plain interpolation discards it.

    Lead with the class name so "the exec timed out" can never again be
    indistinguishable from "the connection dropped". The trailing
    ``status_code``/``error_code`` fields serve the *other* shape of SDK
    failure — an OpenAPI exception carrying an HTTP response — and are
    absent by construction for the detail-free transport case above, which
    never reaches a response at all.
    """
    name = type(exc).__name__
    message = str(exc).strip()
    # Approximation: this tests the whole message, not specifically the
    # detail after a wrapper prefix — a message that legitimately ends in a
    # colon is annotated too. That is harmless, and the alternative means
    # knowing every vendor's prefix.
    if message.endswith(":"):
        message = f"{message} (no detail)"
    described = f"{name}: {message}" if message else f"{name} (no message)"
    details: list[str] = []
    status_code = getattr(exc, "status_code", None)
    # Both fields are "absent" when falsy, but only ``status_code`` has a
    # meaningful zero-ish value to protect (no HTTP status is 0, yet an
    # explicit 0 should still surface as evidence of a malformed response).
    if status_code is not None:
        details.append(f"status_code={status_code}")
    error_code = getattr(exc, "error_code", None)
    if error_code:
        details.append(f"error_code={error_code}")
    if details:
        described = f"{described} [{', '.join(details)}]"
    return described
