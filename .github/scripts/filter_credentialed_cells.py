#!/usr/bin/env python3
"""Drop planned matrix cells whose agent+model credential is absent from the env.

The scope planner (``integration_matrix.py``) is PURE and deterministic: it emits
the full configured roster for a scope without consulting the live environment.
That is the right boundary for a planner, but it means a cell can be planned for
an agent whose required API key is not present in CI. Before this filter existed,
such a cell would spin up a Daytona sandbox, run the agent up to the first LLM
call, fail there, burn the sandbox, and leave the grader logging a *false-red*
slot that looks like a real regression.

This script is the SEPARATE env-aware step the workflow runs between ``plan`` and
``run-matrix``. For every cell it asks ``benchflow.agents.env.resolve_agent_env``
whether the agent+model can resolve a credential from the current process env. A
``ValueError`` whose message says a key is "required ... not set" means a missing
credential: the cell is DROPPED and recorded under ``skipped_uncredentialed`` with
the missing key name. Every other outcome (resolves cleanly, or raises for an
unrelated reason) KEEPS the cell — we only ever drop on a *documented* missing
credential, never on an ambiguous failure.

Fail-OPEN by design: if ``benchflow`` cannot be imported (e.g. the trusted base
during a bootstrap PR, where the package is not installed in the plan job), the
matrix passes through UNCHANGED and a warning is logged. The filter must never be
the thing that blocks a legitimate run.

CLI::

    filter_credentialed_cells.py --matrix matrix.json --out matrix.json [--json]

Input/Output schema (same shape in and out, plus one added top-level key)::

    {"matrix": [ {"agent": ..., "model": ..., ...}, ... ], ...}
    -> {"matrix": [ ...kept cells... ],
        "skipped_uncredentialed": [
            {"agent": ..., "model": ..., "missing_key": "AWS_BEARER_TOKEN_BEDROCK"},
            ...
        ],
        ...}  # all other top-level keys preserved verbatim
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A ValueError raised by resolve_agent_env counts as a missing-credential drop
# only when its message contains BOTH of these markers. resolve_agent_env emits
# "<KEY> required for model '<m>' but not set. ..." (and the Bedrock variant
# "<KEY> required for Bedrock model '<m>' but not set. ...") — both start with
# the missing key as the first whitespace-delimited token.
_MISSING_CRED_MARKERS = ("required", "not set")


def _extract_missing_key(message: str) -> str:
    """Pull the missing env-var name out of a resolve_agent_env ValueError.

    The credential messages are shaped "<KEY> required ... but not set." so the
    missing key is the first whitespace-delimited token. Fall back to the whole
    (stripped) message if the shape is unexpected, so the record is never empty.
    """
    stripped = message.strip()
    if not stripped:
        return ""
    first = stripped.split(None, 1)[0]
    # Defensive: strip stray surrounding punctuation/quotes from the token.
    return first.strip("'\"`:,.")


def _is_missing_credential(message: str) -> bool:
    """True when a ValueError message denotes a missing-credential condition."""
    lowered = message.lower()
    return all(marker in lowered for marker in _MISSING_CRED_MARKERS)


def filter_matrix(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``plan`` with un-credentialed cells dropped.

    Cells whose agent+model required credential is absent are removed from
    ``matrix`` and appended to a new ``skipped_uncredentialed`` list. The
    benchflow import is LAZY and inside this function: if it fails, the plan is
    returned unchanged (fail-open) with a logged warning. The planner output is
    never mutated in place.
    """
    result = dict(plan)
    cells = plan.get("matrix") or []

    try:
        # Lazy import: keeps this module importable (and unit-testable for the
        # pure pass-through path) even when benchflow is not installed.
        from benchflow.agents.env import resolve_agent_env
    except Exception as exc:  # fail open on ANY import failure
        logger.warning(
            "benchflow.agents.env unavailable (%s); passing %d cell(s) through "
            "uncredentialed-unfiltered (fail-open)",
            exc,
            len(cells),
        )
        result["matrix"] = list(cells)
        result["skipped_uncredentialed"] = []
        return result

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cell in cells:
        agent = cell.get("agent")
        model = cell.get("model")
        try:
            resolve_agent_env(agent, model, {})
        except ValueError as exc:
            message = str(exc)
            if _is_missing_credential(message):
                missing_key = _extract_missing_key(message)
                logger.warning(
                    "dropping uncredentialed cell agent=%s model=%s (missing %s)",
                    agent,
                    model,
                    missing_key or "<unknown>",
                )
                skipped.append(
                    {
                        "agent": agent,
                        "model": model,
                        "missing_key": missing_key,
                    }
                )
                continue
            # A ValueError that is NOT a missing-credential (e.g. a malformed
            # provider URL template) is not ours to swallow — keep the cell and
            # let run-matrix surface the real error.
            kept.append(cell)
        except Exception:  # never drop on an ambiguous failure
            # Any other resolution error is not a credential gap; keep the cell.
            kept.append(cell)
        else:
            kept.append(cell)

    result["matrix"] = kept
    result["skipped_uncredentialed"] = skipped
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop planned matrix cells whose required credential is absent, "
            "recording them under 'skipped_uncredentialed'."
        )
    )
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to the planner's matrix.json ({matrix:[cells]}).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the filtered matrix.json (may equal --matrix).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the filtered plan as compact JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    plan = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    filtered = filter_matrix(plan)

    skipped = filtered.get("skipped_uncredentialed") or []
    kept_n = len(filtered.get("matrix") or [])
    if skipped:
        for entry in skipped:
            logger.info(
                "SKIP uncredentialed agent=%s model=%s missing=%s",
                entry.get("agent"),
                entry.get("model"),
                entry.get("missing_key"),
            )
    logger.info(
        "credential filter: kept %d cell(s), skipped %d uncredentialed",
        kept_n,
        len(skipped),
    )

    Path(args.out).write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")

    if args.json:
        sys.stdout.write(json.dumps(filtered, separators=(",", ":")) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
