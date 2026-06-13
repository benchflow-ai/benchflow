"""``bench eval adopt`` — bring an external benchmark into benchflow.

The adoption pipeline reads ``init → convert → verify``: scaffold a benchmark
package, drive the codex ``CONVERT.md`` conversion, then run the parity gate.
It lives under ``eval`` because ``eval`` is the universal benchmark entry point
(``eval create`` runs a benchmark; ``eval adopt`` is the manual path to make a
foreign benchmark runnable). It previously moved off ``bench agent`` (#735) into
a top-level ``bench adopt``; both ``bench adopt`` and ``bench agent
create|run|verify`` now stay as hidden deprecated aliases through 0.6.

``register_eval_adopt`` mounts the canonical group under the ``eval`` Typer;
``register_adopt_deprecated`` mounts the hidden top-level alias. ``cli/main.py``
only wires the calls.
"""

from __future__ import annotations

import typer

from benchflow.agent_router import ADOPT_VERBS, register_agent_router

_ADOPT_HELP = "Adopt an external benchmark into benchflow (init → convert → verify)."


def register_eval_adopt(eval_app: typer.Typer) -> None:
    """Attach the canonical ``eval adopt`` subgroup (``init`` / ``convert`` / ``verify``)."""
    adopt_app = typer.Typer(help=_ADOPT_HELP)
    eval_app.add_typer(adopt_app, name="adopt")
    register_agent_router(adopt_app)


def register_adopt_deprecated(app: typer.Typer) -> None:
    """Attach the hidden deprecated top-level ``bench adopt`` → ``bench eval adopt``."""
    adopt_app = typer.Typer(help="deprecated — use `bench eval adopt`.")
    app.add_typer(adopt_app, name="adopt", hidden=True)
    register_agent_router(adopt_app, verbs=ADOPT_VERBS, deprecated_as="adopt")
