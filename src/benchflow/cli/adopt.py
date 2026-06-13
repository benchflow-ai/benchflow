"""``bench adopt`` — bring an external benchmark into the environment framework.

The adoption pipeline reads ``init → convert → verify``: scaffold a benchmark
package, drive the codex ``CONVERT.md`` conversion, then run the parity gate.
These commands moved out of ``bench agent`` (where ``agent create`` misleadingly
scaffolded a *benchmark*, not an agent); the legacy ``bench agent
create|run|verify`` stay as hidden deprecated aliases through 0.6.

Registered onto the top-level app by :func:`register_adopt`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

import typer

from benchflow.agent_router import register_agent_router


def register_adopt(app: typer.Typer) -> None:
    """Attach the ``adopt`` command group (``init`` / ``convert`` / ``verify``)."""
    adopt_app = typer.Typer(
        help=(
            "Adopt an external benchmark into the environment framework "
            "(init → convert → verify)."
        )
    )
    app.add_typer(adopt_app, name="adopt", rich_help_panel="Core")
    register_agent_router(adopt_app)
