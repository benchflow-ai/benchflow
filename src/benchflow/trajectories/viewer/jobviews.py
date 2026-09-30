"""Serve and export the job-level views (Outcomes, Pareto, Training).

The browse server builds the ``benchflow.outcomes/1`` document once, in a
background thread, and serves it at ``/api/outcomes``. ``bench eval view
--export`` writes the same views as one self-contained HTML file for sharing:
no trial links, no trajectories, paths reduced to names relative to the job,
and every string passed through the redaction ``bench traj upload`` applies
(secret-shaped values masked before anything leaves the machine).
"""

from __future__ import annotations

import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import BrowseRoots
from .outcomes import build_outcomes
from .render import _render_shell


def has_trials(path: Path, max_depth: int = 8) -> bool:
    """Whether a result.json exists within ``max_depth`` folders of ``path``
    (stops at the first; hidden folders are skipped)."""
    base = len(path.parts)
    for folder, subdirs, files in os.walk(path):
        if "result.json" in files:
            return True
        if len(Path(folder).parts) - base >= max_depth:
            subdirs.clear()
        else:
            subdirs[:] = [d for d in subdirs if not d.startswith(".")]
    return False


def build_for_roots(
    roots: BrowseRoots, linked: dict[str, Path] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """The outcomes document for the served roots, with viewer ids as links.

    ``linked`` (when given) receives each linked id and its trial folder.
    """

    def link_for(trial_dir: Path) -> str | None:
        rid = roots.id_for(trial_dir)
        if rid is not None and linked is not None:
            linked[rid] = trial_dir
        return rid

    return build_outcomes(roots.pairs(), link_for=link_for, **kwargs)


class OutcomesCache:
    """Builds the document once in the background; ``get`` waits for it."""

    def __init__(self, roots: BrowseRoots) -> None:
        self.roots = roots
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._doc: dict[str, Any] | None = None
        self._error: str | None = None
        self._linked: dict[str, Path] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and not self._done.is_set():
                return
            self._done.clear()
            self._thread = threading.Thread(target=self._build, daemon=True)
            self._thread.start()

    def _build(self) -> None:
        linked: dict[str, Path] = {}
        try:
            doc, error = build_for_roots(self.roots, linked), None
        except Exception as exc:  # shown on the page, never a crashed server
            doc, error = None, f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._doc, self._error, self._linked = doc, error, linked
        self._done.set()

    def get(self, timeout: float = 600.0) -> tuple[dict[str, Any] | None, str | None]:
        if self._thread is None:
            self.start()
        if not self._done.wait(timeout):
            return None, "still building the outcomes; reload in a moment"
        with self._lock:
            return self._doc, self._error

    def linked(self, rid: str) -> Path | None:
        """The trial folder of a rollout id the finished document links to."""
        with self._lock:
            return self._linked.get(rid)


# A user's home folder named anywhere in a string: /home/<user> or /Users/<user>.
_USER_HOME = re.compile(
    r"/(?:home|Users)/[^/\s\"'<>]+|\b[A-Za-z]:\\+Users\\+[^\\\s\"'<>]+"
)


def redact_for_export(
    doc: dict[str, Any], paths: list[Path]
) -> tuple[dict[str, Any], Counter[str]]:
    """The document as it may leave the machine.

    Trial links are dropped (they only work against the local server), the
    served folders' absolute paths become ``<job>``, any home folder
    (``$HOME``, the account's home, and every ``/home/<user>`` or
    ``/Users/<user>`` prefix, since error text can name paths of other runs)
    becomes ``~``, and every value goes through
    :func:`benchflow.publish.redact.redact_value`.
    """
    from benchflow.publish.redact import redact_value

    shared = dict(doc)
    columns = dict(shared["columns"])
    columns["link"] = [None] * len(columns["link"])
    shared["columns"] = columns
    shared.pop("timing", None)
    homes = {str(Path.home())}
    try:
        import pwd

        homes.add(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        pass
    replacements = sorted(
        {(str(p.resolve()), "<job>") for p in paths}
        | {(str(p), "<job>") for p in paths if p.is_absolute()}
        | {(home, "~") for home in homes},
        key=lambda pair: -len(pair[0]),
    )

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for old, new in replacements:
                if len(old) > 1 and old in value:
                    value = value.replace(old, new)
            return _USER_HOME.sub("~", value)
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    categories: Counter[str] = Counter()
    redacted, _ = redact_value(scrub(shared), categories=categories)
    return redacted, categories


def export_html(paths: list[Path], out: Path) -> tuple[Path, Counter[str], int]:
    """Write the job views of ``paths`` to ``out``; (path, masked kinds, trials)."""
    from benchflow.publish.redact import format_redaction_breakdown

    roots = BrowseRoots(paths)
    doc = build_for_roots(roots)
    if not doc["n"]:
        raise FileNotFoundError(
            "no trial (a folder with result.json) under "
            + ", ".join(str(p) for p in paths)
            + "; nothing to export"
        )
    shared, categories = redact_for_export(doc, paths)
    shared["redaction"] = format_redaction_breakdown(categories) if categories else None
    # The title comes from the redacted document, like everything else.
    page = _render_shell(
        " + ".join(shared["roots"]) + " - outcomes",
        {"mode": "export", "outcomes": shared},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out, categories, int(doc["n"])
