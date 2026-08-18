"""HTTP serving: single-trajectory pages (with --confirm) and browse mode."""

import sys
from pathlib import Path

from .catalog import (
    _discover_rollouts,
    _resolve_browse_rollout,
    _rollout_summary,
    _runs_cap,
)
from .legacy import (
    _NO_TRAJECTORIES_HTML,
    _inject_confirm_bar,
    render_jsonl_file,
    render_rollout,
)
from .payload import _build_acp_payload, _is_acp_rollout_dir, _safe_json
from .render import _render_shell
from .sources import HfDatasetSource, parse_source, resolve_hf_dataset


def serve(
    rollout_path: str,
    port: int = 8888,
    prompts: list[str] | None = None,
    confirm: bool = False,
    redaction_summary: str | None = None,
) -> str | None:
    """Serve a trial directory, a session JSONL file, a directory of runs,
    or an ``hf://`` dataset source.

    ``hf://<org>/<name>[/subpath]`` fetches the viewer-relevant slice of a
    HuggingFace trajectory dataset (e.g. the community ground-truth uploads)
    into the local HF cache and serves it like any local directory.

    Single trajectories (a rollout directory, or a session JSONL file) keep
    the one-page behavior, including the ``confirm=True`` Approve/Reject
    contract documented on :func:`_serve_single`. A directory that is not
    itself a rollout but contains rollout directories is served in browse
    mode instead: a run sidebar plus ``/api/rollout?id=…`` endpoints the page
    uses to load traces dynamically (see :func:`_serve_browse`).

    ``confirm`` needs exactly one trajectory to approve, so combining it with
    a multi-run directory is an error.
    """
    try:
        source = parse_source(str(rollout_path))
    except ValueError as exc:
        print(exc)
        sys.exit(1)
    if isinstance(source, HfDatasetSource):
        path = resolve_hf_dataset(source)
    else:
        path = source.path
    if (
        path.is_dir()
        and not _is_acp_rollout_dir(path)
        and not any(path.glob("turn*.txt"))
    ):
        cap = _runs_cap()
        rollouts = _discover_rollouts(path, cap=cap + 1)
        if rollouts:
            capped = len(rollouts) > cap
            n_runs = min(len(rollouts), cap)
            if confirm:
                print(
                    "--confirm needs a single rollout or session file, but "
                    f"{path} is a directory of {n_runs}{'+' if capped else ''} runs"
                )
                sys.exit(1)
            _serve_browse(path, port, n_runs=n_runs, capped=capped)
            return None
    return _serve_single(path, port, prompts, confirm, redaction_summary)


def _serve_single(
    path: Path,
    port: int,
    prompts: list[str] | None,
    confirm: bool,
    redaction_summary: str | None,
) -> str | None:
    """Serve one trial directory or session JSONL file as a web page.

    With ``confirm=True`` the page carries an Approve/Reject bar posting to
    ``/decision``; the first valid decision shuts the server down and is
    returned as ``"approved"`` or ``"rejected"`` after printing a
    machine-readable ``DECISION: <value>`` line to stdout. Without it the
    server runs until Ctrl+C and the return value is ``None`` — exactly the
    pre-confirm behavior (no bar, no POST endpoint).

    ``redaction_summary`` is an optional caller-composed line (e.g. ``"2 API
    keys, 1 bearer token"``) rendered inside the confirm bar so the reviewer
    sees what upload-time redaction would mask. Presentation-only; it has no
    effect without ``confirm=True``.
    """
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    write_sidecar = False
    if path.is_file():
        html_content = render_jsonl_file(path)
    elif path.is_dir():
        html_content = render_rollout(path, prompts)
        write_sidecar = True
    else:
        print(f"Not a file or directory: {path}")
        sys.exit(1)

    if html_content == _NO_TRAJECTORIES_HTML:
        # Don't write a blank trajectory.html into an unrelated directory or
        # start a server for nothing — fail fast like the not-a-directory path.
        print(f"No trajectories found in {path}")
        sys.exit(1)
    if write_sidecar:
        # The sidecar stays the plain page: the confirm bar is a one-shot
        # interaction against this live server, not part of the artifact.
        (path / "trajectory.html").write_text(html_content)
    if confirm:
        html_content = _inject_confirm_bar(html_content, redaction_summary)

    print(f"Trajectory viewer: http://localhost:{port}")
    print(f"Trial: {path}")
    if confirm:
        print("Waiting for Approve / Not this one in the browser (Ctrl+C to stop)\n")
    else:
        print("Press Ctrl+C to stop\n")

    decision: str | None = None

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format, *args):
            pass

    handler_cls: type[SimpleHTTPRequestHandler] = Handler

    if confirm:

        class ConfirmHandler(Handler):
            def do_POST(self):
                nonlocal decision
                if self.path != "/decision":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("utf-8", "replace").strip()
                if body not in ("approve", "reject"):
                    self.send_error(400, "Body must be 'approve' or 'reject'")
                    return
                decision = "approved" if body == "approve" else "rejected"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(decision.encode())
                # shutdown() blocks until serve_forever returns, and this
                # handler runs inside that loop — stop from a helper thread.
                threading.Thread(target=server.shutdown, daemon=True).start()

        handler_cls = ConfirmHandler

    server = HTTPServer(("localhost", port), handler_cls)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        return None
    finally:
        server.server_close()

    if decision is not None:
        print(f"DECISION: {decision}", flush=True)
    return decision


def _serve_browse(base: Path, port: int, n_runs: int, capped: bool = False) -> None:
    """Multi-rollout browser: sidebar shell + JSON API, rescanned per request."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import parse_qs, urlsplit

    class Handler(SimpleHTTPRequestHandler):
        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            headers: tuple[tuple[str, str], ...] = (),
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _scan(self) -> tuple[list[str], bool]:
            """Fresh id list plus whether the cap truncated it."""
            cap = _runs_cap()
            ids = _discover_rollouts(base, cap=cap + 1)
            return ids[:cap], len(ids) > cap

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                ids, capped = self._scan()
                shell = _render_shell(
                    base.name,
                    {
                        "mode": "browse",
                        "capped": capped,
                        "rollouts": [_rollout_summary(base, r) for r in ids],
                    },
                )
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    shell.encode("utf-8", errors="replace"),
                )
            elif parsed.path == "/api/rollouts":
                ids, truncated = self._scan()
                body = _safe_json([_rollout_summary(base, r) for r in ids])
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    body.encode("utf-8", errors="replace"),
                    # API consumers must be able to detect truncation too; the
                    # body stays a bare list for backward compatibility.
                    headers=(("X-BenchFlow-Capped", "1"),) if truncated else (),
                )
            elif parsed.path == "/api/rollout":
                rid = (parse_qs(parsed.query).get("id") or [None])[0]
                rollout_dir = _resolve_browse_rollout(base, rid)
                if rollout_dir is None:
                    self._send(
                        404,
                        "application/json; charset=utf-8",
                        b'{"error": "unknown rollout id"}',
                    )
                    return
                body = _safe_json(_build_acp_payload(rollout_dir, None).to_payload())
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    body.encode("utf-8", errors="replace"),
                )
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def log_message(self, format, *args):
            pass

    runs_desc = f"{n_runs} runs"
    if capped:
        runs_desc = f"first {n_runs} runs (capped — raise BENCHFLOW_VIEWER_MAX_RUNS)"
    print(f"Trajectory browser: http://localhost:{port}")
    print(f"Scanning: {base} ({runs_desc})")
    print("Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
