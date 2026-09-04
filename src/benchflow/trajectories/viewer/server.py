"""HTTP serving: single-trajectory pages (with --confirm) and browse mode."""

import hmac
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit

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
from .sources import (
    HfDatasetSource,
    ViewerSourceError,
    parse_source,
    resolve_hf_dataset,
)

_CONFIRM_TOKEN_HEADER = "X-BenchFlow-Confirm-Token"
_MAX_DECISION_BYTES = 32


def _utf8_safe_text(value: str) -> str:
    """Replace lone surrogates so rendered pages are always valid UTF-8."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _matches_local_authority(
    value: str, expected_port: int, *, is_origin: bool = False
) -> bool:
    """Accept only the printed localhost authority, never a DNS-rebinding host."""
    try:
        parsed = urlsplit(value if is_origin else f"http://{value}")
        supplied_port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() != "http" or parsed.hostname != "localhost":
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    effective_port = supplied_port if supplied_port is not None else 80
    return effective_port == expected_port


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
        if isinstance(source, HfDatasetSource):
            is_hf_source = True
            path = resolve_hf_dataset(source)
        else:
            is_hf_source = False
            path = source.path
    except (ValueError, ViewerSourceError) as exc:
        # Source parsing/materialization is library code and raises typed
        # errors. This CLI-facing server boundary owns the process contract.
        print(exc)
        raise SystemExit(1) from None
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
    return _serve_single(
        path,
        port,
        prompts,
        confirm,
        redaction_summary,
        persist_sidecar=not is_hf_source,
    )


def _serve_single(
    path: Path,
    port: int,
    prompts: list[str] | None,
    confirm: bool,
    redaction_summary: str | None,
    *,
    persist_sidecar: bool = True,
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

    ``persist_sidecar=False`` renders a directory without writing
    ``trajectory.html`` into remote cache snapshots.
    """
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    write_sidecar = False
    if path.is_file():
        html_content = render_jsonl_file(path)
    elif path.is_dir():
        html_content = render_rollout(path, prompts)
        write_sidecar = persist_sidecar
    else:
        print(f"Not a file or directory: {path}")
        sys.exit(1)

    if html_content == _NO_TRAJECTORIES_HTML:
        # Don't write a blank trajectory.html into an unrelated directory or
        # start a server for nothing — fail fast like the not-a-directory path.
        print(f"No trajectories found in {path}")
        sys.exit(1)
    html_content = _utf8_safe_text(html_content)
    if write_sidecar:
        # The sidecar stays the plain page: the confirm bar is a one-shot
        # interaction against this live server, not part of the artifact.
        (path / "trajectory.html").write_text(html_content, encoding="utf-8")
    confirm_token: str | None = None
    if confirm:
        # This secret is unique to this server invocation and embedded only in
        # its same-origin page. A foreign page can submit a simple cross-origin
        # POST to localhost, but it cannot read or reproduce this token.
        confirm_token = secrets.token_urlsafe(32)
        html_content = _inject_confirm_bar(
            html_content,
            redaction_summary,
            confirm_token=confirm_token,
        )
        # The summary is caller-provided and may itself contain a lone
        # surrogate even though the rendered trajectory was normalized above.
        html_content = _utf8_safe_text(html_content)

    html_bytes = html_content.encode("utf-8")
    decision: str | None = None
    expected_port = port

    class Handler(SimpleHTTPRequestHandler):
        def _has_expected_host(self) -> bool:
            values = self.headers.get_all("Host") or []
            return len(values) == 1 and _matches_local_authority(
                values[0], expected_port
            )

        def _send_page(self, *, include_body: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            if include_body:
                self.wfile.write(html_bytes)

        def do_GET(self):
            if urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            if not self._has_expected_host():
                self.send_error(403, "Invalid Host header")
                return
            self._send_page(include_body=True)

        def do_HEAD(self):
            # The inherited do_HEAD serves filesystem paths rooted at the
            # process cwd; this server has exactly one page — mirror do_GET.
            if urlsplit(self.path).path != "/":
                self.send_response(405)
                self.send_header("Allow", "GET")
                self.end_headers()
                return
            if not self._has_expected_host():
                self.send_error(403, "Invalid Host header")
                return
            self._send_page(include_body=False)

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
                if not self._has_expected_host():
                    self.send_error(403, "Invalid confirmation origin")
                    return
                origins = self.headers.get_all("Origin") or []
                if len(origins) != 1 or not _matches_local_authority(
                    origins[0], expected_port, is_origin=True
                ):
                    self.send_error(403, "Invalid confirmation origin")
                    return
                tokens = self.headers.get_all(_CONFIRM_TOKEN_HEADER) or []
                if (
                    len(tokens) != 1
                    or confirm_token is None
                    or not tokens[0].isascii()
                    or not hmac.compare_digest(tokens[0], confirm_token)
                ):
                    self.send_error(403, "Invalid confirmation token")
                    return
                if self.headers.get("Transfer-Encoding") is not None:
                    self.send_error(400, "Transfer-Encoding is not supported")
                    return
                lengths = self.headers.get_all("Content-Length") or []
                if not lengths:
                    self.send_error(411, "Content-Length is required")
                    return
                if (
                    len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdigit()
                    or len(lengths[0]) > len(str(_MAX_DECISION_BYTES))
                ):
                    self.send_error(400, "Invalid Content-Length")
                    return
                length = int(lengths[0])
                if length > _MAX_DECISION_BYTES:
                    self.close_connection = True
                    self.send_error(413, "Decision body is too large")
                    return
                try:
                    raw_body = self.rfile.read(length)
                    if len(raw_body) != length:
                        self.send_error(400, "Incomplete decision body")
                        return
                    body = raw_body.decode("utf-8").strip()
                except UnicodeDecodeError:
                    self.send_error(400, "Decision body must be UTF-8")
                    return
                if body not in ("approve", "reject"):
                    self.send_error(400, "Body must be 'approve' or 'reject'")
                    return
                if decision is not None:
                    self.send_error(409, "A decision was already recorded")
                    return
                decision = "approved" if body == "approve" else "rejected"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(decision)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(decision.encode("ascii"))
                # shutdown() blocks until serve_forever returns, and this
                # handler runs inside that loop — stop from a helper thread.
                threading.Thread(target=server.shutdown, daemon=True).start()

        handler_cls = ConfirmHandler

    server = HTTPServer(("localhost", port), handler_cls)
    expected_port = server.server_port
    print(f"Trajectory viewer: http://localhost:{expected_port}")
    print(f"Trial: {path}")
    if confirm:
        print("Waiting for Approve / Not this one in the browser (Ctrl+C to stop)\n")
    else:
        print("Press Ctrl+C to stop\n")
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
    from urllib.parse import parse_qs

    expected_port = port

    class Handler(SimpleHTTPRequestHandler):
        def _has_expected_host(self) -> bool:
            values = self.headers.get_all("Host") or []
            return len(values) == 1 and _matches_local_authority(
                values[0], expected_port
            )

        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _scan(self) -> tuple[list[str], bool]:
            """Fresh id list plus whether the cap truncated it."""
            cap = _runs_cap()
            ids = _discover_rollouts(base, cap=cap + 1)
            return ids[:cap], len(ids) > cap

        def do_GET(self):
            if not self._has_expected_host():
                self.send_error(403, "Invalid Host header")
                return
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

        def do_HEAD(self):
            # The inherited do_HEAD serves filesystem paths rooted at the
            # process cwd, bypassing the rollout-id whitelist — answer only
            # for the shell page and refuse everything else.
            if not self._has_expected_host():
                self.send_error(403, "Invalid Host header")
                return
            if urlsplit(self.path).path == "/":
                self._send(200, "text/html; charset=utf-8", b"")
            else:
                self.send_response(405)
                self.send_header("Allow", "GET")
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), Handler)
    expected_port = server.server_port
    runs_desc = f"{n_runs} runs"
    if capped:
        runs_desc = f"first {n_runs} runs (capped — raise BENCHFLOW_VIEWER_MAX_RUNS)"
    print(f"Trajectory browser: http://localhost:{expected_port}")
    print(f"Scanning: {base} ({runs_desc})")
    print("Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
