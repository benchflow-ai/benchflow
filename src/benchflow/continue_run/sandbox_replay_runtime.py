"""Stdlib-only replay proxy runtime uploaded into remote sandboxes."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

try:
    from benchflow_trajectory_redaction import redact_trajectory_obj
except ImportError:
    from benchflow.trajectories.redaction import redact_trajectory_obj

CAPTURE_STATE_WRITE_FAILED = "BENCHFLOW_CAPTURE_STATE_WRITE_FAILED"
# Provider calls can run for ten minutes; teardown must not snapshot a request
# while the provider timeout is still live.
PROVIDER_DRAIN_TIMEOUT_SEC = 610


def _n_messages(body):
    messages = body.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _completion_to_sse(body):
    base_id = body.get("id") or f"chatcmpl-replay-{int(time.time() * 1000)}"
    created = body.get("created") or int(time.time())
    model = body.get("model") or "replay"
    choices = body.get("choices") or [{}]
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "stop"

    def chunk(delta, finish=None):
        return json.dumps(
            {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    payloads = [chunk({"role": "assistant"})]
    content = message.get("content")
    if content:
        delta = {"content": content}
        for key in ("reasoning_content", "thinking"):
            if message.get(key):
                delta[key] = message[key]
        payloads.append(chunk(delta))

    for i, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        payloads.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": tool_call.get("index", i),
                            "id": tool_call.get("id"),
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    ]
                }
            )
        )

    final = {
        "id": base_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if isinstance(body.get("usage"), dict):
        final["usage"] = body["usage"]
    payloads.append(json.dumps(final))
    return payloads


@dataclass
class ReplayState:
    recorded: list
    upstream_url: str
    upstream_api_key: str
    upstream_model: str
    live_log_path: str
    state_path: str
    port: int
    strict_divergence: bool = False
    cursor: int = 0
    divergences: int = 0
    live_attempt_count: int = 0
    live_error_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    quiescing: bool = False
    active_live_requests: int = 0
    active_handlers: int = 0

    def __post_init__(self):
        self.condition = threading.Condition(self.lock)

    def _write_state(self):
        payload = {
            "port": self.port,
            "live_attempt_count": self.live_attempt_count,
            "live_error_count": self.live_error_count,
        }
        temporary = self.state_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
            os.replace(temporary, self.state_path)
        except Exception:
            self.live_error_count += 1
            with contextlib.suppress(OSError):
                os.unlink(self.state_path)
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            print(CAPTURE_STATE_WRITE_FAILED, file=sys.stderr, flush=True)
            raise

    def _check_divergence(self, incoming, recorded_request):
        want = _n_messages(recorded_request)
        got = _n_messages(incoming)
        if want and got and want != got:
            self.divergences += 1
            message = (
                f"replay divergence at turn {self.cursor}: agent sent {got} "
                f"messages, recorded turn had {want}"
            )
            if self.strict_divergence:
                raise RuntimeError(message)
            print(message, file=sys.stderr, flush=True)

    def next_response(self, request_body):
        with self.condition:
            if self.quiescing:
                self.live_attempt_count += 1
                self.live_error_count += 1
                self._write_state()
                return "error", 503, {"error": {"message": "replay proxy is quiescing"}}
            if self.cursor < len(self.recorded):
                exchange = self.recorded[self.cursor]
                self._check_divergence(
                    request_body,
                    ((exchange.get("request") or {}).get("body") or {}),
                )
                self.cursor += 1
                response = exchange.get("response") or {}
                return (
                    "replay",
                    int(response.get("status_code") or 200),
                    dict(response.get("body") or {}),
                )
            self.cursor += 1
            self.live_attempt_count += 1
            live_attempt = self.live_attempt_count
            self.active_live_requests += 1
            try:
                self._write_state()
            except Exception:
                self.active_live_requests -= 1
                self.condition.notify_all()
                raise

        try:
            status, body, provider_observed = self._forward_live(request_body)
            if provider_observed:
                try:
                    with self.lock:
                        self._append_live_exchange(
                            request_body, status, body, live_attempt
                        )
                except Exception:
                    with self.lock:
                        self.live_error_count += 1
                        self._write_state()
                    raise
            else:
                with self.lock:
                    self.live_error_count += 1
                    self._write_state()
            return "live", status, body
        finally:
            with self.condition:
                self.active_live_requests -= 1
                self.condition.notify_all()

    def quiesce(self, timeout=PROVIDER_DRAIN_TIMEOUT_SEC):
        deadline = time.monotonic() + timeout
        with self.condition:
            self.quiescing = True
            while self.active_live_requests or self.active_handlers > 1:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True

    def begin_quiesce(self):
        with self.condition:
            self.quiescing = True

    def handler_started(self):
        with self.condition:
            self.active_handlers += 1

    def handler_finished(self):
        with self.condition:
            self.active_handlers -= 1
            self.condition.notify_all()

    def _forward_live(self, request_body):
        forwarded = dict(request_body)
        forwarded["model"] = self.upstream_model
        forwarded["stream"] = False
        data = json.dumps(forwarded).encode("utf-8")
        request = urllib.request.Request(
            self.upstream_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.upstream_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode("utf-8")
                return int(response.status), json.loads(raw or "{}"), True
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw or "{}")
            except json.JSONDecodeError:
                body = {"error": {"message": raw or str(exc)}}
            return int(exc.code), body, True
        except Exception as exc:
            traceback.print_exc()
            return 500, {"error": {"message": str(exc)}}, False

    def _append_live_exchange(self, request_body, status, body, live_attempt):
        row = redact_trajectory_obj(
            {
                "request": {"body": request_body},
                "response": {"status_code": status, "body": body},
                "metadata": {"continuation_attempt": live_attempt},
            }
        )
        with open(self.live_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()


class ReplayHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print("replay-proxy " + format % args, file=sys.stderr, flush=True)

    @property
    def replay_server(self) -> ReplayServer:
        return cast("ReplayServer", self.server)

    @property
    def state(self) -> ReplayState:
        return self.replay_server.state

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, payloads):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for payload in payloads:
            self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/health/liveliness", "/v1/health"):
            self._send_json(200, {"status": "ok"})
            return
        if path in ("/v1/models", "/models"):
            self._send_json(
                200, {"object": "list", "data": [{"id": "replay", "object": "model"}]}
            )
            return
        self._send_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/benchflow/quiesce":
            # Stop the accept loop before the barrier can report success. The
            # server counts connections in ``process_request`` (before their
            # worker threads start), so every already-accepted handler is in
            # ``active_handlers`` even if it has not reached ``do_POST`` yet.
            self.state.begin_quiesce()
            self.replay_server.shutdown()
            self.replay_server.server_close()
            quiesced = self.state.quiesce()
            # The control request is the one handler deliberately excluded
            # from the barrier above. Retire it from the accepted-handler
            # accounting before publishing success so the response is also a
            # reliable, race-free signal that the capture lifecycle drained.
            self.replay_server.release_current_handler()
            self._send_json(
                200 if quiesced else 503,
                {"status": "quiesced" if quiesced else "quiesce_timeout"},
            )
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": f"not found: {path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"bad request: {exc}"}})
            return

        want_stream = bool(body.get("stream"))
        try:
            _, status, response = self.state.next_response(body)
        except RuntimeError as exc:
            self._send_json(409, {"error": {"message": str(exc), "type": "divergence"}})
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        if status >= 400 or not response.get("choices"):
            self._send_json(status, response)
            return
        if want_stream:
            self._send_sse(_completion_to_sse(response))
        else:
            self._send_json(status, response)


class ReplayServer(ThreadingHTTPServer):
    # ``/benchflow/quiesce`` closes the listener from its own worker thread.
    # Do not let ``server_close`` try to join that current thread; the normal
    # non-daemon handler lifecycle still keeps the process alive until the
    # response and every already-accepted request have finished.
    block_on_close = False

    def __init__(self, address, handler, state):
        super().__init__(address, handler)
        self.state = state
        self._handler_local = threading.local()

    def process_request(self, request, client_address):
        self.state.handler_started()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.state.handler_finished()
            raise

    def process_request_thread(self, request, client_address):
        self._handler_local.counted = True
        try:
            super().process_request_thread(request, client_address)
        finally:
            if self._handler_local.counted:
                self.state.handler_finished()
            self._handler_local.counted = False

    def release_current_handler(self):
        if getattr(self._handler_local, "counted", False):
            self._handler_local.counted = False
            self.state.handler_finished()


def main():
    with open(sys.argv[1], encoding="utf-8") as config_file:
        cfg = json.load(config_file)
    state = ReplayState(
        recorded=cfg["recorded"],
        upstream_url=cfg["upstream_url"],
        upstream_api_key=cfg["upstream_api_key"],
        upstream_model=cfg["upstream_model"],
        live_log_path=cfg["live_log_path"],
        state_path=cfg["state_path"],
        port=int(cfg["port"]),
        strict_divergence=bool(cfg.get("strict_divergence")),
    )
    server = ReplayServer(("127.0.0.1", int(cfg["port"])), ReplayHandler, state)
    state._write_state()
    server.serve_forever()


if __name__ == "__main__":
    main()
