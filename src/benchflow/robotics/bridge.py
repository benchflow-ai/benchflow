"""A per-trial, capability-limited bridge to an already commissioned arm.

The agent never receives the operator socket, USB devices, host filesystem,
or an endpoint for changing the safety envelope. This is an additional gate;
the Metal harness remains responsible for IK, trajectories and motion limits.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import secrets
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class HarnessTransport:
    def __init__(self, socket_path: Path, timeout: float = 90):
        self.socket_path = socket_path
        self.timeout = timeout

    def __call__(self, command: str, args: list[str]) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(str(self.socket_path))
            with client.makefile("rwb") as stream:
                stream.write(
                    json.dumps({"command": command, "args": args}).encode() + b"\n"
                )
                stream.flush()
                line = stream.readline(8_000_000)
        if not line:
            raise ConnectionError("Harness closed without a command receipt")
        return json.loads(line)


class TrialLease:
    """A persistent lock: a crashed owner needs operator review, not auto-retry."""

    def __init__(self, root: Path, robot_id: str, trial_id: str):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / hashlib.sha256(robot_id.encode()).hexdigest()[:24]
        self.trial_id = trial_id

    def __enter__(self):
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"Robot already leased; inspect {self.path} before another trial"
            ) from exc
        write_json(
            self.path / "owner.json", {"pid": os.getpid(), "trial_id": self.trial_id}
        )
        return self

    def __exit__(self, *_):
        (self.path / "owner.json").unlink()
        self.path.rmdir()


def validate_command(command: Any, args: Any) -> tuple[str, list[str]]:
    sizes = {
        "observe": 0,
        "status": 0,
        "tip": 4,
        "roll": 1,
        "gripper": 1,
        "monitor": 1,
        "rest": 0,
        "finish": 0,
    }
    if not isinstance(command, str) or command not in sizes:
        raise ValueError(
            "Only observe, status, tip, roll, gripper, monitor, rest, finish are allowed"
        )
    if not isinstance(args, list) or len(args) != sizes[command]:
        raise ValueError("Incorrect argument count")
    if any(isinstance(v, bool) or not isinstance(v, (str, int, float)) for v in args):
        raise ValueError("Arguments must be finite numbers")
    try:
        values = [float(v) for v in args]
    except (ValueError, OverflowError) as exc:
        raise ValueError("Arguments must be finite numbers") from exc
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Arguments must be finite numbers")
    if command == "monitor" and not 0 < values[0] <= 10:
        raise ValueError("Monitor duration must be in (0, 10] seconds")
    # These broad sanity bounds cannot relax the harness's commissioned limits.
    if command == "tip" and not (
        abs(values[0]) <= 0.7
        and abs(values[1]) <= 0.7
        and 0 <= values[2] <= 0.7
        and -90 <= values[3] <= 90
    ):
        raise ValueError("Invalid end-effector pose")
    if command == "gripper" and not 0 <= values[0] <= 112:
        raise ValueError("Gripper command must be between 0 and 112 degrees")
    if command == "roll" and not -180 <= values[0] <= 180:
        raise ValueError("Invalid wrist roll")
    normalized = [str(v) for v in values]
    if command == "roll":
        return "goto", ["wrist_roll=" + normalized[0]]
    return command, normalized


class TrialBridge:
    def __init__(
        self,
        *,
        output: Path,
        transport: Callable | None = None,
        frames_root: Path | None = None,
        robot_id: str,
        allow_motion: bool = False,
        timeout: float = 1800,
        max_commands: int = 150,
        recording_healthy: Callable[[], bool] = lambda: False,
        arms: dict[str, dict] | None = None,
        min_separation_m: float = 0.20,
    ):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        (output / "observations").mkdir(exist_ok=True)
        # One commissioned harness session per arm. A single-arm bridge keeps the
        # original receipt and observation naming; a multi-arm bridge tags both
        # with the arm name and refuses tip targets close to the other gripper.
        if arms is None:
            if transport is None or frames_root is None:
                raise ValueError("Provide transport and frames_root, or arms")
            arms = {"arm": {"transport": transport, "frames_root": frames_root}}
        if not arms:
            raise ValueError("At least one arm is required")
        self.arms: dict[str, dict] = {}
        for name, spec in arms.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Arm names must be identifiers")
            offset = tuple(float(v) for v in spec.get("offset_m", (0.0, 0.0, 0.0)))
            if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
                raise ValueError("Arm offset must be three finite metres")
            self.arms[name] = {
                "transport": spec["transport"],
                "frames_root": Path(spec["frames_root"]).resolve(),
                "offset_m": offset,
            }
        self.default_arm = next(iter(self.arms))
        self.multi_arm = len(self.arms) > 1
        self.min_separation_m = min_separation_m
        #: Last known gripper tip of each arm, in the default arm's base frame.
        self.last_tip: dict[str, tuple[float, float, float]] = {}
        self.robot_id = robot_id
        self.allow_motion = allow_motion
        self.timeout = timeout
        self.max_commands = max_commands
        self.recording_healthy = recording_healthy
        self.token = secrets.token_urlsafe(32)
        self.started: float | None = None
        self.closed = False
        self.halted = False
        self.halt_reason: str | None = None
        self.count = 0
        self.lock = threading.Lock()
        self.receipts: dict[str, tuple[str, dict]] = {}
        self.server: ThreadingHTTPServer | None = None

    @property
    def transport(self) -> Callable:
        """The default arm's harness transport (single-arm compatibility)."""
        return self.arms[self.default_arm]["transport"]

    @transport.setter
    def transport(self, value: Callable) -> None:
        self.arms[self.default_arm]["transport"] = value

    @property
    def frames_root(self) -> Path:
        return self.arms[self.default_arm]["frames_root"]

    @frames_root.setter
    def frames_root(self, value: Path) -> None:
        self.arms[self.default_arm]["frames_root"] = Path(value).resolve()

    def event(self, event: str, **data) -> None:
        with (self.output / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": event,
                        "monotonic": time.monotonic(),
                        "utc_epoch": time.time(),
                        **data,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def activate(self) -> None:
        if not self.recording_healthy():
            raise RuntimeError(
                "Both camera recordings must be healthy before activation"
            )
        self.started = time.monotonic()
        self.event("activated", allow_motion=self.allow_motion)

    def _common_frame(self, arm: str, tip: list | tuple) -> tuple[float, float, float]:
        offset = self.arms[arm]["offset_m"]
        return (
            float(tip[0]) + offset[0],
            float(tip[1]) + offset[1],
            float(tip[2]) + offset[2],
        )

    def note_state(self, arm: str, reply: dict) -> None:
        """Remember where an arm's gripper is, for the separation guard."""
        tip = reply.get("tip_m")
        if isinstance(tip, (list, tuple)) and len(tip) == 3:
            with contextlib.suppress(TypeError, ValueError):
                self.last_tip[arm] = self._common_frame(arm, tip)

    def separation_rejection(self, arm: str, target: list[float]) -> str | None:
        """Refuse a tip target within `min_separation_m` (horizontally) of another
        arm's last known gripper position. The harness does not check arm-to-arm
        collisions, and a wrist housing reaches well beyond its own tip."""
        if not self.multi_arm:
            return None
        mine = self._common_frame(arm, target)
        for other, tip in self.last_tip.items():
            if other == arm:
                continue
            distance = math.hypot(mine[0] - tip[0], mine[1] - tip[1])
            if distance < self.min_separation_m:
                return (
                    f"Rejected: that target is {distance * 100:.0f} cm horizontally from the "
                    f"{other} arm's gripper (minimum {self.min_separation_m * 100:.0f} cm). "
                    f"Move the {other} arm away first."
                )
        return None

    def _images(self, reply: dict, arm: str | None = None) -> list[dict]:
        arm = arm or self.default_arm
        frames_root = self.arms[arm]["frames_root"]
        images = []
        for raw_path in reply.get("frames", []):
            path = Path(raw_path).resolve()
            if not path.is_relative_to(frames_root) or path.suffix.lower() != ".jpg":
                raise RuntimeError("Harness returned an unexpected frame path")
            camera = next(
                (n for n in ("wrist", "side") if path.stem.endswith("_" + n)), None
            )
            if camera is None:
                continue
            payload = path.read_bytes()
            name = (
                f"{self.count:04d}_{arm}_{camera}.jpg"
                if self.multi_arm
                else f"{self.count:04d}_{camera}.jpg"
            )
            (self.output / "observations" / name).write_bytes(payload)
            image = {
                "name": name,
                "camera": camera,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "base64": base64.b64encode(payload).decode(),
            }
            if self.multi_arm:
                image["arm"] = arm
            images.append(image)
        return images

    def execute(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError("A unique request_id is required")
        command, args = validate_command(
            request.get("command"), request.get("args", [])
        )
        arm = request.get("arm")
        if arm is None:
            arm = self.default_arm
        elif not isinstance(arm, str) or arm not in self.arms:
            raise ValueError(f"Unknown arm; choose one of {sorted(self.arms)}")
        fingerprint = json.dumps([arm, command, args])
        if not self.lock.acquire(blocking=False):
            return {
                "ok": False,
                "error": "Another command is in flight; commands are never queued",
            }
        try:
            if request_id in self.receipts:
                previous, receipt = self.receipts[request_id]
                if previous != fingerprint:
                    raise ValueError("request_id was reused for a different command")
                return receipt
            if self.closed or self.halted or (self.output / "STOP").exists():
                return {
                    "ok": False,
                    "error": "Trial closed or stopped; no new commands accepted",
                }
            if self.started is None:
                return {"ok": False, "error": "Trial not activated"}
            if (
                time.monotonic() - self.started > self.timeout
                or self.count >= self.max_commands
            ):
                self.halted = True
                self.halt_reason = "budget_exhausted"
                return {"ok": False, "error": "Trial budget exhausted"}
            is_motion = command in {"tip", "goto", "gripper", "rest"}
            if is_motion and not self.allow_motion:
                return {"ok": False, "error": "Read-only trial: motion disabled"}
            if command == "tip":
                rejection = self.separation_rejection(arm, [float(v) for v in args[:3]])
                if rejection:
                    self.event(
                        "command_rejected",
                        request_id=request_id,
                        arm=arm,
                        command=command,
                        args=args,
                        reason="arm_separation",
                    )
                    return {"ok": False, "error": rejection, "arm_name": arm}
            if not self.recording_healthy():
                self.halted = True
                self.halt_reason = "recording_lost"
                return {"ok": False, "error": "Camera recording lost; trial halted"}
            self.count += 1
            self.event(
                "command_started",
                request_id=request_id,
                arm=arm,
                command=command,
                args=args,
            )
            receipt: dict[str, Any]
            try:
                transport = self.arms[arm]["transport"]
                reply = transport("observe" if command == "finish" else command, args)
                if reply.get("arm") and reply["arm"] != self.robot_id:
                    raise RuntimeError("Connected arm identity changed")
                images = self._images(reply, arm)
                if (
                    (command in {"observe", "finish"} or is_motion)
                    and reply.get("ok")
                    and {image["camera"] for image in images} != {"wrist", "side"}
                ):
                    raise RuntimeError("Both post-command camera frames are required")
                self.note_state(arm, reply)
                if command == "finish" and self.multi_arm:
                    # Final evidence from every arm, not only the addressed one.
                    reply = dict(
                        reply,
                        arms={
                            arm: {
                                k: v
                                for k, v in reply.items()
                                if k
                                in {
                                    "joints",
                                    "tip_m",
                                    "pitch_deg",
                                    "clearance_mm",
                                    "armed",
                                }
                            }
                        },
                    )
                    for other, spec in self.arms.items():
                        if other == arm:
                            continue
                        other_reply = spec["transport"]("observe", [])
                        images.extend(self._images(other_reply, other))
                        self.note_state(other, other_reply)
                        reply["arms"][other] = {
                            k: v
                            for k, v in other_reply.items()
                            if k
                            in {"joints", "tip_m", "pitch_deg", "clearance_mm", "armed"}
                        }
                if reply.get("abort") or reply.get("faults"):
                    self.halted = True
                    self.halt_reason = "hardware_abort"
                receipt = {
                    k: v
                    for k, v in reply.items()
                    # Strip server-side safety-guard fields from the agent-facing
                    # observation: the harness still enforces zones, but the tester
                    # must stay blind to them (blind the tester, not the controller).
                    if k not in {"frames", "trace", "text", "zones", "surface"}
                }
                receipt["text"] = "\n".join(
                    line
                    for line in reply.get("text", "").splitlines()
                    if not line.startswith("frames:")
                )
                receipt.update(request_id=request_id, images=images)
                if self.multi_arm:
                    receipt["arm_name"] = arm
                if command == "finish":
                    self.closed = True
            except Exception as exc:
                # Never retry a command whose physical execution is uncertain.
                self.halted = True
                self.halt_reason = "uncertain_outcome"
                receipt = {
                    "ok": False,
                    "request_id": request_id,
                    "error": "Command outcome uncertain; operator inspection required",
                    "diagnostic_type": type(exc).__name__,
                }
                self.event("transport_failure", diagnostic_type=type(exc).__name__)
            logged = {
                **receipt,
                "images": [
                    {k: v for k, v in im.items() if k != "base64"}
                    for im in receipt.get("images", [])
                ],
            }
            self.event("command_finished", receipt=logged)
            self.receipts[request_id] = fingerprint, receipt
            return receipt
        finally:
            self.lock.release()

    def serve(self, bind: str = "127.0.0.1", port: int = 0) -> int:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Authorization headers and request bodies never go to access logs.

            def do_POST(self):
                status = 200
                try:
                    self.connection.settimeout(5)
                    auth = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(auth, "Bearer " + bridge.token):
                        status, response = 403, {"ok": False, "error": "Unauthorized"}
                    elif self.path != "/command":
                        status, response = (
                            404,
                            {"ok": False, "error": "Unknown endpoint"},
                        )
                    else:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 4096:
                            raise ValueError("Invalid request size")
                        response = bridge.execute(json.loads(self.rfile.read(length)))
                except (ValueError, TypeError, TimeoutError):
                    status, response = 400, {"ok": False, "error": "Invalid request"}
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(payload)

        self.server = ThreadingHTTPServer((bind, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_port

    def close(self) -> None:
        # Wait for the bounded transport call; do not claim an in-flight move was cancelled.
        with self.lock:
            self.closed = True
            self.event(
                "closed",
                halted=self.halted,
                halt_reason=self.halt_reason,
                commands=self.count,
            )
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
