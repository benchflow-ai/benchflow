"""Independent host camera sidecar with recoverable MJPEG and timestamp logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .bridge import write_json


def normalize_jpeg(payload: bytes) -> bytes:
    """Strip USB camera zero padding after JPEG EOI, rejecting truncated frames."""
    end = payload.rfind(b"\xff\xd9")
    if (
        len(payload) > 8_000_000
        or not payload.startswith(b"\xff\xd8")
        or end < 2
        or payload[end + 2 :].strip(b"\x00")
    ):
        raise ValueError("Not a complete JPEG frame")
    return payload[: end + 2]


def record(output: Path, cameras: dict[str, str], fps: float) -> None:
    if not math.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("fps must be between 1 and 30")
    if not {"wrist", "side"} <= set(cameras):
        raise ValueError("Both wrist and side feeds are required")
    output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.monotonic()
    stats: dict[str, dict] = {}

    def camera(name: str, url: str):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        count = errors = 0
        last_frame = None
        maximum_gap = 0.0
        deadline = time.monotonic()
        with (
            (output / f"{name}.mjpeg").open("wb", buffering=0) as video,
            (output / f"{name}.jsonl").open("w", buffering=1) as index,
        ):
            while not stop.is_set() and not (output / "STOP").exists():
                stop.wait(max(0, deadline - time.monotonic()))
                if stop.is_set():
                    break
                try:
                    with opener.open(url, timeout=2) as response:
                        payload = response.read(8_000_001)
                    payload = normalize_jpeg(payload)
                    now = time.monotonic()
                    maximum_gap = max(
                        maximum_gap, now - last_frame if last_frame else now - started
                    )
                    offset = video.tell()
                    video.write(payload)
                    count += 1
                    last_frame = now
                    index.write(
                        json.dumps(
                            {
                                "frame": count,
                                "monotonic": now,
                                "utc_epoch": time.time(),
                                "offset": offset,
                                "length": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        )
                        + "\n"
                    )
                except Exception as exc:
                    errors += 1
                    index.write(
                        json.dumps(
                            {
                                "error_type": type(exc).__name__,
                                "monotonic": time.monotonic(),
                            }
                        )
                        + "\n"
                    )
                stats[name] = {
                    "frames": count,
                    "errors": errors,
                    "last_frame": last_frame,
                    "maximum_gap_s": maximum_gap,
                }
                write_json(output / f"{name}-status.json", stats[name])
                deadline += 1 / fps
                if deadline < time.monotonic() - 1 / fps:
                    deadline = time.monotonic()

    workers = [
        threading.Thread(target=camera, args=(name, url))
        for name, url in cameras.items()
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    write_json(
        output / "capture-summary.json",
        {
            "fps": fps,
            "started_monotonic": started,
            "ended_monotonic": time.monotonic(),
            "cameras": stats,
        },
    )


class RecordingSidecar:
    def __init__(self, output: Path, cameras: dict[str, str], fps: float = 5):
        self.output, self.cameras, self.fps = output, cameras, fps
        self.process: subprocess.Popen | None = None
        self.console = None

    def start(self) -> None:
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.console = (self.output.parent / "recorder.log").open("w")
        # A separate process survives CLI/agent process exits and steered agent turns.
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchflow.robotics.recording",
                str(self.output),
                "--cameras",
                json.dumps(self.cameras),
                "--fps",
                str(self.fps),
            ],
            stdout=self.console,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Camera sidecar exited; inspect recorder.log")
            if self.healthy():
                return
            time.sleep(0.1)
        raise RuntimeError("Both camera feeds did not become ready")

    def healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            return all(
                (
                    status := json.loads(
                        (self.output / f"{name}-status.json").read_text()
                    )
                )["frames"]
                >= 2
                and status["last_frame"] is not None
                and time.monotonic() - status["last_frame"] < 4
                for name in self.cameras
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def stop(self) -> dict:
        if self.output.exists():
            (self.output / "STOP").touch()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if self.console:
            self.console.close()
        summary_path = self.output / "capture-summary.json"
        if not summary_path.exists():
            return {
                "complete": False,
                "error": "Missing capture summary; raw MJPEG remains recoverable",
            }
        summary = json.loads(summary_path.read_text())
        summary["complete"] = set(summary["cameras"]) == set(self.cameras) and all(
            s["frames"] >= 2
            and s["errors"] == 0
            and s["maximum_gap_s"] < 3
            and summary["ended_monotonic"] - s["last_frame"] < 3
            for s in summary["cameras"].values()
        )
        summary["exports"] = {}
        for name in self.cameras:
            output = self.output / f"{name}.mp4"
            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-f",
                        "mjpeg",
                        "-framerate",
                        str(self.fps),
                        "-i",
                        str(self.output / f"{name}.mjpeg"),
                        "-c:v",
                        "libx264",
                        "-threads",
                        "2",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(output),
                    ],
                    capture_output=True,
                    timeout=600,
                    check=True,
                )
                summary["exports"][name] = {
                    "mp4": str(output),
                    "ok": result.returncode == 0,
                }
            except (OSError, subprocess.SubprocessError) as exc:
                summary["exports"][name] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                }
        write_json(self.output / "summary.json", summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args()
    record(args.output, json.loads(args.cameras), args.fps)
