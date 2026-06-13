#!/usr/bin/env python3
"""Original Cua SDK runner for the local computer-use smoke task."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


async def run_original_async(task_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    descriptor = json.loads((task_dir / "computer-use-task.json").read_text())
    expected = descriptor["expected_result"]

    from cua_sandbox import Image, Sandbox

    image = Image.linux(
        distro=os.environ.get("BENCHFLOW_CUA_LINUX_DISTRO", "ubuntu"),
        version=os.environ.get("BENCHFLOW_CUA_LINUX_VERSION", "24.04"),
        kind=os.environ.get("BENCHFLOW_CUA_LINUX_KIND", "container"),
    )
    name = f"computer-use-smoke-original-{uuid.uuid4().hex[:8]}"
    sandbox = await Sandbox.create(
        image,
        name=name,
        local=True,
        api_key=os.environ.get("CUA_API_KEY") or None,
        telemetry_enabled=False,
    )
    try:
        setup = await sandbox.shell.run(
            "sudo -n /bin/sh -c 'mkdir -p /app /logs/artifacts && "
            "chown -R cua:cua /app /logs' || "
            "mkdir -p /home/cua/computer-use-smoke",
            timeout=30,
        )
        if _return_code(setup) != 0:
            raise RuntimeError(f"original setup failed: {setup.stderr or setup.stdout}")

        script = (
            "cat > /app/computer_use_result.txt <<'EOF'\n"
            f"{expected}\n"
            "EOF\n"
            "cp /app/computer_use_result.txt /app/computer_use_roundtrip.txt\n"
            "cat /app/computer_use_roundtrip.txt\n"
        )
        shell_result = await sandbox.shell.run(script, timeout=30)
        final_result = (shell_result.stdout or "").strip()
        dimensions = await sandbox.get_dimensions()
        screenshot = await sandbox.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode()
        passed = final_result == expected and len(screenshot) > 0
        return {
            "framework": "computer-use-smoke-original",
            "task_id": descriptor["task_id"],
            "final_result": final_result,
            "score": 1.0 if passed else 0.0,
            "steps": [
                {"action": "create_cua_sandbox", "name": name},
                {"action": "write_file", "path": descriptor["expected_file"]},
                {"action": "read_file", "value": final_result},
                {"action": "screenshot", "bytes": len(screenshot)},
            ],
            "screenshots_b64": [screenshot_b64],
            "dimensions": list(dimensions),
            "num_steps": 4,
            "duration_sec": round(time.perf_counter() - started, 6),
            "error": None if passed else "expected result or screenshot missing",
        }
    finally:
        destroy = getattr(sandbox, "destroy", None)
        if destroy is not None:
            await destroy()


def _return_code(result: Any) -> int:
    value = getattr(result, "returncode", None)
    if isinstance(value, int):
        return value
    value = getattr(result, "return_code", None)
    if isinstance(value, int):
        return value
    return int(getattr(result, "exit_code", 0) or 0)


def run_original(task_dir: Path) -> dict[str, Any]:
    return asyncio.run(run_original_async(task_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run_original(args.task_dir)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
