#!/usr/bin/env python3
"""Original Cua SDK runner for cookbook desktop smoke slices."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shlex
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


async def run_original_async(task_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    expected = "setup-ok"
    osworld_path = task_dir / "tests" / "osworld_task.json"
    osworld_task = (
        json.loads(osworld_path.read_text()) if osworld_path.is_file() else None
    )
    task_id = (
        str(osworld_task["id"])
        if isinstance(osworld_task, dict) and osworld_task.get("id")
        else task_dir.name
    )
    setup_steps, setup_marker, framework = _setup_plan(task_dir, osworld_task)

    from cua_sandbox import Image, Sandbox

    image = Image.linux(
        distro=os.environ.get("BENCHFLOW_CUA_LINUX_DISTRO", "ubuntu"),
        version=os.environ.get("BENCHFLOW_CUA_LINUX_VERSION", "24.04"),
        kind=os.environ.get("BENCHFLOW_CUA_LINUX_KIND", "container"),
    )
    name = f"use-computer-cookbook-original-{uuid.uuid4().hex[:8]}"
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
            "chown -R cua:cua /app /logs && mkdir -p /home/user && "
            "chmod 777 /home/user' || mkdir -p /home/cua/osworld-smoke /home/user",
            timeout=30,
        )
        if _return_code(setup) != 0:
            raise RuntimeError(f"original setup failed: {setup.stderr or setup.stdout}")

        cuagym_original = _cuagym_original_dir(task_dir)
        if cuagym_original is not None:
            return await _run_cuagym_original(
                sandbox,
                cuagym_original,
                started=started,
                sandbox_name=name,
            )

        for command in setup_steps:
            result = await sandbox.shell.run(command, timeout=30)
            if _return_code(result) != 0:
                raise RuntimeError(
                    f"cookbook setup command failed: {result.stderr or result.stdout}"
                )

        read_setup = await sandbox.shell.run(
            f"cat {shlex.quote(setup_marker)}",
            timeout=30,
        )
        setup_output = (read_setup.stdout or "").strip()
        script = (
            "cat > /app/computer_use_result.txt <<'EOF'\n"
            f"{setup_output}\n"
            "EOF\n"
            "cp /app/computer_use_result.txt /app/computer_use_roundtrip.txt\n"
            "cat /app/computer_use_roundtrip.txt\n"
        )
        shell_result = await sandbox.shell.run(script, timeout=30)
        final_result = (shell_result.stdout or "").strip()
        dimensions = await sandbox.get_dimensions()
        screenshot = await sandbox.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode()
        passed = (
            setup_output == expected
            and final_result == expected
            and len(screenshot) > 0
        )
        return {
            "framework": framework,
            "task_id": task_id,
            "final_result": final_result,
            "setup_output": setup_output,
            "score": 1.0 if passed else 0.0,
            "steps": [
                {"action": "create_cua_sandbox", "name": name},
                {"action": "task_setup", "commands": setup_steps},
                {"action": "read_setup", "value": setup_output},
                {"action": "write_file", "path": "/app/computer_use_result.txt"},
                {"action": "screenshot", "bytes": len(screenshot)},
            ],
            "screenshots_b64": [screenshot_b64],
            "dimensions": list(dimensions),
            "num_steps": 5,
            "duration_sec": round(time.perf_counter() - started, 6),
            "error": None if passed else "setup, final result, or screenshot missing",
        }
    finally:
        destroy = getattr(sandbox, "destroy", None)
        if destroy is not None:
            await destroy()


def _setup_plan(
    task_dir: Path,
    osworld_task: dict[str, Any] | None,
) -> tuple[list[str], str, str]:
    if osworld_task is None:
        setup_script = task_dir / "tests" / "setup" / "pre_command.sh"
        if setup_script.is_file():
            script = setup_script.read_text()
            if "runner-cuagym-setup-ok" in script:
                return (
                    [f"bash -lc {shlex.quote(script)}"],
                    "/tmp/runner-cuagym-setup-ok",
                    "use-computer-cookbook-cuagym-original",
                )
        return (
            ["printf 'setup-ok\\n' > /tmp/runner-cookbook-setup-ok"],
            "/tmp/runner-cookbook-setup-ok",
            "use-computer-cookbook-original",
        )

    return (
        _osworld_setup_commands(osworld_task),
        "/tmp/runner-osworld-setup-ok",
        "use-computer-cookbook-osworld-original",
    )


def _cuagym_original_dir(task_dir: Path) -> Path | None:
    original = task_dir / "tests" / "cuagym" / "original"
    if (original / "task.json").is_file() and (original / "reward.py").is_file():
        return original
    return None


async def _run_cuagym_original(
    sandbox: Any,
    original_dir: Path,
    *,
    started: float,
    sandbox_name: str,
) -> dict[str, Any]:
    task_json = json.loads((original_dir / "task.json").read_text())
    task_id = str(task_json.get("id") or original_dir.name)
    remote_task_dir = f"/tmp/cuagym/tasks/{task_id}"

    upload = await sandbox.shell.run(
        "sudo -n /bin/sh -c 'mkdir -p /tmp/cuagym/tasks /home/user /logs/artifacts /app "
        "&& chmod 777 /home/user /app /logs/artifacts' || "
        "mkdir -p /tmp/cuagym/tasks /home/user /logs/artifacts /app",
        timeout=30,
    )
    if _return_code(upload) != 0:
        raise RuntimeError(f"original CUA-Gym mkdir failed: {upload.stderr or upload.stdout}")
    await _upload_dir_to_sandbox(sandbox, original_dir, remote_task_dir)
    setup_output = await _run_cuagym_setup(sandbox, remote_task_dir)

    final_result = "observed"
    shell_result = await sandbox.shell.run(
        "printf 'observed\\n' > /app/computer_use_result.txt && "
        "cp /app/computer_use_result.txt /app/computer_use_roundtrip.txt && "
        "cat /app/computer_use_roundtrip.txt",
        timeout=30,
    )
    final_result = (shell_result.stdout or "").strip() or final_result
    dimensions = await sandbox.get_dimensions()
    screenshot = await sandbox.screenshot()
    screenshot_b64 = base64.b64encode(screenshot).decode()
    postconfig_output = await _run_cuagym_postconfig(sandbox, remote_task_dir)
    await _run_cuagym_reward_dependencies(sandbox, remote_task_dir)
    reward_command = f"cd {shlex.quote(remote_task_dir)} && HOME=/home/user python3 reward.py"
    reward_output = await sandbox.shell.run(
        f"sudo -n /bin/sh -c {shlex.quote(reward_command)} || {reward_command}",
        timeout=120,
    )
    reward_text = (reward_output.stdout or "") + (reward_output.stderr or "")
    reward = _extract_reward(reward_text)
    return {
        "framework": "use-computer-cookbook-cuagym-python-original",
        "task_id": task_id,
        "final_result": final_result,
        "setup_output": setup_output,
        "score": reward,
        "steps": [
            {"action": "create_cua_sandbox", "name": sandbox_name},
            {"action": "upload_cuagym_task", "path": remote_task_dir},
            {"action": "task_setup", "value": setup_output},
            {"action": "write_file", "path": "/app/computer_use_result.txt"},
            {"action": "postconfig", "value": postconfig_output},
            {"action": "reward_py", "reward": reward},
            {"action": "screenshot", "bytes": len(screenshot)},
        ],
        "screenshots_b64": [screenshot_b64],
        "dimensions": list(dimensions),
        "num_steps": 6,
        "duration_sec": round(time.perf_counter() - started, 6),
        "error": None if 0.0 <= reward <= 1.0 else "reward.py did not emit reward",
        "reward_stdout_tail": reward_text[-2000:],
    }


async def _upload_dir_to_sandbox(sandbox: Any, source_dir: Path, target_dir: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as tar:
            for child in sorted(path for path in source_dir.rglob("*") if path.is_file()):
                tar.add(child, arcname=child.relative_to(source_dir).as_posix())
        archive.seek(0)
        payload = base64.b64encode(archive.read()).decode()
    inner = (
        "HOME=/home/user DISPLAY=:1 XAUTHORITY=/home/cua/.Xauthority python3 - <<'PY'\n"
        "import base64, io, pathlib, tarfile\n"
        f"target = pathlib.Path({target_dir!r})\n"
        "target.mkdir(parents=True, exist_ok=True)\n"
        f"payload = base64.b64decode({payload!r})\n"
        "with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:\n"
        "    tar.extractall(target)\n"
        "PY"
    )
    script = f"sudo -n /bin/sh -c {shlex.quote(inner)} || {inner}"
    result = await sandbox.shell.run(script, timeout=120)
    if _return_code(result) != 0:
        raise RuntimeError(f"original CUA-Gym upload failed: {result.stderr or result.stdout}")


async def _run_cuagym_setup(sandbox: Any, remote_task_dir: str) -> str:
    inner = (
        "python3 - <<'PY'\n"
        "import json, os, pathlib, shlex, shutil, subprocess, sys, time\n"
        f"task_dir = pathlib.Path({remote_task_dir!r})\n"
        "task_json = json.loads((task_dir / 'task.json').read_text())\n"
        "def source_path(source):\n"
        "    if source.startswith('./'):\n"
        "        return task_dir / source[2:]\n"
        "    return task_dir / pathlib.Path(source).name\n"
        "def run_command(command):\n"
        "    env = os.environ.copy(); env.setdefault('DISPLAY', ':1'); env.setdefault('HOME', '/home/user')\n"
        "    if isinstance(command, str):\n"
        "        subprocess.run(command, shell=True, check=True, env=env)\n"
        "    elif isinstance(command, list) and command:\n"
        "        subprocess.run([str(part) for part in command], check=True, env=env)\n"
        "def launch_command(command):\n"
        "    env = os.environ.copy(); env.setdefault('DISPLAY', ':1'); env.setdefault('HOME', '/home/user')\n"
        "    if isinstance(command, str):\n"
        "        argv = shlex.split(command)\n"
        "    elif isinstance(command, list) and command:\n"
        "        argv = [str(part) for part in command]\n"
        "    else:\n"
        "        return\n"
        "    if argv:\n"
        "        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)\n"
        "for step in task_json.get('config') or []:\n"
        "    kind = step.get('type'); params = step.get('parameters') or {}\n"
        "    if kind == 'download':\n"
        "        for item in params.get('files') or []:\n"
        "            src = source_path(str(item.get('url') or ''))\n"
        "            dst = pathlib.Path(str(item.get('path') or ''))\n"
        "            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)\n"
        "    elif kind in {'execute', 'command'}:\n"
        "        run_command(params.get('command') or [])\n"
        "    elif kind == 'launch':\n"
        "        launch_command(params.get('command') or [])\n"
        "    elif kind == 'sleep':\n"
        "        time.sleep(float(params.get('seconds', 1)))\n"
        "    else:\n"
        "        raise RuntimeError(f'unsupported setup step: {kind}')\n"
        "print('setup-ok')\n"
        "PY"
    )
    script = f"sudo -n /bin/sh -c {shlex.quote(inner)} || {inner}"
    result = await sandbox.shell.run(script, timeout=120)
    if _return_code(result) != 0:
        raise RuntimeError(f"original CUA-Gym setup failed: {result.stderr or result.stdout}")
    return (result.stdout or "").strip()


async def _run_cuagym_postconfig(sandbox: Any, remote_task_dir: str) -> str:
    inner = (
        "python3 - <<'PY'\n"
        "import json, os, pathlib, subprocess, time\n"
        "from Xlib import X, XK, display\n"
        "from Xlib.ext import xtest\n"
        f"task_dir = pathlib.Path({remote_task_dir!r})\n"
        "task_json = json.loads((task_dir / 'task.json').read_text())\n"
        "postconfig = (task_json.get('evaluator') or {}).get('postconfig') or []\n"
        "def send_save_hotkey():\n"
        "    d = display.Display(':1')\n"
        "    ctrl = d.keysym_to_keycode(XK.string_to_keysym('Control_L'))\n"
        "    s_key = d.keysym_to_keycode(XK.string_to_keysym('s'))\n"
        "    xtest.fake_input(d, X.KeyPress, ctrl)\n"
        "    xtest.fake_input(d, X.KeyPress, s_key)\n"
        "    xtest.fake_input(d, X.KeyRelease, s_key)\n"
        "    xtest.fake_input(d, X.KeyRelease, ctrl)\n"
        "    d.sync()\n"
        "for step in postconfig:\n"
        "    kind = step.get('type'); params = step.get('parameters') or {}\n"
        "    if kind == 'sleep':\n"
        "        time.sleep(float(params.get('seconds', 1)))\n"
        "    elif kind in {'execute', 'command'}:\n"
        "        command = params.get('command') or []\n"
        "        if command in [\n"
        "            ['python', '-c', 'import pyautogui; pyautogui.hotkey(\"ctrl\", \"s\");'],\n"
        "            ['python3', '-c', 'import pyautogui; pyautogui.hotkey(\"ctrl\", \"s\");'],\n"
        "        ]:\n"
        "            send_save_hotkey()\n"
        "        else:\n"
        "            env = os.environ.copy(); env['DISPLAY'] = ':1'; env['XAUTHORITY'] = '/home/cua/.Xauthority'; env['HOME'] = '/home/user'\n"
        "            subprocess.run(command, check=True, env=env)\n"
        "    elif kind:\n"
        "        raise RuntimeError(f'unsupported postconfig step: {kind}')\n"
        "print('postconfig-ok' if postconfig else 'postconfig-none')\n"
        "PY"
    )
    script = f"sudo -n /bin/sh -c {shlex.quote(inner)} || {inner}"
    result = await sandbox.shell.run(script, timeout=120)
    if _return_code(result) != 0:
        raise RuntimeError(
            f"original CUA-Gym postconfig failed: {result.stderr or result.stdout}"
        )
    return (result.stdout or "").strip()


async def _run_cuagym_reward_dependencies(sandbox: Any, remote_task_dir: str) -> str:
    inner = (
        "python3 - <<'PY'\n"
        "import ast, importlib, pathlib, subprocess, sys\n"
        "dependency_packages = {'PIL': 'Pillow', 'PyPDF2': 'PyPDF2', 'docx': 'python-docx', 'gimpformats': 'gimpformats', 'numpy': 'numpy', 'odf': 'odfpy', 'openpyxl': 'openpyxl', 'pandas': 'pandas', 'pptx': 'python-pptx', 'pyperclip': 'pyperclip'}\n"
        f"source = (pathlib.Path({remote_task_dir!r}) / 'reward.py').read_text(errors='replace')\n"
        "tree = ast.parse(source)\n"
        "imports = set()\n"
        "for node in ast.walk(tree):\n"
        "    if isinstance(node, ast.Import):\n"
        "        for alias in node.names:\n"
        "            imports.add(alias.name.split('.', 1)[0])\n"
        "    elif isinstance(node, ast.ImportFrom) and node.module:\n"
        "        imports.add(node.module.split('.', 1)[0])\n"
        "for import_name in sorted(imports):\n"
        "    package = dependency_packages.get(import_name)\n"
        "    if package is None:\n"
        "        continue\n"
        "    try:\n"
        "        importlib.import_module(import_name)\n"
        "    except ImportError:\n"
        "        subprocess.check_call([\n"
        "            sys.executable,\n"
        "            '-m',\n"
        "            'pip',\n"
        "            'install',\n"
        "            '--quiet',\n"
        "            '--disable-pip-version-check',\n"
        "            '--root-user-action=ignore',\n"
        "            package,\n"
        "        ])\n"
        "print('reward-dependencies-ok')\n"
        "PY"
    )
    script = f"sudo -n /bin/sh -c {shlex.quote(inner)} || {inner}"
    result = await sandbox.shell.run(script, timeout=180)
    if _return_code(result) != 0:
        raise RuntimeError(
            f"original CUA-Gym reward dependency setup failed: {result.stderr or result.stdout}"
        )
    return (result.stdout or "").strip()


def _extract_reward(output: str) -> float:
    import re

    matches = re.findall(r"REWARD\s*[:=]\s*([0-9]*\.?[0-9]+)", output, re.I)
    if not matches:
        return 0.0
    return max(0.0, min(1.0, float(matches[-1])))


def _osworld_setup_commands(osworld_task: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for step in osworld_task.get("config") or []:
        if not isinstance(step, dict) or step.get("type") not in {
            "execute",
            "command",
        }:
            continue
        params = step.get("parameters") or {}
        command = params.get("command")
        if isinstance(command, list) and all(
            isinstance(part, str) for part in command
        ):
            commands.append(shlex.join(command))
        elif isinstance(command, str):
            commands.append(command)
    return commands or ["printf 'setup-ok\\n' > /tmp/runner-osworld-setup-ok"]


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
