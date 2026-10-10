"""BenchFlow: run an agent and stream the steps it records in a JSON file.

usage: python3 follow.py FILE ARRAYS DROP -- COMMAND...

Runs COMMAND, printing its stdout and stderr as they come. FILE is the JSON
file the agent rewrites as it works (a glob picks the newest match). Every
two seconds, and once more after the agent exits, each new element of the
file's top-level arrays named in ARRAYS (comma-separated) is printed as one
line, {"@bf": "<array>", "item": <element>}, without the keys named in DROP
(comma-separated, may be empty). After the agent exits, the file's other
top-level values that are not arrays are printed as {"@bf": "info", "item":
{...}}. Exits with the agent's exit status (128 + N when signal N killed it).
Standard library only.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time


def main():
    args = sys.argv[1:]
    split = args.index("--")
    pattern, arrays, drop = (args[:split] + ["", "", ""])[:3]
    command = args[split + 1 :]
    pattern = os.path.expanduser(pattern)
    arrays = [a for a in arrays.split(",") if a]
    drop = {d for d in drop.split(",") if d}
    out = sys.stdout.buffer
    lock = threading.Lock()

    def emit(data):
        with lock:
            out.write(data)
            out.flush()

    def record(name, value):
        text = json.dumps({"@bf": name, "item": value}, separators=(",", ":"), ensure_ascii=False)
        emit(text.encode("utf-8", "replace") + b"\n")

    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)

    def pump():
        for line in iter(proc.stdout.readline, b""):
            emit(line if line.endswith(b"\n") else line + b"\n")

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    sent = dict.fromkeys(arrays, 0)
    seen = [None]

    def newest():
        if not any(c in pattern for c in "*?["):
            return pattern if os.path.isfile(pattern) else None
        best = None
        for path in glob.glob(pattern):
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path)
        return best[1] if best else None

    def poll(final):
        path = newest()
        if path is None:
            return None
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size)
            if key == seen[0] and not final:
                return None
            with open(path, "rb") as f:
                data = json.loads(f.read())
        except (OSError, ValueError):
            # Mid-write; the next poll reads it whole.
            return None
        seen[0] = key
        if not isinstance(data, dict):
            return None
        for name in arrays:
            items = data.get(name)
            if not isinstance(items, list):
                continue
            for item in items[sent[name] :]:
                if drop and isinstance(item, dict):
                    item = {k: v for k, v in item.items() if k not in drop}
                record(name, item)
            sent[name] = max(sent[name], len(items))
        return data

    while proc.poll() is None:
        time.sleep(2)
        poll(False)
    reader.join(10)
    data = poll(True)
    if data is not None:
        record("info", {k: v for k, v in data.items() if k not in arrays and not isinstance(v, list)})
    code = proc.returncode
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    sys.exit(main())
