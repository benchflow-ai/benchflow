"""A fake ``docker`` executable for provider tests that must not need a daemon.

``install(tmp_path, monkeypatch)`` puts a ``docker`` script first on ``PATH``.
Every call appends one JSON line to ``calls.jsonl``: the argv after the
docker CLI's global flags (``-H``, ``--context``, ``--tlsverify`` ...), those
flags, the ``DOCKER_HOST`` / ``DOCKER_CONTEXT`` / ``DOCKER_TLS_VERIFY`` /
``DOCKER_CERT_PATH`` it targets (from a flag or its environment), and the
connection variables of its environment (``env``). Its answers come from ``script.json``, which a test edits through
:class:`FakeDocker`:

- ``fail``: ``{"<first word>": "<stderr text>"}`` makes every call whose
  subcommand (``info``, ``build``, ``up``, ``down``, ``exec``, ``ps``, ...)
  matches exit 1 with that text on stderr.
- ``info``: the JSON document ``docker info --format '{{json .}}'`` prints.
- ``leftovers``: ids ``docker ps/network ls/volume ls --filter label=...``
  list, per kind (``containers``, ``networks``, ``volumes``); ``rm`` calls
  remove them, so a second listing is empty.
- ``project``: what ``compose config --format json`` prints.
- ``config_stderr``: text ``compose config`` writes to stderr first (Compose's
  ``level=warning`` lines).
- ``volumes_on_host``: ``{"<name>": <volume inspect object>}`` for
  ``volume inspect``; any other name is "no such volume".
- ``built_images``: the lines ``image ls`` prints; ``image rm`` records its
  references in ``removed_images``.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

_SCRIPT = r"""#!{python}
import json, os, sys
from pathlib import Path

HERE = Path({here!r})


def save(script):
    # Atomic: a call running at the same time never reads a half-written file.
    tmp = HERE / ("script.json.%d" % os.getpid())
    tmp.write_text(json.dumps(script))
    os.replace(tmp, HERE / "script.json")

argv = sys.argv[1:]
# The docker CLI's global flags (docker -H URL --tlsverify ... compose ...):
# a flag names the daemon as its variable would.
global_flags = {{}}
global_argv = []
while argv and argv[0].startswith("-"):
    flag = argv.pop(0)
    global_argv.append(flag)
    if flag in ("--tls", "--tlsverify"):
        global_flags[flag] = True
    else:
        global_flags[flag] = argv.pop(0)
        global_argv.append(global_flags[flag])
host = global_flags.get("-H", global_flags.get("--host"))
with (HERE / "calls.jsonl").open("a") as fh:
    fh.write(json.dumps({{
        "argv": argv,
        "global": global_argv,
        "host": host,
        "PATH": os.environ.get("PATH"),
        "BUILDX_BUILDER": os.environ.get("BUILDX_BUILDER"),
        "SSH_AUTH_SOCK": os.environ.get("SSH_AUTH_SOCK"),
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG"),
        "USER": os.environ.get("USER"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "DOCKER_CONTEXT": os.environ.get("DOCKER_CONTEXT"),
        "DOCKER_TLS_VERIFY": os.environ.get("DOCKER_TLS_VERIFY"),
        "DOCKER_CERT_PATH": os.environ.get("DOCKER_CERT_PATH"),
        "env": {{k: os.environ.get(k) for k in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "SSH_AUTH_SOCK")}},
    }}) + "\n")
script = json.loads((HERE / "script.json").read_text())

words = list(argv)
if words and words[0] == "compose":
    # Skip compose's own flags to reach its subcommand.
    rest = words[1:]
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("--project-name", "-p", "--project-directory", "-f"):
            rest.pop(0)
    words = rest
sub = words[0] if words else ""
if sub in ("network", "volume", "container", "image") and len(words) > 1:
    sub = sub + " " + words[1]

fail = script.get("fail", {{}})
for key in (sub, sub.split(" ")[0]):
    if key in fail:
        sys.stderr.write(fail[key] + "\n")
        sys.exit(1)

left = script.setdefault("leftovers", {{}})
if sub == "info":
    print(json.dumps(script.get("info", {{}})))
elif sub == "ps" and "-aq" in words:
    print("\n".join(left.get("containers", [])))
elif words == ["ps", "--all", "-q"]:
    # Container discovery requires a real-shaped id and service/mount record.
    print("c0ffee000000")
elif words == ["inspect", "c0ffee000000"]:
    print(json.dumps([{{"State": {{"Running": script.get("running", False), "ExitCode": 0}}, "Config": {{"Entrypoint": [], "Cmd": ["/bin/sleep", "infinity"], "Env": [], "Labels": {{"com.docker.compose.service": "main"}}}}, "Mounts": []}}]))
elif sub == "ps" and "--filter" in words:
    # The raw label lookup (docker ps -q --no-trunc --filter ...) gets a
    # full id; compose ps keeps the short non-id below, so exec stays on
    # docker compose.
    print("c0ffee" * 10 + "0c0f")
elif sub == "ps":
    print("c0ffee000000")
elif sub in ("network ls", "network prune"):
    if sub == "network ls":
        print("\n".join(left.get("networks", [])))
    else:
        left["networks"] = []
elif sub == "volume ls":
    print("\n".join(left.get("volumes", [])))
elif sub == "config":
    # Compose writes its warnings to stderr ahead of the project.
    sys.stderr.write(script.get("config_stderr", ""))
    sys.stderr.flush()
    print(json.dumps(script.get("project", {{"services": {{"main": {{}}}}}})))
elif sub == "volume inspect":
    existing = script.get("volumes_on_host", {{}})
    found = [existing[name] for name in words[2:] if name in existing]
    print(json.dumps(found))
    missing = [name for name in words[2:] if name not in existing]
    if missing:
        sys.stderr.write("Error response from daemon: get " + missing[0] + ": no such volume\n")
        save(script)
        sys.exit(1)
elif sub == "image ls":
    print("\n".join(script.get("built_images", [])))
elif sub == "image rm":
    script["removed_images"] = script.get("removed_images", []) + words[2:]
elif sub == "rm":
    left["containers"] = [c for c in left.get("containers", []) if c not in words]
elif sub == "network rm":
    left["networks"] = [n for n in left.get("networks", []) if n not in words]
elif sub == "volume rm":
    left["volumes"] = [v for v in left.get("volumes", []) if v not in words]
elif sub in ("create", "up", "down"):
    script["running"] = sub == "up"
save(script)
sys.exit(0)
"""

_INFO = {"NCPU": 4, "MemTotal": 8 * 1024**3, "ServerVersion": "28.3.3"}


class FakeDocker:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def script(self) -> dict[str, Any]:
        return json.loads((self.root / "script.json").read_text())

    def set(self, **values: Any) -> None:
        script = self.script
        script.update(values)
        (self.root / "script.json").write_text(json.dumps(script))

    def calls(self) -> list[dict[str, Any]]:
        path = self.root / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def argvs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls()]


def install(tmp_path: Path, monkeypatch: Any) -> FakeDocker:
    root = tmp_path / "fake-docker"
    root.mkdir()
    exe = root / "docker"
    exe.write_text(_SCRIPT.format(python=sys.executable, here=str(root)))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (root / "script.json").write_text(json.dumps({"info": _INFO}))
    monkeypatch.setenv("PATH", f"{root}{os.pathsep}{os.environ.get('PATH', '')}")
    return FakeDocker(root)
