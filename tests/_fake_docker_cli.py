"""A fake ``docker`` executable for provider tests that must not need a daemon.

``install(tmp_path, monkeypatch)`` puts a ``docker`` script first on ``PATH``.
Every call appends one JSON line to ``calls.jsonl``: the argv and the
``DOCKER_HOST`` / ``DOCKER_CONTEXT`` / ``DOCKER_TLS_VERIFY`` / ``DOCKER_CERT_PATH``
it saw. Its answers come from ``script.json``, which a test edits through
:class:`FakeDocker`:

- ``fail``: ``{"<first word>": "<stderr text>"}`` makes every call whose
  subcommand (``info``, ``build``, ``up``, ``down``, ``exec``, ``ps``, ...)
  matches exit 1 with that text on stderr.
- ``info``: the JSON document ``docker info --format '{{json .}}'`` prints.
- ``leftovers``: ids ``docker ps/network ls/volume ls --filter label=...``
  list, per kind (``containers``, ``networks``, ``volumes``); ``rm`` calls
  remove them, so a second listing is empty.
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
argv = sys.argv[1:]
with (HERE / "calls.jsonl").open("a") as fh:
    fh.write(json.dumps({{
        "argv": argv,
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "DOCKER_CONTEXT": os.environ.get("DOCKER_CONTEXT"),
        "DOCKER_TLS_VERIFY": os.environ.get("DOCKER_TLS_VERIFY"),
        "DOCKER_CERT_PATH": os.environ.get("DOCKER_CERT_PATH"),
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
elif sub == "ps":
    print("c0ffee")
elif sub in ("network ls", "network prune"):
    if sub == "network ls":
        print("\n".join(left.get("networks", [])))
    else:
        left["networks"] = []
elif sub == "volume ls":
    print("\n".join(left.get("volumes", [])))
elif sub == "rm":
    left["containers"] = [c for c in left.get("containers", []) if c not in words]
elif sub == "network rm":
    left["networks"] = [n for n in left.get("networks", []) if n not in words]
elif sub == "volume rm":
    left["volumes"] = [v for v in left.get("volumes", []) if v not in words]
(HERE / "script.json").write_text(json.dumps(script))
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
