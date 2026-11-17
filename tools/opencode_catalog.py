"""Build, measure and check BenchFlow's pinned OpenCode model catalog.

A refresh is a reviewed PR that runs, on a machine with git, bun, node and
network::

    git clone https://github.com/anomalyco/models.dev && git -C models.dev checkout <commit>
    python tools/opencode_catalog.py build --models-dev models.dev
    python tools/opencode_catalog.py measure --opencode <opencode-linux-x64 OPENCODE_VERSION>/bin/opencode
    python tools/opencode_catalog.py check

``build`` builds models.dev's ``api.json`` at that commit with its own web
build (``bun install --frozen-lockfile``, ``bun run build`` in
``packages/web``: ``dist/_api.json``, what models.dev and models.opencode.ai
serve), keeps the providers BenchFlow's OpenCode launches read, and writes
the snapshot, its metadata and models.dev's licence (it stops unless that is
MIT). ``measure`` runs the pinned OpenCode over ACP against the snapshot, as
a trial runs it, and writes the Gemini effort table. ``check`` (no network,
no binary) fails when the snapshot, its metadata and the table disagree.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import http.server
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src/benchflow/agents/opencode_catalog"
SNAPSHOT = PACKAGE / "models-dev.json"
METADATA = PACKAGE / "snapshot.json"
EFFORTS = PACKAGE / "gemini-efforts.json"
LICENSE = PACKAGE / "LICENSE.models-dev"
SOURCE = "https://github.com/anomalyco/models.dev"
#: The providers BenchFlow's OpenCode launches read from the catalog: the
#: ``-proxy`` wrapper registers every other route as a gateway alias under
#: its own provider, which the catalog doesn't hold, and points OpenCode's
#: ``google`` provider at the gateway's Gemini pass-through.
PROVIDERS = ("google",)
MIT = "MIT License"
#: A catalog URL nothing listens on: OpenCode's live fetch fails, as on a
#: closed network.
CLOSED_URL = "http://127.0.0.1:9"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_text(api: dict[str, Any], providers: tuple[str, ...] = PROVIDERS) -> str:
    """The snapshot file's text: *api*'s *providers*, keys sorted."""
    missing = [p for p in providers if p not in api]
    if missing:
        raise SystemExit(f"api.json has no provider {', '.join(missing)}")
    return json.dumps({p: api[p] for p in providers}, indent=1, sort_keys=True) + "\n"


def build(models_dev: Path, bun: str) -> None:
    """Snapshot models.dev at the commit *models_dev* has checked out."""
    licence = (models_dev / "LICENSE").read_text(encoding="utf-8")
    if not licence.startswith(MIT):
        raise SystemExit(
            "models.dev's LICENSE is not the MIT licence at this commit: stop, "
            "and ask before shipping its data"
        )
    git = ["git", "-C", str(models_dev)]
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    committed = subprocess.run(
        [*git, "log", "-1", "--format=%cI"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if subprocess.run(
        [*git, "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip():
        raise SystemExit("the models.dev checkout has local changes")
    bun_version = subprocess.run(
        [bun, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run([bun, "install", "--frozen-lockfile"], cwd=models_dev, check=True)
    web = models_dev / "packages/web"
    subprocess.run([bun, "run", "build"], cwd=web, check=True)
    raw = (web / "dist/_api.json").read_bytes()
    api = json.loads(raw)
    text = snapshot_text(api)
    SNAPSHOT.write_text(text, encoding="utf-8")
    LICENSE.write_text(licence, encoding="utf-8")
    date = datetime.datetime.fromisoformat(committed).astimezone(datetime.UTC).date()
    metadata = {
        "source": SOURCE,
        "commit": commit,
        "committed": committed,
        "date": date.isoformat(),
        "license": "MIT (LICENSE.models-dev, at the commit)",
        "built_with": (
            f"bun {bun_version}: bun install --frozen-lockfile, then bun run build "
            "in packages/web; dist/_api.json"
        ),
        "api_json_sha256": _sha256(raw),
        "api_json_providers": len(api),
        "providers": list(PROVIDERS),
        "models": {p: len(api[p]["models"]) for p in PROVIDERS},
        "sha256": _sha256(text.encode()),
    }
    METADATA.write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
    print(f"snapshot {commit[:12]} ({date}): {len(text)} bytes, {metadata['models']}")


# --- measuring ---------------------------------------------------------------


def hostile(catalog: dict[str, Any]) -> dict[str, Any]:
    """A catalog unlike *catalog*: the live fetch a pinned OpenCode must
    ignore (argon offers only ``low``, hydrogen ``low`` and ``medium``,
    gemini-3-flash-preview is gone, a model no snapshot has appears)."""
    other = copy.deepcopy(catalog)
    models = other["google"]["models"]
    effort = {"type": "effort", "values": ["low"]}
    models["gemini-4-argon"]["reasoning_options"] = [effort]
    models["gemini-4-hydrogen"]["reasoning_options"] = [
        {"type": "effort", "values": ["low", "medium"]}
    ]
    models.pop("gemini-3-flash-preview", None)
    models["gemini-hostile-0"] = dict(
        copy.deepcopy(models["gemini-4-argon"]), id="gemini-hostile-0", name="Hostile"
    )
    return other


def measure(
    binary: str,
    models: list[str],
    *,
    models_path: str | None,
    live: dict[str, Any] | None = None,
    fetch: bool | str = True,
) -> dict[str, dict[str, list[str]] | None]:
    """What *binary* (an OpenCode) does with each effort level it lists for
    each google/<model>, over ACP the way a trial runs it: session/new, the
    model set, the level set, one prompt. ``OPENCODE_MODELS_PATH`` is
    *models_path* (unset when None). Its google provider and its live catalog
    fetch (``OPENCODE_MODELS_URL``) point at one local stub: the stub serves
    *live* as ``/api.json`` (the snapshot's catalog when None) and answers each
    model call 400 after recording it, so no model is called. With *fetch*,
    measuring waits for OpenCode's own fetch of that catalog first; with
    ``fetch="closed"`` the fetch points at :data:`CLOSED_URL` and fails, as on
    a closed network; with False it is off (``OPENCODE_DISABLE_MODELS_FETCH``).

    Per model: ``sent``, the levels that reach the stub as ``thinkingLevel``
    equal to the level asked; ``budget``, those sent as ``thinkingBudget``;
    ``unsent``, those that fail before any call. None for a model it doesn't
    know."""
    calls: list[tuple[str, dict[str, Any]]] = []
    served = threading.Event()
    catalog = live if live is not None else json.loads(SNAPSHOT.read_text())
    live_bytes = json.dumps(catalog).encode()

    class Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] != "/api.json":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(live_bytes)))
            self.end_headers()
            self.wfile.write(live_bytes)
            served.set()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            calls.append((self.path, json.loads(body or b"{}")))
            out = json.dumps({"error": {"code": 400, "message": "stub"}}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{stub.server_port}"
    home = tempfile.mkdtemp()
    os.makedirs(f"{home}/.config/opencode")
    with open(f"{home}/.config/opencode/opencode.json", "w") as f:
        json.dump(
            {
                "provider": {
                    "google": {
                        "options": {
                            "baseURL": f"{base}/v1beta",
                            "apiKey": "fake-not-a-key",
                        }
                    }
                }
            },
            f,
        )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": home,
        "OPENCODE_MODELS_URL": base if fetch is True else CLOSED_URL,
    }
    if models_path is not None:
        env["OPENCODE_MODELS_PATH"] = models_path
    if fetch is False:
        env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
    proc = subprocess.Popen(
        [binary, "acp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=home,
    )
    stdin, stdout = proc.stdin, proc.stdout
    assert stdin is not None and stdout is not None
    replies: queue.Queue = queue.Queue()

    def read():
        for line in stdout:
            with contextlib.suppress(ValueError):
                replies.put(json.loads(line))

    threading.Thread(target=read, daemon=True).start()
    ids = iter(range(1, 10**6))

    def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        rid = next(ids)
        message = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        stdin.write((json.dumps(message) + "\n").encode())
        stdin.flush()
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                reply = replies.get(timeout=1)
            except queue.Empty:
                continue
            if reply.get("id") == rid:
                return reply
        raise TimeoutError(method)

    def session(model: str, level: str | None = None) -> tuple[str, dict[str, Any]]:
        sid = call("session/new", {"cwd": home, "mcpServers": []})["result"][
            "sessionId"
        ]
        answer = call(
            "session/set_config_option",
            {"sessionId": sid, "configId": "model", "value": f"google/{model}"},
        )
        if level is not None:
            call(
                "session/set_config_option",
                {"sessionId": sid, "configId": "effort", "value": level},
            )
        return sid, answer

    measured: dict[str, dict[str, list[str]] | None] = {}
    try:
        call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        # The first session starts OpenCode's catalog service and its fetch.
        call("session/new", {"cwd": home, "mcpServers": []})
        if fetch is True:
            if not served.wait(60):
                raise TimeoutError("OpenCode never fetched the stub's catalog")
            # The fetch writes the cache file, then repopulates from it.
            time.sleep(3)
        for model in models:
            _, answer = session(model)
            if "result" not in answer:
                measured[model] = None
                continue
            options = {o["id"]: o for o in answer["result"].get("configOptions", [])}
            listed = [
                o["value"]
                for o in (options.get("effort") or {}).get("options", [])
                if o["value"] != "default"
            ]
            row: dict[str, list[str]] = {}
            for level in listed:
                sid, _ = session(model, level)
                before = len(calls)
                reply = call(
                    "session/prompt",
                    {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]},
                )
                thinking = [
                    body.get("generationConfig", {}).get("thinkingConfig", {})
                    for path, body in calls[before:]
                    if f"models/{model}:" in path
                ]
                if any(t.get("thinkingLevel") == level for t in thinking):
                    kind = "sent"
                elif any("thinkingBudget" in t for t in thinking):
                    kind = "budget"
                elif "invalid google provider options" in str(reply.get("error")):
                    kind = "unsent"
                else:
                    raise AssertionError((model, level, thinking, reply))
                row.setdefault(kind, []).append(level)
            measured[model] = row
    finally:
        proc.kill()
        stub.shutdown()
    return measured


def opencode_version(binary: str) -> str:
    return subprocess.run(
        [binary, "--version"],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": tempfile.mkdtemp(),
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
        },
    ).stdout.strip()


def measure_table(binary: str) -> dict[str, Any]:
    """The effort table for the snapshot, measured on *binary* with the
    snapshot pinned and the stub's live catalog :func:`hostile`."""
    catalog = json.loads(SNAPSHOT.read_text())
    models = sorted(catalog["google"]["models"])
    measured = measure(binary, models, models_path=str(SNAPSHOT), live=hostile(catalog))
    google = catalog["google"]["models"]
    # OpenCode deletes a deprecated model (and an alpha one, unless asked for
    # experimental models) from its provider: those rows are null.
    unknown = [
        m
        for m, row in measured.items()
        if row is None and google[m].get("status") not in ("deprecated", "alpha")
    ]
    if unknown:
        raise SystemExit(f"OpenCode refused models the snapshot has: {unknown}")
    for model, row in measured.items():
        if row and "sent" in row and "budget" in row:
            raise SystemExit(f"{model}: levels both as thinkingLevel and as a budget")
    return {
        "opencode": opencode_version(binary),
        "catalog_sha256": _sha256(SNAPSHOT.read_bytes()),
        "measured": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "how": (
            "tools/opencode_catalog.py measure: over ACP, each level OpenCode "
            "lists set and prompted once, OPENCODE_MODELS_PATH the snapshot, the "
            "google provider and the live catalog fetch at a local stub (no model "
            "call) serving a different catalog"
        ),
        "models": measured,
    }


def write_table(binary: str) -> None:
    table = measure_table(binary)
    EFFORTS.write_text(
        json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"measured {len(table['models'])} models on OpenCode {table['opencode']}")


# --- checking ----------------------------------------------------------------


def problems(*, table: bool = True) -> list[str]:
    """Where the snapshot, its metadata, its licence and (with *table*) the
    effort table disagree."""
    found = []
    data = SNAPSHOT.read_bytes()
    metadata = json.loads(METADATA.read_text())
    catalog = json.loads(data)
    if metadata.get("sha256") != _sha256(data):
        found.append("snapshot.json's sha256 is not models-dev.json's: rebuild")
    if sorted(catalog) != sorted(metadata.get("providers", [])):
        found.append("models-dev.json's providers are not snapshot.json's")
    if data.decode() != snapshot_text(catalog, tuple(sorted(catalog))):
        found.append("models-dev.json is not as build writes it (sorted, indent 1)")
    if not LICENSE.read_text().startswith(MIT):
        found.append("LICENSE.models-dev is not models.dev's MIT licence")
    if not table:
        return found
    efforts = json.loads(EFFORTS.read_text())
    if efforts.get("catalog_sha256") != _sha256(data):
        found.append(
            "gemini-efforts.json was measured on another snapshot: measure again"
        )
    if sorted(efforts.get("models", {})) != sorted(catalog["google"]["models"]):
        found.append(
            "gemini-efforts.json's models are not the snapshot's google models"
        )
    for model, row in efforts.get("models", {}).items():
        status = catalog["google"]["models"].get(model, {}).get("status")
        if (row is None) != (status in ("deprecated", "alpha")):
            found.append(
                f"gemini-efforts.json: {model} is {row} but models.dev marks it {status}"
            )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="snapshot models.dev at the checked-out commit")
    b.add_argument("--models-dev", type=Path, required=True)
    b.add_argument("--bun", default=shutil.which("bun") or "bun")
    m = sub.add_parser(
        "measure", help="measure the Gemini effort table on the pinned OpenCode"
    )
    m.add_argument("--opencode", required=True)
    sub.add_parser("check", help="the snapshot, its metadata and the table agree")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.models_dev, args.bun)
    elif args.command == "measure":
        write_table(args.opencode)
    found = problems(table=args.command != "build")
    for problem in found:
        print(problem, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
