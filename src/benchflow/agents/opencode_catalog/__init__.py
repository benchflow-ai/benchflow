"""The OpenCode model catalog BenchFlow pins: one models.dev snapshot.

OpenCode's ``ModelsDev.populate`` reads its model catalog from
``OPENCODE_MODELS_PATH`` when that is set, else from its cache file, else
from the catalog it bundles, and never waits on the network. Unless
``OPENCODE_DISABLE_MODELS_FETCH`` is set, a background refresh fetches
``$OPENCODE_MODELS_URL/api.json`` (models.opencode.ai by default, which
serves models.dev's ``api.json``) into that cache at launch and every 60
minutes. The effort levels it offers a Gemini model come from that catalog,
so they could change between a dry run and a run, or between two runs, with
no change to BenchFlow or to the pinned OpenCode, and a trial on a closed
network got the bundled catalog.

BenchFlow ships one snapshot (:data:`SNAPSHOT`: the providers its OpenCode
launches read, at the models.dev commit :data:`METADATA` records), writes it
into the sandbox with the agent's install (:func:`install_cmd`), and launches
every OpenCode with ``OPENCODE_MODELS_PATH`` pointing at it. The dry run reads
the same file, and the Gemini effort table (:data:`EFFORTS`) was measured on
the pinned OpenCode against it. models.dev is MIT licensed
(``LICENSE.models-dev``); ``README.md`` has the attribution and the refresh
command (``tools/opencode_catalog.py``).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
#: The snapshot: models.dev's ``api.json`` with only the providers BenchFlow's
#: OpenCode launches read (``METADATA["providers"]``), keys sorted.
SNAPSHOT = _HERE / "models-dev.json"
#: Where the snapshot came from: the models.dev commit and its date, how it
#: was built, and the sha256 of the full ``api.json`` and of :data:`SNAPSHOT`.
METADATA = _HERE / "snapshot.json"
#: The Gemini effort levels the pinned OpenCode sends against :data:`SNAPSHOT`,
#: measured by ``tools/opencode_catalog.py measure``.
EFFORTS = _HERE / "gemini-efforts.json"
#: models.dev's licence, verbatim at the snapshot's commit.
LICENSE = _HERE / "LICENSE.models-dev"
#: The environment variable OpenCode reads its catalog from.
MODELS_PATH_ENV = "OPENCODE_MODELS_PATH"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


#: The snapshot's metadata (:data:`METADATA`).
SOURCE: dict[str, Any] = _json(METADATA)
#: The snapshot's date: its models.dev commit's, UTC.
DATE: str = SOURCE["date"]


def snapshot_bytes() -> bytes:
    """:data:`SNAPSHOT`'s bytes, as the install writes them."""
    return SNAPSHOT.read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def catalog() -> dict[str, Any]:
    """The snapshot, parsed."""
    return json.loads(snapshot_bytes())


def provider_models(provider: str) -> frozenset[str]:
    """The model ids the snapshot holds for *provider* (none for a provider
    it leaves out)."""
    return frozenset((catalog().get(provider) or {}).get("models") or {})


def efforts() -> dict[str, Any]:
    """The measured Gemini effort table (:data:`EFFORTS`)."""
    return _json(EFFORTS)


def install_cmd(path: str, node: str) -> str:
    """A shell step that writes the snapshot to *path* in the sandbox with the
    BenchFlow-installed *node*, refusing (exit 1) if what it decodes is not
    the snapshot (its sha256). The file and its two folders are left a+rX
    whatever the umask the install runs under. The bytes travel gzipped in the command, so a
    changed snapshot is a changed install (and install-layer key)."""
    data = snapshot_bytes()
    payload = base64.b64encode(gzip.compress(data, mtime=0)).decode()
    script = (
        'const fs = require("fs"), path = require("path");'
        "const [b64, out, sha] = process.argv.slice(1);"
        'const data = require("zlib").gunzipSync(Buffer.from(b64, "base64"));'
        'const got = require("crypto").createHash("sha256").update(data).digest("hex");'
        "if (got !== sha) {"
        ' console.error("benchflow: the pinned OpenCode catalog decoded to sha256 " + got + ", not " + sha);'
        " process.exit(1);"
        "}"
        "fs.mkdirSync(path.dirname(out), { recursive: true, mode: 0o755 });"
        "fs.writeFileSync(out, data, { mode: 0o644 });"
        "fs.chmodSync(out, 0o644);"
        # Both folders a+rX whatever the sandbox's umask (R600 L2: 700 under
        # 077, so a non-root agent couldn't read the catalog).
        "for (const d of [path.dirname(out), path.dirname(path.dirname(out))])"
        " fs.chmodSync(d, fs.statSync(d).mode | 0o555);"
    )
    return " ".join(
        shlex.quote(part) for part in (node, "-e", script, payload, path, sha256(data))
    )
