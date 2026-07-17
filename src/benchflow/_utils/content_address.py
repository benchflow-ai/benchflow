"""Content addressing shared by the registry/overlay modules.

A single home for the ``"sha256:" + sha256(bytes).hexdigest()`` idiom so the
environment registry (the ``S`` axis), the config overlay (the ``C`` axis), and
any future content-addressed artifact hash the same way.
"""

from __future__ import annotations

import hashlib


def sha256_prefixed(data: bytes) -> str:
    """Return ``"sha256:<hex>"`` for ``data`` — the repo's content-address form."""
    return "sha256:" + hashlib.sha256(data).hexdigest()
