"""Public text names no internal tracker key.

CHANGELOG.md and README.md ship in the sdist and on the docs site. The
release leak gate checks the built artifacts against a private pattern file;
this test keeps the source clean before a release ever runs. It names no real
key: any key-shaped token (capital letters, a hyphen, a number) is refused
unless it is a public name, listed here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY_SHAPED = re.compile(r"\b([A-Z][A-Z0-9]{1,7})-\d+\b")
# Public names that look like keys: CVE ids, hashes and ciphers, standards,
# model families.
PUBLIC_NAMES = frozenset({"AES", "CVE", "GPT", "ISO", "PEP", "RFC", "SHA", "UTF"})


def key_shaped_tokens(text: str) -> list[str]:
    return sorted(
        {
            m.group(0)
            for m in KEY_SHAPED.finditer(text)
            if m.group(1) not in PUBLIC_NAMES
        }
    )


@pytest.mark.parametrize("name", ["CHANGELOG.md", "README.md"])
def test_public_text_names_no_internal_key(name):
    assert key_shaped_tokens((ROOT / name).read_text()) == []


def test_a_key_is_caught_and_public_names_are_not():
    text = "Fixed in TEAM-12 (see CVE-2026-1234, SHA-256, ISO-8601, GPT-6)."
    assert key_shaped_tokens(text) == ["TEAM-12"]
