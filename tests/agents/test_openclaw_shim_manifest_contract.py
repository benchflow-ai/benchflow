"""The OpenClaw ACP shim bytes are frozen by the published agents manifest.

``registry.AGENTS["openclaw"].install_cmd`` embeds ``openclaw_acp_shim.py``
verbatim (via ``_install_python_script``). ``benchflow-ai/agents`` ships an
``acp/openclaw/manifest.toml`` whose ``install_cmd`` is the same string, and the
``manifest-parity`` CI job asserts the two are byte-identical
(``test_manifest_byte_identical_to_core[openclaw]``) before the source-of-truth
flip. So *any* edit to the core shim silently changes the published
``install_cmd`` and breaks that gate unless the agents manifest is regenerated
in lockstep.

That is exactly what happened on ``sdk-update-2026-09-27``: an incidental
cleanup dropped a redundant ``import re`` from the shim, which was enough to
make ``[openclaw]`` drift from ``benchflow-ai/agents@af39feb5`` and fail the
gate (while ``main`` stayed green). The manifest regeneration is owned by the
extraction PRs (agents #72 "move ACP shim ownership to agents", benchflow #1093
"extract OpenClaw from BenchFlow core"), so core must not edit the shim ahead of
them.

This hermetic guard pins the shim bytes so a stray edit fails loudly in the
default suite — where the parity gate does not run — instead of only in the
dedicated CI job. When the extraction lands and the core shim is removed, the
guard skips.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import benchflow

_SHIM = Path(benchflow.__file__).resolve().parent / "agents" / "openclaw_acp_shim.py"

# sha256 of the shim bytes embedded in benchflow-ai/agents@af39feb5's
# acp/openclaw/manifest.toml install_cmd. Update this ONLY together with that
# manifest (or when the shim moves to the agents repo); never on its own.
_PINNED_SHIM_SHA256 = "54643ec5a0fd38ce97bdbf0be800029f5f75da5046c6c7aad318170fe0b5b00e"


def test_openclaw_shim_bytes_match_the_published_manifest() -> None:
    if not _SHIM.is_file():
        pytest.skip("openclaw shim extracted from core (agents #72 / benchflow #1093)")
    actual = hashlib.sha256(_SHIM.read_bytes()).hexdigest()
    assert actual == _PINNED_SHIM_SHA256, (
        "openclaw_acp_shim.py changed. Its bytes are embedded verbatim in "
        "benchflow-ai/agents' openclaw manifest install_cmd, and the "
        "manifest-parity gate requires them to stay byte-identical. Regenerate "
        "acp/openclaw/manifest.toml in benchflow-ai/agents (see agents #72) and "
        "update _PINNED_SHIM_SHA256 in the same change; do not edit the shim on "
        f"its own. got {actual}, pinned {_PINNED_SHIM_SHA256}"
    )


def test_openclaw_install_cmd_embeds_the_shim() -> None:
    import base64

    from benchflow.agents.registry import AGENTS

    cfg = AGENTS.get("openclaw")
    if cfg is None:
        pytest.skip("openclaw extracted from core (benchflow #1093)")
    from benchflow.agents.registry import _OPENCLAW_SHIM

    # install_cmd base64-transports the shim source, so a shim edit is a
    # manifest edit — the coupling the parity gate then checks byte-for-byte.
    assert _SHIM.read_text() == _OPENCLAW_SHIM
    encoded = base64.b64encode(_OPENCLAW_SHIM.encode()).decode()
    assert encoded in cfg.install_cmd
