"""Docker/compose allowlist egress enforcement.

Generates a compose override that confines the ``main`` service to an
``internal: true`` network (no direct route off-box) and routes its HTTP(S)
traffic through a sidecar (``bf-egress``) running ``_egress_proxy.py``, which
forwards only to ``allowed_hosts``. Used by the docker and daytona-dind
backends when the resolved policy is ``ALLOWLIST`` (see ``network_policy.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

#: Minimal image used for the proxy sidecar. Pinned; only needs stdlib python3.
DEFAULT_EGRESS_PROXY_IMAGE = "python:3.12-alpine"
_PROXY_SCRIPT = Path(__file__).parent / "_egress_proxy.py"
_EGRESS_INTERNAL_NET = "bf_egress_internal"
_EGRESS_EXTERNAL_NET = "bf_egress_external"
_EGRESS_SERVICE = "bf-egress"
_EGRESS_PORT = 8080


def build_egress_override(
    allowed_hosts: list[str] | tuple[str, ...],
    *,
    out_dir: Path,
    proxy_image: str = DEFAULT_EGRESS_PROXY_IMAGE,
) -> Path:
    """Write the allowlist egress compose override into *out_dir*; return its path.

    Copies the proxy script next to the override so both live on a host path the
    daemon can bind-mount. The agent service ``main`` is detached from the
    default bridge (sequence ``networks`` is replaced on merge) and pointed at
    the proxy via ``HTTP(S)_PROXY``; the proxy alone bridges to the outside.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    script_dst = out_dir / "egress_proxy.py"
    script_dst.write_text(_PROXY_SCRIPT.read_text())
    proxy_url = f"http://{_EGRESS_SERVICE}:{_EGRESS_PORT}"
    override = {
        "services": {
            "main": {
                # Sequence networks are REPLACED on compose merge → drops `default`.
                "networks": [_EGRESS_INTERNAL_NET],
                "environment": {
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "NO_PROXY": "localhost,127.0.0.1",
                    "no_proxy": "localhost,127.0.0.1",
                },
                "depends_on": [_EGRESS_SERVICE],
            },
            _EGRESS_SERVICE: {
                "image": proxy_image,
                "networks": [_EGRESS_INTERNAL_NET, _EGRESS_EXTERNAL_NET],
                "command": ["python3", "/egress_proxy.py"],
                "environment": {
                    "ALLOWED_HOSTS": ",".join(allowed_hosts),
                    "PORT": str(_EGRESS_PORT),
                },
                "volumes": [f"{script_dst.resolve()}:/egress_proxy.py:ro"],
                "labels": {"benchflow.owned": "true"},
                "restart": "on-failure",
            },
        },
        "networks": {
            _EGRESS_INTERNAL_NET: {
                "internal": True,
                "labels": {"benchflow.owned": "true"},
            },
            _EGRESS_EXTERNAL_NET: {"labels": {"benchflow.owned": "true"}},
        },
    }
    path = out_dir / "docker-compose-egress.json"
    path.write_text(json.dumps(override, indent=2))
    return path
