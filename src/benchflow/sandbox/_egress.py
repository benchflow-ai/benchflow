"""Docker/compose allowlist egress enforcement.

Generates a compose override that confines the ``main`` service to an
``internal: true`` network (no direct route off-box) and routes its HTTP(S)
traffic through a sidecar (``bf-egress``) running ``_egress_proxy.py``, which
forwards only to ``allowed_hosts``. Used by the docker and daytona-dind
backends when the resolved policy is ``ALLOWLIST`` (see ``network_policy.py``).
"""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Any

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
    model_lane: str | None = None,
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
    egress_env: dict[str, str] = {
        "ALLOWED_HOSTS": ",".join(allowed_hosts),
        "PORT": str(_EGRESS_PORT),
    }
    egress_service: dict[str, Any] = {
        "image": proxy_image,
        "networks": [_EGRESS_INTERNAL_NET, _EGRESS_EXTERNAL_NET],
        "command": ["python3", "/egress_proxy.py"],
        "environment": egress_env,
        "volumes": [f"{script_dst.resolve()}:/egress_proxy.py:ro"],
        "labels": {"benchflow.owned": "true"},
        "restart": "on-failure",
        "healthcheck": {
            "test": [
                "CMD",
                "python3",
                "-c",
                (
                    "import os, socket; "
                    "port = int(os.environ.get('PORT', '8080')); "
                    "sock = socket.create_connection(('127.0.0.1', port), timeout=1); "
                    "sock.close()"
                ),
            ],
            "interval": "1s",
            "timeout": "2s",
            "retries": 30,
            "start_period": "1s",
        },
    }
    if model_lane:
        # Always-allow lane to the host-side model proxy. The agent's base_url
        # already targets the docker host (_docker_host_address():port), so we only
        # need the sidecar to (a) permit that host and (b) be able to route to it.
        egress_env["BENCHFLOW_EGRESS_LANE_HOST"] = model_lane
        try:
            ipaddress.ip_address(model_lane)
        except ValueError:
            # Hostname (e.g. host.docker.internal on macOS): give the sidecar the
            # blessed container->host route. IP literals are already routable.
            egress_service["extra_hosts"] = [f"{model_lane}:host-gateway"]
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
                "depends_on": {
                    _EGRESS_SERVICE: {"condition": "service_healthy"},
                },
            },
            _EGRESS_SERVICE: egress_service,
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
