"""Local-service sidecars for official Toolathlon tasks.

Upstream Toolathlon runs poste.io / Canvas / WooCommerce / kind as separate host
containers that a task's preprocess, MCP servers and verifier reach at fixed
``localhost`` ports (see ``global_preparation/deploy_containers.sh`` upstream).
benchflow's official-variant sandbox is single-container, so those ~35 tasks
cannot run. This module detects which local service(s) a materialized task needs
and writes an ``environment/docker-compose.yaml`` that stands them up as sidecars
beside the agent's ``main`` container. The mere presence of that compose file
auto-routes the task to the DinD / Docker compose sandbox strategy
(``sandbox/daytona.py`` picks ``_DaytonaDinD`` when ``docker-compose.yaml``
exists), so no run-time flag change is needed.

Implemented: **poste.io** email (23 tasks). The poste sidecar boots the stock
``analogic/poste.io`` image, seeds the fixed 503-user roster upstream ships in
``configs/users_data.json`` (vendored here as ``_toolathlon_poste_users.tsv`` so
the feature is hermetic), and re-applies poste's plaintext-auth config on every
boot (poste regenerates that config on start). The task's email ``*_config.json``
files are rewritten to reach the sidecar by service name, so the ``emails`` MCP
server, preprocess and verifier all transparently talk to it.

Canvas / WooCommerce / kind are heavier multi-container stacks and are TODO;
``required_services`` and ``apply_service_sidecars`` are the extension points.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

POSTE = "poste"

# Pulled via the GCR Docker Hub mirror to dodge anonymous Docker Hub rate limits
# inside concurrent DinD sandboxes (stock analogic/poste.io, unmodified).
_POSTE_IMAGE = "mirror.gcr.io/analogic/poste.io:2.5.5"

# Host-published port (as it appears in the upstream task configs) -> the port
# poste actually listens on inside the container. We rewrite the task's email
# configs to reach the sidecar by service name on its internal port, so the
# sidecar needs no host publishing.
_POSTE_PORT_MAP = {10005: 80, 2525: 25, 1143: 143, 1587: 587}

_LOCALHOSTS = frozenset({"localhost", "127.0.0.1"})


def required_services(task_source: Path) -> set[str]:
    """Return the local-service sidecars a task needs, from its upstream source.

    Scans the *upstream* task dir (the official variant git-clones task files
    into the container's ``/workspace`` at image build, so they are not present
    in the materialized host ``task_dir``)."""
    services: set[str] = set()
    if _email_config_files(task_source):
        services.add(POSTE)
    return services


def apply_service_sidecars(task_dir: Path, services: set[str]) -> None:
    """Materialize compose + assets so *services* run as task-local sidecars."""
    if not services:
        return
    if POSTE in services:
        _write_poste_assets(task_dir)
    _write(
        task_dir / "environment" / "docker-compose.yaml",
        _compose_yaml(services),
    )


def poste_config_rewrite_command(task_name: str) -> str:
    """An in-container setup command (run before preprocess) that points a poste
    task's localhost mail configs at the ``poste`` sidecar. The configs are
    git-cloned into ``/workspace`` at image build, so they are rewritten there,
    not on the host."""
    port_map = json.dumps({str(k): v for k, v in _POSTE_PORT_MAP.items()})
    return "\n".join(
        [
            "/usr/bin/python3 - <<'PY'",
            "import json, glob",
            f"task_dir = '/workspace/tasks/finalpool/{task_name}'",
            f"port_map = {{int(k): v for k, v in {port_map}.items()}}",
            "hosts = ('localhost', '127.0.0.1')",
            "for path in glob.glob(task_dir + '/**/*.json', recursive=True):",
            "    try:",
            "        data = json.load(open(path))",
            "    except Exception:",
            "        continue",
            "    if not isinstance(data, dict):",
            "        continue",
            "    changed = False",
            "    for key in ('imap_server', 'smtp_server'):",
            "        if data.get(key) in hosts:",
            "            data[key] = 'poste'; changed = True",
            "    for key in ('imap_port', 'smtp_port'):",
            "        if data.get(key) in port_map:",
            "            data[key] = port_map[data[key]]; changed = True",
            "    if changed:",
            "        json.dump(data, open(path, 'w'), indent=2)",
            "        print('poste-rewrite:', path)",
            "PY",
        ]
    )


# --------------------------------------------------------------------------- #
# poste.io
# --------------------------------------------------------------------------- #
def _email_config_files(task_dir: Path) -> list[Path]:
    """JSON files describing a poste mailbox (an ``imap_server``/``smtp_server``
    pointing at localhost). Covers the ``emails`` MCP server's
    ``${token.emails_config_file}`` plus preprocess/verifier sender & receiver
    configs — all read from these files."""
    out: list[Path] = []
    for path in sorted(task_dir.rglob("*.json")):
        try:
            if path.stat().st_size > 100_000:
                continue
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and _is_localhost_mail_config(data):
            out.append(path)
    return out


def _is_localhost_mail_config(data: dict) -> bool:
    return any(
        data.get(key) in _LOCALHOSTS for key in ("imap_server", "smtp_server")
    )


def _write_poste_assets(task_dir: Path) -> None:
    """Write the sidecar's boot assets under ``environment/poste/``."""
    poste_dir = task_dir / "environment" / POSTE
    _write(poste_dir / "pairs.tsv", _poste_users_tsv())
    _write(poste_dir / "plaintext_fix.sh", _POSTE_PLAINTEXT_FIX)
    _write(poste_dir / "entry.sh", _POSTE_ENTRY)


def _poste_users_tsv() -> str:
    return (
        resources.files("benchflow.adapters")
        .joinpath("_toolathlon_poste_users.tsv")
        .read_text()
    )


def _compose_yaml(services: set[str]) -> str:
    blocks = ["services:", "  main:", "    depends_on:"]
    for name in sorted(services):
        blocks.append(f"      {name}:")
        blocks.append("        condition: service_healthy")
    if POSTE in services:
        blocks.append(_POSTE_COMPOSE_SERVICE)
    return "\n".join(blocks) + "\n"


# poste seeds ~503 users + re-applies plaintext auth from entry.sh, then touches
# /tmp/poste-ready; main gates on that sentinel so preprocess never races an
# unseeded mailbox. start_period is generous — first-boot seeding takes ~1-2 min.
_POSTE_COMPOSE_SERVICE = """  poste:
    image: mirror.gcr.io/analogic/poste.io:2.5.5
    hostname: mcp.com
    cap_add:
      - NET_ADMIN
      - NET_RAW
      - NET_BIND_SERVICE
      - SYS_PTRACE
    environment:
      DISABLE_CLAMAV: "TRUE"
      DISABLE_RSPAMD: "TRUE"
      DISABLE_P0F: "TRUE"
      HTTPS_FORCE: "0"
      HTTPS: "OFF"
    volumes:
      - ./poste:/toolathlon-poste:ro
    entrypoint: ["/bin/bash", "/toolathlon-poste/entry.sh"]
    healthcheck:
      test: ["CMD-SHELL", "test -f /tmp/poste-ready"]
      interval: 10s
      timeout: 5s
      retries: 90
      start_period: 30s"""


# Re-apply poste's plaintext-auth config (poste regenerates dovecot/haraka
# config on each boot, so this must run every start, not be baked as files).
# Mirrors upstream deployment/poste/scripts/setup.sh::configure_dovecot.
_POSTE_PLAINTEXT_FIX = r"""#!/bin/bash
set +e
for i in $(seq 1 90); do
  [ -f /etc/dovecot/conf.d/10-ssl.conf ] && [ -f /opt/haraka-submission/config/auth.ini ] && break
  sleep 2
done
sleep 6
sed -i 's/ssl = required/ssl = yes/' /etc/dovecot/conf.d/10-ssl.conf
sed -i 's/auth_allow_cleartext = no/auth_allow_cleartext = yes/' /etc/dovecot/conf.d/10-auth.conf
sed -i '/disable_plaintext_auth/d' /etc/dovecot/conf.d/10-auth.conf
sed -i 's/tls_required = true/tls_required = false/' /opt/haraka-smtp/config/auth.ini
sed -i 's/tls_required = true/tls_required = false/' /opt/haraka-submission/config/auth.ini
sed -i 's#^auth/poste#\#auth/poste#' /opt/haraka-submission/config/plugins
printf '127.0.0.1/8\n192.168.0.0/16\n172.16.0.0/12\n10.0.0.0/8\n' > /opt/haraka-submission/config/relay_acl_allow
doveadm reload 2>/dev/null || kill -HUP "$(pgrep dovecot | head -1)" 2>/dev/null
kill "$(pgrep -f 'haraka.*smtp')" 2>/dev/null
kill "$(pgrep -f 'haraka.*submission')" 2>/dev/null
"""


# Runs as the poste entrypoint: launch poste's own /init (PID 1), then in the
# background wait for the admin console, create the mcp.com domain, seed the
# roster (uid 8 = mail, matching upstream's `docker exec --user=8`), apply the
# plaintext-auth fix, and finally signal readiness.
_POSTE_ENTRY = r"""#!/bin/bash
run8() { setpriv --reuid=8 --regid=8 --clear-groups "$@"; }
(
  for i in $(seq 1 120); do
    run8 php /opt/admin/bin/console domain:list >/dev/null 2>&1 && break
    sleep 3
  done
  run8 php /opt/admin/bin/console domain:create mcp.com >/dev/null 2>&1
  n=0
  while IFS=$'\t' read -r email pass name; do
    [ -z "$email" ] && continue
    run8 php /opt/admin/bin/console email:create "$email" "$pass" "$name" >/dev/null 2>&1 &
    n=$((n + 1))
    [ $((n % 40)) -eq 0 ] && wait
  done < /toolathlon-poste/pairs.tsv
  wait
  bash /toolathlon-poste/plaintext_fix.sh
  touch /tmp/poste-ready
  echo "[poste entry] seeded $n mailboxes, plaintext auth applied, ready"
) &
exec /init
"""


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
