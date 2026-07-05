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

Implemented: **poste.io** email (24 tasks), **Canvas** (8 tasks),
**WooCommerce** (9 tasks), and **k8s/kind** (5 tasks). The poste sidecar boots the stock
``analogic/poste.io`` image, seeds the fixed 503-user roster upstream ships in
``configs/users_data.json`` (vendored here as ``_toolathlon_poste_users.tsv`` so
the feature is hermetic), and re-applies poste's plaintext-auth config on every
boot (poste regenerates that config on start). The task's email ``*_config.json``
files are rewritten to reach the sidecar by service name, so the ``emails`` MCP
server, preprocess and verifier all transparently talk to it.

Canvas and WooCommerce boot stock upstream images with task-local first-boot
seed scripts. k8s tasks run their upstream kind scripts from the main container
against the DinD VM's Docker daemon through the Docker socket and host
networking.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

POSTE = "poste"
WOO = "woo"
CANVAS = "canvas"
K8S = "k8s"

# Pulled via the GCR Docker Hub mirror to dodge anonymous Docker Hub rate limits
# inside concurrent DinD sandboxes (stock analogic/poste.io, unmodified).
_POSTE_IMAGE = "mirror.gcr.io/analogic/poste.io:2.5.5"
_WOO_DB_IMAGE = "mysql:8.0"
_WOO_WP_IMAGE = "wordpress:6.8.2-php8.2-apache"
_CANVAS_IMAGE = "lbjay/canvas-docker"

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
    if _needs_woo(task_source):
        services.add(WOO)
    if _needs_canvas(task_source):
        services.add(CANVAS)
    if _needs_k8s(task_source):
        services.add(K8S)
    return services


def apply_service_sidecars(
    task_dir: Path, services: set[str], *, source_root: Path | None = None
) -> None:
    """Materialize compose + assets so *services* run as task-local sidecars."""
    if not services:
        return
    if POSTE in services:
        _write_poste_assets(task_dir)
    if WOO in services:
        _write_woo_assets(task_dir, source_root)
    if CANVAS in services:
        _write_canvas_assets(task_dir, source_root)
    _write(
        task_dir / "environment" / "docker-compose.yaml",
        _compose_yaml(services),
    )


def poste_config_rewrite_command(task_name: str) -> str:
    """An in-container setup command (run before preprocess) that points a poste
    task's localhost mail configs at the ``poste`` sidecar. Recurses into nested
    dicts/lists so per-mailbox configs are covered too."""
    port_map = json.dumps({str(k): v for k, v in _POSTE_PORT_MAP.items()})
    return "\n".join(
        [
            "/usr/bin/python3 - <<'PY'",
            "import json, glob",
            f"task_dir = '/workspace/tasks/finalpool/{task_name}'",
            f"port_map = {{int(k): v for k, v in {port_map}.items()}}",
            "hosts = ('localhost', '127.0.0.1')",
            "ch = [False]",
            "def walk(o):",
            "    if isinstance(o, dict):",
            "        for k in ('imap_server', 'smtp_server'):",
            "            if o.get(k) in hosts:",
            "                o[k] = 'poste'; ch[0] = True",
            "        for k in ('imap_port', 'smtp_port'):",
            "            if o.get(k) in port_map:",
            "                o[k] = port_map[o[k]]; ch[0] = True",
            "        for v in o.values():",
            "            walk(v)",
            "    elif isinstance(o, list):",
            "        for v in o:",
            "            walk(v)",
            "for path in glob.glob(task_dir + '/**/*.json', recursive=True):",
            "    try:",
            "        data = json.load(open(path))",
            "    except Exception:",
            "        continue",
            "    ch[0] = False",
            "    walk(data)",
            "    if ch[0]:",
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
    return any(data.get(key) in _LOCALHOSTS for key in ("imap_server", "smtp_server"))


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
    sidecars = sorted(services & {CANVAS, POSTE, WOO})
    blocks = ["services:", "  main:"]
    if sidecars:
        blocks.append("    depends_on:")
        for name in sidecars:
            blocks.append(f"      {name}:")
            blocks.append("        condition: service_healthy")
    if K8S in services:
        blocks.extend(_K8S_MAIN_COMPOSE_BLOCK)
    if POSTE in services:
        blocks.append(_POSTE_COMPOSE_SERVICE)
    if WOO in services:
        blocks.append(_WOO_COMPOSE_SERVICE)
    if CANVAS in services:
        blocks.append(_CANVAS_COMPOSE_SERVICE)
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
    ports:
      - "10005:80"
      - "2525:25"
      - "1143:143"
      - "1587:587"
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


def _copy_users_data(
    task_dir: Path, service_name: str, source_root: Path | None
) -> None:
    users_data = "{}\n"
    if source_root is not None:
        source = source_root / "configs" / "users_data.json"
        if source.is_file():
            users_data = source.read_text()
    _write(task_dir / "environment" / service_name / "users_data.json", users_data)


# --------------------------------------------------------------------------- #
# k8s / kind
# --------------------------------------------------------------------------- #
def _needs_k8s(task_dir: Path) -> bool:
    cfg = task_dir / "task_config.json"
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    servers = data.get("needed_mcp_servers") or []
    return any(str(name).lower() == K8S for name in servers)


def k8s_runtime_probe_command() -> str:
    """Fail early if a k8s task cannot reach the mounted Docker daemon/tools."""
    return "\n".join(
        [
            "set -e",
            "docker info >/dev/null",
            "kind --version",
            "kubectl version --client=true --output=yaml | head -20",
            "helm version --short",
        ]
    )


_K8S_MAIN_COMPOSE_BLOCK = [
    "    network_mode: host",
    "    volumes:",
    "      - /var/run/docker.sock:/var/run/docker.sock",
    "    environment:",
    '      DOCKER_HOST: "unix:///var/run/docker.sock"',
    '      KIND_EXPERIMENTAL_PROVIDER: "docker"',
    '      TOOLATHLON_K8S_HOST_NETWORK: "1"',
]


# --------------------------------------------------------------------------- #
# woocommerce (pre-seeded WordPress multisite + WooCommerce, store81..100)
# --------------------------------------------------------------------------- #
def _needs_woo(task_dir: Path) -> bool:
    for path in task_dir.rglob("token_key_session.py"):
        try:
            text = path.read_text()
        except OSError:
            continue
        if "woocommerce_site_url" in text and "localhost" in text:
            return True
    return False


def woo_config_rewrite_command(task_name: str) -> str:
    """Point a WooCommerce task's token file at the ``woo`` sidecar."""
    return "\n".join(
        [
            "/usr/bin/python3 - <<'PY'",
            "import glob",
            f"task_dir = '/workspace/tasks/finalpool/{task_name}'",
            "for path in glob.glob(task_dir + '/**/token_key_session.py', recursive=True):",
            "    try:",
            "        s = open(path).read()",
            "    except Exception:",
            "        continue",
            "    s2 = s.replace('localhost:10003', 'woo')",
            "    if s2 != s:",
            "        open(path, 'w').write(s2); print('woo-rewrite:', path)",
            "PY",
        ]
    )


def _write_woo_assets(task_dir: Path, source_root: Path | None) -> None:
    woo_dir = task_dir / "environment" / WOO
    _copy_users_data(task_dir, WOO, source_root)
    _write(woo_dir / "entry.sh", _WOO_ENTRY)


_WOO_COMPOSE_SERVICE = f"""  woo-db:
    image: {_WOO_DB_IMAGE}
    environment:
      MYSQL_ROOT_PASSWORD: "rootpass123"
      MYSQL_DATABASE: "wordpress"
      MYSQL_USER: "wordpress"
      MYSQL_PASSWORD: "wppass123"
    healthcheck:
      test: ["CMD-SHELL", "mysqladmin ping -h 127.0.0.1 -uwordpress -pwppass123 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 90
      start_period: 30s
  woo:
    image: {_WOO_WP_IMAGE}
    hostname: woo
    depends_on:
      woo-db:
        condition: service_healthy
    environment:
      WORDPRESS_DB_HOST: "woo-db"
      WORDPRESS_DB_USER: "wordpress"
      WORDPRESS_DB_PASSWORD: "wppass123"
      WORDPRESS_DB_NAME: "wordpress"
    volumes:
      - ./woo:/toolathlon-woo:ro
    entrypoint: ["/bin/bash", "/toolathlon-woo/entry.sh"]
    healthcheck:
      test: ["CMD-SHELL", "test -f /tmp/woo-ready && curl -s -o /dev/null -w '%{{http_code}}' -H 'Host: woo' http://localhost/store97/wp-json/ | grep -qE '200|401' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 90
      start_period: 45s"""


_WOO_ENTRY = r"""#!/bin/bash
set -euo pipefail

WP_PATH=/var/www/html
WP_URL=http://woo
export WP_CLI_ALLOW_ROOT=1

log() { echo "[toolathlon woo] $*"; }

docker-entrypoint.sh apache2-foreground &
apache_pid=$!

cleanup() {
  kill "$apache_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if [ -f "$WP_PATH/wp-config.php" ]; then
    break
  fi
  sleep 2
done

if ! command -v wp >/dev/null 2>&1; then
  log "installing wp-cli"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --connect-timeout 20 --max-time 120 \
      -o /usr/local/bin/wp \
      https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar
  else
    php -r "copy('https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar', '/usr/local/bin/wp');"
  fi
  chmod +x /usr/local/bin/wp
fi

db_ready=0
for _ in $(seq 1 180); do
  if php -r '$m = @mysqli_connect(getenv("WORDPRESS_DB_HOST"), getenv("WORDPRESS_DB_USER"), getenv("WORDPRESS_DB_PASSWORD"), getenv("WORDPRESS_DB_NAME")); exit($m ? 0 : 1);'; then
    db_ready=1
    break
  fi
  sleep 2
done
if [ "$db_ready" != "1" ]; then
  log "database readiness timed out"
  exit 1
fi

if ! wp core is-installed --url="$WP_URL" --path="$WP_PATH" >/dev/null 2>&1; then
  log "installing WordPress"
  wp core install \
    --url="$WP_URL" \
    --title="My WooCommerce Store" \
    --admin_user="mcpwoocommerce" \
    --admin_password="mcpwoocommerce" \
    --admin_email="woocommerce@mcp.com" \
    --skip-email \
    --path="$WP_PATH"
fi

if ! wp plugin is-installed woocommerce --path="$WP_PATH" >/dev/null 2>&1; then
  log "installing WooCommerce"
  wp plugin install woocommerce --version=10.7.0 --path="$WP_PATH"
fi
wp plugin activate woocommerce --path="$WP_PATH" >/dev/null 2>&1 || true
wp rewrite structure '/%postname%/' --path="$WP_PATH" >/dev/null 2>&1 || true
wp rewrite flush --path="$WP_PATH" >/dev/null 2>&1 || true
touch "$WP_PATH/.htaccess"
grep -q 'HTTP_AUTHORIZATION' "$WP_PATH/.htaccess" 2>/dev/null || cat >> "$WP_PATH/.htaccess" <<'EOF'
RewriteEngine On
RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]
SetEnvIf Authorization (.+) HTTPS=on
EOF

if ! wp eval "echo is_multisite() ? 'true' : 'false';" --path="$WP_PATH" 2>/dev/null | grep -q true; then
  log "converting to multisite"
  wp core multisite-convert --title="My Multisite Network" --path="$WP_PATH"
fi
cat > "$WP_PATH/.htaccess" <<'EOF'
# BEGIN WordPress Multisite
RewriteEngine On
RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]
RewriteBase /
RewriteRule ^index\.php$ - [L]
RewriteRule ^([_0-9a-zA-Z-]+/)?wp-admin$ $1wp-admin/ [R=301,L]
RewriteCond %{REQUEST_FILENAME} -f [OR]
RewriteCond %{REQUEST_FILENAME} -d
RewriteRule ^ - [L]
RewriteRule ^([_0-9a-zA-Z-]+/)?(wp-(content|admin|includes).*) $2 [L]
RewriteRule ^([_0-9a-zA-Z-]+/)?(.*\.php)$ $2 [L]
RewriteRule . index.php [L]
# END WordPress Multisite
SetEnvIf Authorization (.+) HTTPS=on
EOF
wp plugin activate woocommerce --network --path="$WP_PATH" >/dev/null 2>&1 || true
mkdir -p "$WP_PATH/wp-content/uploads"
chown -R www-data:www-data "$WP_PATH/wp-content/uploads"
chmod -R ug+rwX "$WP_PATH/wp-content/uploads"

log "creating WooCommerce task stores"
php <<'PHP' > /tmp/toolathlon_woo_users.tsv
<?php
$data = json_decode(file_get_contents('/toolathlon-woo/users_data.json'), true);
foreach (($data['users'] ?? array()) as $user) {
    $uid = intval($user['id'] ?? 0);
    if ($uid < 81 || $uid > 100) {
        continue;
    }
    $fields = array(
        strval($uid),
        strval($user['first_name'] ?? ''),
        strval($user['last_name'] ?? ''),
        strval($user['full_name'] ?? ''),
        strval($user['email'] ?? ''),
        strval($user['woocommerce_consumer_key'] ?? ''),
        strval($user['woocommerce_consumer_secret'] ?? ''),
    );
    echo implode("\t", array_map(fn($v) => str_replace("\t", " ", $v), $fields)) . "\n";
}
?>
PHP

created=0
while IFS=$'\t' read -r user_id first_name last_name full_name email consumer_key consumer_secret; do
  [ -n "${user_id:-}" ] || continue
  slug="store${user_id}"
  site_url="${WP_URL}/${slug}/"
  if ! wp site list --field=url --path="$WP_PATH" | grep -Fxq "$site_url"; then
    wp site create \
      --slug="$slug" \
      --title="${first_name} ${last_name}'s Store" \
      --email="$email" \
      --path="$WP_PATH" >/dev/null
  fi
  site_id="$(wp site list --path="$WP_PATH" --fields=blog_id,url --format=csv \
    | awk -F, -v url="$site_url" 'NR > 1 && $2 == url { print $1; exit }')"
  if [ -z "$site_id" ]; then
    site_id=$((user_id - 79))
  fi
  month_path="$(date -u +%Y/%m)"
  site_uploads="$WP_PATH/wp-content/uploads/sites/${site_id}"
  mkdir -p \
    "$site_uploads/$month_path" \
    "$site_uploads/wc-logs" \
    "$site_uploads/woocommerce_uploads"
  chown -R www-data:www-data "$site_uploads"
  chmod -R ug+rwX "$site_uploads"
  TOOLATHLON_WOO_CONSUMER_KEY="$consumer_key" \
  TOOLATHLON_WOO_CONSUMER_SECRET="$consumer_secret" \
  TOOLATHLON_WOO_DESCRIPTION="${slug} API Key" \
  wp eval '
global $wpdb;
$consumer_key = getenv("TOOLATHLON_WOO_CONSUMER_KEY");
$consumer_secret = getenv("TOOLATHLON_WOO_CONSUMER_SECRET");
$description = getenv("TOOLATHLON_WOO_DESCRIPTION") ?: "Toolathlon API Key";
$wpdb->delete($wpdb->prefix . "woocommerce_api_keys", array("description" => $description));
$wpdb->insert(
    $wpdb->prefix . "woocommerce_api_keys",
    array(
        "user_id" => 1,
        "description" => $description,
        "permissions" => "read_write",
        "consumer_key" => wc_api_hash($consumer_key),
        "consumer_secret" => $consumer_secret,
        "truncated_key" => substr($consumer_key, -7)
    )
);
if ($wpdb->last_error) { fwrite(STDERR, $wpdb->last_error . "\n"); exit(1); }
' \
    --url="$site_url" \
    --path="$WP_PATH" >/dev/null
  created=$((created + 1))
done < /tmp/toolathlon_woo_users.tsv

log "ready with ${created} stores"
touch /tmp/woo-ready
wait "$apache_pid"
"""


# --------------------------------------------------------------------------- #
# Canvas LMS (pre-seeded single-container Canvas)
# --------------------------------------------------------------------------- #
def _needs_canvas(task_dir: Path) -> bool:
    """Canvas tasks declare the canvas MCP server in task_config.json."""
    cfg = task_dir / "task_config.json"
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return False
    servers = data.get("needed_mcp_servers") or []
    return any("canvas" in str(name).lower() for name in servers)


def canvas_config_rewrite_command(task_name: str) -> str:
    """Point Canvas preprocess and MCP settings at the ``canvas`` sidecar."""
    return "\n".join(
        [
            "/usr/bin/python3 - <<'PY'",
            "import glob",
            f"task_dir = '/workspace/tasks/finalpool/{task_name}'",
            "reps = [('localhost:10001', 'canvas:3000'), ('127.0.0.1:10001', 'canvas:3000')]",
            "for path in glob.glob(task_dir + '/**/*', recursive=True):",
            "    if not path.endswith(('.py', '.json', '.txt', '.md', '.cfg', '.yaml', '.yml')):",
            "        continue",
            "    try:",
            "        s = open(path).read()",
            "    except Exception:",
            "        continue",
            "    s2 = s",
            "    for old, new in reps:",
            "        s2 = s2.replace(old, new)",
            "    if s2 != s:",
            "        open(path, 'w').write(s2); print('canvas-rewrite:', path)",
            "token_paths = ['/workspace/configs/token_key_session.py']",
            "token_paths += glob.glob(task_dir + '/**/token_key_session.py', recursive=True)",
            "domain_reps = [('localhost:20001', 'canvas:3000'), ('127.0.0.1:20001', 'canvas:3000'), ('localhost:10001', 'canvas:3000')]",
            "for path in sorted(set(token_paths)):",
            "    try:",
            "        lines = open(path).read().splitlines(keepends=True)",
            "    except Exception:",
            "        continue",
            "    changed = False",
            "    for i, line in enumerate(lines):",
            "        if 'canvas_domain' in line:",
            "            new_line = line",
            "            for old, new in domain_reps:",
            "                new_line = new_line.replace(old, new)",
            "            if new_line != line:",
            "                lines[i] = new_line; changed = True",
            "    if changed:",
            "        open(path, 'w').write(''.join(lines)); print('canvas-rewrite-domain:', path)",
            "PY",
        ]
    )


def _write_canvas_assets(task_dir: Path, source_root: Path | None) -> None:
    canvas_dir = task_dir / "environment" / CANVAS
    _copy_users_data(task_dir, CANVAS, source_root)
    _write(canvas_dir / "entry.sh", _CANVAS_ENTRY)
    _write(canvas_dir / "seed_canvas.rb", _CANVAS_SEED_RB)


_CANVAS_COMPOSE_SERVICE = f"""  canvas:
    image: {_CANVAS_IMAGE}
    hostname: canvas
    volumes:
      - ./canvas:/toolathlon-canvas:ro
    entrypoint: ["/bin/bash", "/toolathlon-canvas/entry.sh"]
    healthcheck:
      test: ["CMD-SHELL", "test -f /tmp/canvas-ready && curl -s -o /dev/null -w '%{{http_code}}' http://localhost:3000/login | grep -qE '200|302' || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 60
      start_period: 120s"""


_CANVAS_ENTRY = r"""#!/bin/bash
set -euo pipefail

CANVAS_DIR=/opt/canvas/canvas-lms
BUNDLE_PATH=/opt/canvas/.gems/bin/bundle

log() { echo "[toolathlon canvas] $*"; }

rm -f "$CANVAS_DIR"/tmp/pids/server.pid "$CANVAS_DIR"/tmp/pids/*.pid 2>/dev/null || true
if [ -f "$CANVAS_DIR/config/domain.yml" ]; then
  sed -i 's/domain: "localhost:3000"/domain: "canvas:3000"/' "$CANVAS_DIR/config/domain.yml" || true
  sed -i 's/domain: "localhost:10001"/domain: "canvas:3000"/' "$CANVAS_DIR/config/domain.yml" || true
fi

if command -v supervisord >/dev/null 2>&1 && [ -f /etc/supervisor/supervisord.conf ]; then
  log "starting supervisord"
  supervisord -c /etc/supervisor/supervisord.conf &
  canvas_pid=$!
elif [ -x /sbin/my_init ]; then
  log "starting my_init"
  /sbin/my_init &
  canvas_pid=$!
else
  log "no known Canvas init command found; keeping container alive for logs"
  sleep infinity &
  canvas_pid=$!
fi

canvas_ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://localhost:3000/login -o /dev/null 2>&1; then
    if cd "$CANVAS_DIR" && GEM_HOME=/opt/canvas/.gems "$BUNDLE_PATH" exec rails runner 'puts Account.default.id' 2>/dev/null | grep -qE '^[0-9]+$'; then
      canvas_ready=1
      break
    fi
  fi
  sleep 3
done
if [ "$canvas_ready" != "1" ]; then
  log "Canvas rails runner readiness timed out"
  exit 1
fi

log "seeding users and repairing predefined API tokens"
cd "$CANVAS_DIR"
GEM_HOME=/opt/canvas/.gems "$BUNDLE_PATH" exec rails runner /toolathlon-canvas/seed_canvas.rb

for _ in $(seq 1 60); do
  code="$(curl -sS -o /tmp/canvas-admin-probe.json -w '%{http_code}' \
    -H 'Authorization: Bearer mcpcanvasadmintoken2' \
    http://localhost:3000/api/v1/accounts/1/courses || true)"
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 2
done
if [ "${code:-}" != "200" ]; then
  log "Canvas admin token probe failed with HTTP ${code:-none}"
  cat /tmp/canvas-admin-probe.json >&2 || true
  exit 1
fi

touch /tmp/canvas-ready
log "ready"

while true; do
  if ! kill -0 "$canvas_pid" 2>/dev/null; then
    log "canvas init process exited; keeping container alive for service health"
    sleep infinity &
    canvas_pid=$!
  fi
  wait "$canvas_pid" || true
done
"""


_CANVAS_SEED_RB = r"""require 'json'

users_data = JSON.parse(File.read('/toolathlon-canvas/users_data.json'))
account = Account.default

def ensure_user(account, user_data)
  email = user_data.fetch('email')
  pseudonym = Pseudonym.where(unique_id: email).first
  user = pseudonym&.user
  unless user
    user = User.create!(
      name: user_data.fetch('full_name'),
      short_name: user_data.fetch('first_name')
    )
    pseudonym = Pseudonym.new(user: user, account: account, unique_id: email)
  end
  pseudonym.password = user_data.fetch('password')
  pseudonym.password_confirmation = user_data.fetch('password')
  pseudonym.sis_user_id ||= format('MCP%06d', user_data.fetch('id').to_i)
  pseudonym.save!
  user
end

def reset_token(user, purpose, token_value)
  return if token_value.to_s.empty?
  user.access_tokens.where(purpose: purpose).delete_all
  user.access_tokens.create!(purpose: purpose, token: token_value)
end

created = 0
(users_data['users'] || []).each do |user_data|
  next if user_data['full_name'].to_s.start_with?('MCP Canvas Admin')
  user = ensure_user(account, user_data)
  reset_token(user, 'Predefined API Token', user_data['canvas_token'])
  created += 1
end

admin_users = [
  {
    'full_name' => 'MCP Canvas Admin 1',
    'first_name' => 'Admin1',
    'email' => 'mcpcanvasadmin1@mcp.com',
    'password' => 'mcpcanvasadminpass1',
    'canvas_token' => 'mcpcanvasadmintoken1',
    'sis_user_id' => 'ADMIN001'
  },
  {
    'full_name' => 'MCP Canvas Admin 2',
    'first_name' => 'Admin2',
    'email' => 'mcpcanvasadmin2@mcp.com',
    'password' => 'mcpcanvasadminpass2',
    'canvas_token' => 'mcpcanvasadmintoken2',
    'sis_user_id' => 'ADMIN002'
  },
  {
    'full_name' => 'MCP Canvas Admin 3',
    'first_name' => 'Admin3',
    'email' => 'mcpcanvasadmin3@mcp.com',
    'password' => 'mcpcanvasadminpass3',
    'canvas_token' => 'mcpcanvasadmintoken3',
    'sis_user_id' => 'ADMIN003'
  }
]

Role.ensure_built_in_roles!
Role.clear_built_in_roles!
admin_role = Role.get_built_in_role('AccountAdmin')
admin_created = 0
admin_users.each do |admin_data|
  data = {
    'id' => 0,
    'full_name' => admin_data['full_name'],
    'first_name' => admin_data['first_name'],
    'email' => admin_data['email'],
    'password' => admin_data['password'],
    'canvas_token' => admin_data['canvas_token']
  }
  user = ensure_user(account, data)
  pseudonym = Pseudonym.where(unique_id: admin_data['email']).first
  pseudonym.update!(sis_user_id: admin_data['sis_user_id']) if pseudonym
  reset_token(user, 'Admin API Token', admin_data['canvas_token'])
  account_user = AccountUser.where(account: account, user: user).first_or_initialize
  account_user.role = admin_role
  account_user.workflow_state = 'active'
  account_user.save!
  admin_created += 1
end

puts "Toolathlon Canvas seed complete: #{created} users, #{admin_created} admins"
"""
