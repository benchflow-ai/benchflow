"""network_mode='denylist': a root-owned loopback proxy keeps listed URLs out of the agent's reach.

The sandbox user can only reach loopback (the uid firewall in ``lockdown``),
so every HTTP(S) request goes through the proxy, which refuses the denylist
and tunnels everything else. Only hosts named in ``blocked_urls`` are
TLS-intercepted; their leaf certificates are minted on the host and the CA
private key never enters the sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from benchflow.sandbox._egress_denylist_proxy import host_key
from benchflow.sandbox.lockdown import (
    EGRESS_DENYLIST_ENV,
    _exec_failure_detail,
    _exec_return_code,
)

__all__ = [
    "CA_BUNDLE_PATH",
    "CA_CERT_PATH",
    "EGRESS_DENYLIST_ENV",
    "EGRESS_PORT",
    "TRAJECTORY_LOG_NAME",
    "EgressDenylist",
    "certificate_material",
    "denylist_agent_env",
    "egress_denylist_for",
    "start_egress_denylist",
    "stop_egress_denylist",
]

EGRESS_PORT = 18628
TRAJECTORY_LOG_NAME = "egress_denylist.jsonl"
RUNTIME_DIR = "/opt/benchflow-egress"
CA_DIR = "/etc/benchflow-egress"
CA_CERT_PATH = f"{CA_DIR}/ca.crt"
CA_BUNDLE_PATH = f"{CA_DIR}/ca-bundle.crt"
PROXY_URL = f"http://127.0.0.1:{EGRESS_PORT}"
NO_PROXY = "127.0.0.1,localhost,::1"

_PROXY_SCRIPT = Path(__file__).with_name("_egress_denylist_proxy.py")
_LOG_PATH = f"{RUNTIME_DIR}/blocked.jsonl"
_PID_PATH = f"{RUNTIME_DIR}/proxy.pid"
_STDERR_PATH = f"{RUNTIME_DIR}/stderr.log"
_CERT_DAYS = 30
_HEALTH_POLL_SEC = 0.5


@dataclass(frozen=True)
class EgressDenylist:
    """URL prefixes and hosts a denylist task keeps out of reach."""

    blocked_urls: tuple[str, ...]
    blocked_hosts: tuple[str, ...]

    @property
    def inspect_hosts(self) -> tuple[str, ...]:
        """Hosts that must be TLS-intercepted so their paths are visible."""
        hosts: dict[str, None] = {}
        for url in self.blocked_urls:
            parts = urllib.parse.urlsplit(url if "://" in url else f"https://{url}")
            if parts.hostname:
                hosts.setdefault(host_key(parts.hostname))
        return tuple(hosts)


def egress_denylist_for(sandbox_config: Any) -> EgressDenylist | None:
    """The denylist a sandbox config declares, or None for every other network mode."""
    if getattr(sandbox_config, "network_mode", None) != "denylist":
        return None
    return EgressDenylist(
        tuple(getattr(sandbox_config, "blocked_urls", None) or ()),
        tuple(getattr(sandbox_config, "blocked_hosts", None) or ()),
    )


def denylist_agent_env(agent_env: dict[str, str]) -> dict[str, str]:
    """A copy of ``agent_env`` routed through the proxy and trusting its CA."""
    return {
        **agent_env,
        EGRESS_DENYLIST_ENV: "1",
        "HTTP_PROXY": PROXY_URL,
        "HTTPS_PROXY": PROXY_URL,
        "http_proxy": PROXY_URL,
        "https_proxy": PROXY_URL,
        "NO_PROXY": NO_PROXY,
        "no_proxy": NO_PROXY,
        "SSL_CERT_FILE": CA_BUNDLE_PATH,
        "REQUESTS_CA_BUNDLE": CA_BUNDLE_PATH,
        "CURL_CA_BUNDLE": CA_BUNDLE_PATH,
        "GIT_SSL_CAINFO": CA_BUNDLE_PATH,
        "NODE_EXTRA_CA_CERTS": CA_CERT_PATH,
        "NODE_USE_ENV_PROXY": "1",
    }


def certificate_material(
    hosts: tuple[str, ...], *, now: datetime | None = None
) -> dict[str, bytes]:
    """PEM files for the proxy: ``ca.crt`` plus one ``<host>.pem`` (leaf cert and key) per host."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise RuntimeError(
            "network_mode='denylist' needs the 'cryptography' package on the host: "
            "pip install cryptography"
        ) from exc
    now = now or datetime.now(UTC)
    not_before, not_after = now - timedelta(minutes=5), now + timedelta(days=_CERT_DAYS)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "BenchFlow egress policy CA")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    files = {"ca.crt": ca_cert.public_bytes(serialization.Encoding.PEM)}
    for host in hosts:
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(not_before)
            .not_valid_after(not_after)
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName(host), x509.DNSName(f"www.{host}")]
                ),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        files[f"{host}.pem"] = cert.public_bytes(
            serialization.Encoding.PEM
        ) + key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    return files


def _setup_cmd(*, runtime_dir: str = RUNTIME_DIR, ca_dir: str = CA_DIR) -> str:
    """Root shell: find python, install the CA and bundle, replace any running proxy, start it detached."""
    q = shlex.quote
    ca_cert, bundle = f"{ca_dir}/ca.crt", f"{ca_dir}/ca-bundle.crt"
    log, pid = f"{runtime_dir}/blocked.jsonl", f"{runtime_dir}/proxy.pid"
    proxy = (
        f'"$PY" {q(runtime_dir + "/proxy.py")} --port {EGRESS_PORT} '
        f"--policy {q(runtime_dir + '/policy.json')} --cert-dir {q(runtime_dir + '/certs')} --log {q(log)}"
    )
    return (
        "set -e; "
        'PY="$(command -v python3 || command -v python || true)"; '
        '[ -n "$PY" ] || { echo "network_mode=denylist needs python3 in the task image" >&2; exit 87; }; '
        f"mkdir -p {q(ca_dir)} && chmod 755 {q(ca_dir)}; "
        f"cp {q(runtime_dir + '/ca.crt')} {q(ca_cert)} && chmod 644 {q(ca_cert)}; "
        'SYS=""; for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do '
        'if [ -s "$f" ]; then SYS="$f"; break; fi; done; '
        '[ -n "$SYS" ] || SYS="$("$PY" -c \'import ssl; print(ssl.get_default_verify_paths().cafile or "")\')"; '
        f'if [ -n "$SYS" ] && [ -s "$SYS" ]; then cat "$SYS" {q(ca_cert)} > {q(bundle)}; '
        f"else cp {q(ca_cert)} {q(bundle)}; fi; chmod 644 {q(bundle)}; "
        "if command -v update-ca-certificates >/dev/null 2>&1; then "
        f"mkdir -p /usr/local/share/ca-certificates && cp {q(ca_cert)} /usr/local/share/ca-certificates/benchflow-egress.crt "
        "&& update-ca-certificates >/dev/null 2>&1 || true; fi; "
        f"touch {q(log)}; chmod 600 {q(log)}; "
        f'if [ -s {q(pid)} ]; then old="$(cat {q(pid)})"; '
        'kill -TERM "$old" 2>/dev/null || true; '
        'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$old" 2>/dev/null || break; sleep 0.5; done; fi; '
        f"nohup {proxy} </dev/null >{q(runtime_dir + '/stdout.log')} 2>{q(runtime_dir + '/stderr.log')} & "
        f"echo $! > {q(pid)}"
    )


def _health_cmd() -> str:
    probe = (
        "import urllib.request, sys; "
        "opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); "
        f'sys.exit(0 if opener.open("{PROXY_URL}/healthz", timeout=2).status == 200 else 1)'
    )
    return f'PY="$(command -v python3 || command -v python)"; "$PY" -c {shlex.quote(probe)}'


async def _run(env: Any, command: str, label: str, *, timeout_sec: int) -> None:
    result = await env.exec(command, user="root", timeout_sec=timeout_sec)
    rc = _exec_return_code(result)
    if rc != 0:
        raise RuntimeError(
            f"{label} failed with rc={rc}.{_exec_failure_detail(result)}"
        )


async def _upload(env: Any, files: dict[str, bytes]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            local = Path(tmp) / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            await env.upload_file(local, f"{RUNTIME_DIR}/{name}", mode="600")


async def _wait_healthy(env: Any, timeout_sec: int) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_sec
    probe = getattr(env, "exec_transient", None) or env.exec
    while True:
        result = await probe(_health_cmd(), user="root", timeout_sec=10)
        if _exec_return_code(result) == 0:
            return
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(_HEALTH_POLL_SEC)
    tail = await env.exec(
        f"tail -c 2000 {shlex.quote(_STDERR_PATH)} 2>/dev/null",
        user="root",
        timeout_sec=10,
    )
    raise RuntimeError(
        f"egress denylist proxy did not become healthy within {timeout_sec}s: "
        f"{(getattr(tail, 'stdout', '') or '').strip()[-2000:]}"
    )


async def start_egress_denylist(
    env: Any,
    sandbox_user: str | None,
    denylist: EgressDenylist,
    *,
    timeout_sec: int = 120,
) -> None:
    """Upload policy, certificates and the proxy script, then start the proxy as root."""
    if not sandbox_user:
        raise RuntimeError("network_mode='denylist' requires a sandbox_user")
    material = certificate_material(denylist.inspect_hosts)
    policy = {
        "blocked_urls": list(denylist.blocked_urls),
        "blocked_hosts": list(denylist.blocked_hosts),
    }
    files = {
        "policy.json": json.dumps(policy, indent=2).encode("utf-8"),
        "proxy.py": _PROXY_SCRIPT.read_bytes(),
        "ca.crt": material["ca.crt"],
        **{f"certs/{name}": pem for name, pem in material.items() if name != "ca.crt"},
    }
    await _run(
        env,
        f"mkdir -p {shlex.quote(RUNTIME_DIR + '/certs')} && chmod 700 {shlex.quote(RUNTIME_DIR)}",
        "egress runtime dir",
        timeout_sec=30,
    )
    await _upload(env, files)
    await _run(
        env, _setup_cmd(), "egress denylist proxy setup", timeout_sec=timeout_sec
    )
    await _wait_healthy(env, timeout_sec)


async def stop_egress_denylist(env: Any, rollout_dir: Path) -> None:
    """Pull the block log into the rollout trajectory dir, then stop the proxy; never raises."""
    target = Path(rollout_dir) / "trajectory" / TRAJECTORY_LOG_NAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        await env.download_file(_LOG_PATH, target)
    except Exception:
        try:
            result = await env.exec(
                f"cat {shlex.quote(_LOG_PATH)}", user="root", timeout_sec=30
            )
            if _exec_return_code(result) == 0:
                target.write_text(getattr(result, "stdout", "") or "", encoding="utf-8")
        except Exception:
            pass
    with contextlib.suppress(Exception):
        await env.exec(
            f"kill -TERM $(cat {shlex.quote(_PID_PATH)}) 2>/dev/null; "
            f"rm -rf {shlex.quote(RUNTIME_DIR)} {shlex.quote(CA_DIR)}",
            user="root",
            timeout_sec=30,
        )
