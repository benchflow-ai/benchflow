"""The agent's network policy: resolve it, issue its TLS material, run its filter.

A task's ``network_mode`` of ``blocklist`` or ``allowlist`` becomes a
:class:`NetworkPolicy`. The rollout hands the policy to the LiteLLM proxy
through the agent environment, so provider-side web tools see it, and starts
the in-sandbox egress filter (:mod:`benchflow.sandbox.egress_filter`) beside
the proxy, so everything the agent fetches itself passes through it. The
same policy marker turns on the uid firewall that makes loopback the agent's
only exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.sandbox.egress_filter import Policy
from benchflow.task.config import NetworkMode

#: Seen by the agent: a policy is in force, nothing about its contents.
NETWORK_POLICY_MARKER_ENV = "BENCHFLOW_NETWORK_POLICY"
#: Seen by the model proxy only: the policy itself.
NETWORK_POLICY_ENV = "BENCHFLOW_NETWORK_POLICY_JSON"
#: What the running filter adds to the agent environment.
AGENT_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)
EGRESS_SANDBOX_ROOT = "/tmp/benchflow-egress"
_STATE_DEADLINE_SEC = 30.0
_FILTER_SOURCE = Path(__file__).with_name("egress_filter.py")


@dataclass(frozen=True)
class NetworkPolicy:
    """What the agent may reach: a filtering mode and its entries."""

    mode: NetworkMode
    entries: tuple[str, ...]

    @classmethod
    def resolve(cls, config: Any) -> NetworkPolicy | None:
        """The agent's policy: its own override, else the sandbox default.

        Only ``allowlist`` and ``blocklist`` need a filter; ``public`` and
        ``no-network`` are decided by the backend at sandbox creation.
        """
        for section in (config.agent, config.sandbox):
            mode = section.network_mode
            if mode is None:
                continue
            if mode is NetworkMode.ALLOWLIST:
                return cls(mode, tuple(section.allowed_hosts or ()))
            if mode is NetworkMode.BLOCKLIST:
                return cls(mode, tuple(section.blocked_urls or ()))
            return None
        return None

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "rules": list(self.entries)}

    def as_policy(self) -> Policy:
        return Policy.from_json(self.to_json())

    def agent_env(self, agent_env: dict[str, str]) -> dict[str, str]:
        """Mark the agent's environment; the rules themselves stay out of it."""
        return {**agent_env, NETWORK_POLICY_MARKER_ENV: "1"}

    def proxy_env(self, proxy_env: dict[str, str]) -> dict[str, str]:
        """Hand the model proxy the policy for provider-side tools."""
        return {
            **proxy_env,
            NETWORK_POLICY_ENV: json.dumps(self.to_json(), separators=(",", ":")),
        }


@dataclass(frozen=True)
class TlsMaterial:
    ca_pem: bytes
    cert_pem: bytes
    key_pem: bytes


def issue_tls_material(
    hosts: tuple[str, ...], *, valid_for: timedelta = timedelta(days=7)
) -> TlsMaterial:
    """A throw-away CA and one leaf certificate naming ``hosts`` and their subdomains."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.now(UTC)

    def certificate(
        subject: str, key: Any, *, issuer: Any, issuer_key: Any, ca: bool
    ) -> Any:
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + valid_for)
            .add_extension(
                x509.BasicConstraints(ca=ca, path_length=None), critical=True
            )
        )
        if not ca:
            names = [
                x509.DNSName(name) for host in hosts for name in (host, f"*.{host}")
            ]
            builder = (
                builder.add_extension(
                    x509.SubjectAlternativeName(names), critical=False
                )
                .add_extension(
                    x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]),
                    critical=False,
                )
                .add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        content_commitment=False,
                        key_encipherment=True,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
            )
        return builder.sign(issuer_key, hashes.SHA256())

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "BenchFlow egress filter")]
    )
    ca_cert = certificate(
        "BenchFlow egress filter", ca_key, issuer=ca_name, issuer_key=ca_key, ca=True
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = certificate(
        hosts[0], leaf_key, issuer=ca_name, issuer_key=ca_key, ca=False
    )
    return TlsMaterial(
        ca_pem=ca_cert.public_bytes(serialization.Encoding.PEM),
        cert_pem=leaf_cert.public_bytes(serialization.Encoding.PEM),
        key_pem=leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


@dataclass
class EgressFilterProcess:
    """A running filter inside one sandbox, and what the agent needs to use it."""

    sandbox: Any
    runtime_dir: str
    port: int
    inspecting: bool
    paths: dict[str, str]

    @property
    def agent_env(self) -> dict[str, str]:
        proxy = f"http://127.0.0.1:{self.port}"
        env = {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        if self.inspecting:
            # One bundle for tools that replace the trust store, the bare CA
            # for Node, which only ever adds to it.
            env.update(
                SSL_CERT_FILE=self.paths["ca_bundle"],
                REQUESTS_CA_BUNDLE=self.paths["ca_bundle"],
                CURL_CA_BUNDLE=self.paths["ca_bundle"],
                NODE_EXTRA_CA_CERTS=self.paths["ca"],
            )
        return env

    async def stop(self, *, log_destination: Path | None = None) -> None:
        """Stop the filter, collect its decision log, then remove its files.

        The files stay behind if the log could not be collected, so the audit
        trail is never the thing that gets lost.
        """
        with contextlib.suppress(Exception):
            await _terminate(self.sandbox, pid_path=self.paths["pid"])
        if log_destination is not None:
            try:
                log_destination.parent.mkdir(parents=True, exist_ok=True)
                await self.sandbox.download_file(self.paths["log"], log_destination)
            except Exception:
                return
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self.runtime_dir)}", timeout_sec=10
            )


async def start_egress_filter(
    sandbox: Any, policy: NetworkPolicy, *, python: str
) -> EgressFilterProcess:
    """Upload the filter, its policy, and its certificate, then detach it."""
    runtime_dir = f"{EGRESS_SANDBOX_ROOT}/{uuid4().hex[:16]}"
    paths = {
        name: f"{runtime_dir}/{filename}"
        for name, filename in {
            "filter": "egress_filter.py",
            "policy": "policy.json",
            "launch": "launch.json",
            "state": "state.json",
            "pid": "filter.pid",
            "log": "decisions.jsonl",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "ca": "ca.pem",
            "ca_bundle": "ca-bundle.pem",
            "cert": "cert.pem",
            "key": "key.pem",
        }.items()
    }
    # Traversable, not listable: the agent may read the CA files by name and
    # nothing else; the policy, key, state, and log stay with the filter's user.
    result = await sandbox.exec(
        f"mkdir -p {shlex.quote(runtime_dir)} && chmod 711 {shlex.quote(runtime_dir)}",
        timeout_sec=20,
    )
    if result.return_code != 0:
        raise RuntimeError(
            f"prepare egress filter directory failed: {_details(result)}"
        )
    try:
        inspected = await _install(sandbox, policy, paths, python=python)
        state = await _wait_for_state(
            sandbox, state_path=paths["state"], stderr_path=paths["stderr"]
        )
    except BaseException:
        # Never leave a half-started filter, or its key, behind.
        with contextlib.suppress(Exception):
            await _terminate(sandbox, pid_path=paths["pid"])
        with contextlib.suppress(Exception):
            await sandbox.exec(f"rm -rf {shlex.quote(runtime_dir)}", timeout_sec=10)
        raise
    return EgressFilterProcess(
        sandbox=sandbox,
        runtime_dir=runtime_dir,
        port=int(state["port"]),
        inspecting=bool(inspected),
        paths=paths,
    )


async def _install(
    sandbox: Any, policy: NetworkPolicy, paths: dict[str, str], *, python: str
) -> tuple[str, ...]:
    """Upload everything the filter needs and detach it; return the inspected hosts."""
    launch: dict[str, str] = {
        key: paths[key] for key in ("policy", "state", "pid", "log", "stdout", "stderr")
    }
    inspected = policy.as_policy().inspected_hosts
    if inspected:
        material = issue_tls_material(inspected)
        await _upload(sandbox, material.ca_pem, paths["ca"], mode="644")
        await _upload(sandbox, material.cert_pem, paths["cert"], mode="600")
        await _upload(sandbox, material.key_pem, paths["key"], mode="600")
        launch.update({key: paths[key] for key in ("ca", "ca_bundle", "cert", "key")})
    await _upload(sandbox, _FILTER_SOURCE.read_bytes(), paths["filter"], mode="644")
    await _upload(
        sandbox, json.dumps(policy.to_json()).encode(), paths["policy"], mode="600"
    )
    await _upload(sandbox, json.dumps(launch).encode(), paths["launch"], mode="600")
    command = (
        f"{shlex.quote(python)} {shlex.quote(paths['filter'])} "
        f"launch {shlex.quote(paths['launch'])}"
    )
    result = await sandbox.exec(command, timeout_sec=20)
    if result.return_code != 0:
        raise RuntimeError(f"start egress filter failed: {_details(result)}")
    return inspected


async def _terminate(sandbox: Any, *, pid_path: str) -> None:
    """Stop the detached filter named by ``pid_path`` and wait for it to go."""
    quoted = shlex.quote(pid_path)
    await sandbox.exec(
        f"if [ -s {quoted} ]; then pid=$(cat {quoted}); "
        f'kill -TERM "$pid" 2>/dev/null || true; '
        f'for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || exit 0; sleep 0.5; done; '
        f'kill -KILL "$pid" 2>/dev/null || true; fi',
        timeout_sec=15,
    )


async def stop_egress_filter(
    process: EgressFilterProcess, *, log_destination: Path | None = None
) -> None:
    await process.stop(log_destination=log_destination)


async def _upload(sandbox: Any, content: bytes, target: str, *, mode: str) -> None:
    with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
        handle.write(content)
        source = Path(handle.name)
    try:
        await sandbox.upload_file(source, target, mode=mode)
    finally:
        source.unlink(missing_ok=True)


async def _wait_for_state(
    sandbox: Any, *, state_path: str, stderr_path: str
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STATE_DEADLINE_SEC
    while loop.time() < deadline:
        result = await sandbox.exec(
            f"cat {shlex.quote(state_path)} 2>/dev/null || true", timeout_sec=5
        )
        text = (result.stdout or "").strip()
        if text:
            with contextlib.suppress(ValueError):
                state = json.loads(text)
                if int(state.get("port") or 0) > 0:
                    return state
        await asyncio.sleep(0.25)
    stderr = await sandbox.exec(
        f"tail -c 4000 {shlex.quote(stderr_path)} 2>/dev/null || true", timeout_sec=5
    )
    raise RuntimeError(
        f"egress filter did not publish its port: {(stderr.stdout or '').strip()}"
    )


def _details(result: Any) -> str:
    parts = [f"exit code {getattr(result, 'return_code', '?')}"]
    for stream in ("stdout", "stderr"):
        text = (getattr(result, stream, "") or "").strip()
        if text:
            parts.append(f"{stream}: {text}")
    return "; ".join(parts)
