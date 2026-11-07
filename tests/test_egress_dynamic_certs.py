"""Dynamic TLS certificate regressions for the CONNECT fix after b3b8afaf."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cryptography import x509

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import _setup_cmd, certificate_material


def _signer(tmp_path: Path) -> tuple[Path, Path, Path]:
    material = certificate_material((), include_ca_key=True)
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    ca_cert, ca_key = tmp_path / "ca.crt", tmp_path / "ca.key"
    ca_cert.write_bytes(material["ca.crt"])
    ca_key.write_bytes(material["ca.key"])
    return cert_dir, ca_cert, ca_key


def test_certificate_material_only_exports_signer_when_requested() -> None:
    """Guards the root signing-key boundary added after commit b3b8afaf."""
    assert "ca.key" not in certificate_material(())
    material = certificate_material((), include_ca_key=True)
    assert set(material) == {"ca.crt", "ca.key"}
    assert b"PRIVATE KEY" in material["ca.key"]
    assert b"PRIVATE KEY" not in material["ca.crt"]


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL unavailable")
def test_cert_store_mints_and_caches_unknown_host(tmp_path: Path) -> None:
    """Guards the fix after b3b8afaf: inspect HTTPS on previously unseen hosts."""
    cert_dir, ca_cert, ca_key = _signer(tmp_path)
    store = proxy_mod.CertStore(str(cert_dir), ca_cert=str(ca_cert), ca_key=str(ca_key))

    first = store.context_for("Allowed.Example.")
    assert first is store.context_for("allowed.example")
    generated = list(cert_dir.glob("dynamic-*.pem"))
    assert len(generated) == 1
    assert os.stat(generated[0]).st_mode & 0o777 == 0o600
    leaf = x509.load_pem_x509_certificate(generated[0].read_bytes())
    san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(san.get_values_for_type(x509.DNSName)) == {
        "allowed.example",
        "www.allowed.example",
    }
    assert leaf.issuer == x509.load_pem_x509_certificate(ca_cert.read_bytes()).subject
    assert first.sni_callback is proxy_mod._check_sni


@pytest.mark.parametrize(
    "host",
    ["../escape.test", "bad\nDNS.2=escape.test", "bad..test", "-bad.test", "127.0.0.1"],
)
def test_cert_store_rejects_unsafe_names_before_minting(
    tmp_path: Path, host: str
) -> None:
    """Guards the fix after b3b8afaf against CONNECT file/config injection."""
    cert_dir, ca_cert, ca_key = _signer(tmp_path)
    store = proxy_mod.CertStore(str(cert_dir), ca_cert=str(ca_cert), ca_key=str(ca_key))
    with pytest.raises(ValueError, match="invalid certificate hostname"):
        store.context_for(host)
    assert list(cert_dir.iterdir()) == []


def test_cert_store_fails_closed_without_dynamic_signer(tmp_path: Path) -> None:
    """Guards the fix after b3b8afaf against opaque fallback on cache miss."""
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    with pytest.raises(RuntimeError, match="no signer configured"):
        proxy_mod.CertStore(str(cert_dir)).context_for("allowed.example")


def test_setup_fails_before_launch_when_openssl_is_missing(tmp_path: Path) -> None:
    """Guards the fix after b3b8afaf: missing signing runtime fails closed."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to("/usr/bin/true")
    ca_dir = tmp_path / "public-ca"
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            _setup_cmd(runtime_dir=str(tmp_path / "runtime"), ca_dir=str(ca_dir)),
        ],
        env={"PATH": str(binaries)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 87
    assert "needs openssl" in result.stderr
    assert not ca_dir.exists(), "startup proceeded without the signing runtime"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL unavailable")
def test_dynamic_certificate_supports_long_dns_names(tmp_path: Path) -> None:
    """Guards the fix after b3b8afaf: DNS SANs may exceed the 64-byte CN limit."""
    cert_dir, ca_cert, ca_key = _signer(tmp_path)
    host = "a" * 60 + "." + "b" * 60 + ".example"
    store = proxy_mod.CertStore(str(cert_dir), ca_cert=str(ca_cert), ca_key=str(ca_key))
    store.context_for(host)
    leaf = x509.load_pem_x509_certificate(
        next(cert_dir.glob("dynamic-*.pem")).read_bytes()
    )
    assert leaf.serial_number.bit_length() <= 159
    assert host in leaf.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value.get_values_for_type(x509.DNSName)
