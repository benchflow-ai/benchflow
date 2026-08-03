"""Sealed (ciphertext-only) transfer channel for AgentCore.

Every command body sent to an AgentCore runtime is recorded permanently by
the platform's shell channel in the runtime's CloudWatch log group, and
base64 is reversible — so nothing secret may ever appear in a command,
encoded or not. This module owns the one safe way to move bytes and
environments into the sandbox:

- The sandbox generates an RSA keypair once; only the **public** key ever
  appears in command output.
- Payloads travel as AES-256-CTR ciphertext with an HMAC-SHA256 tag over
  the ciphertext (encrypt-then-MAC; ``openssl enc`` rejects AEAD ciphers,
  so GCM is not an option). The tag is verified *before* decryption —
  decryption never runs on unauthenticated bytes.
- One RSA-OAEP envelope wraps 64 bytes: AES key ‖ MAC key. The decrypted
  key material is only ever read from a file inside the sandbox; the
  logged command text carries ciphertext, public material, and literal
  ``$(od ...)`` expansions — never key bytes.
- Environments are staged as mode-0600 sourceable files through the same
  channel (:meth:`SealedChannel.stage_env`), so ``exec(env=...)`` commands
  reference them by path only.

The channel deliberately depends on a *raw* exec callable that performs no
environment injection: routing through the sandbox's public ``exec`` would
re-enter env staging and recurse.
"""

from __future__ import annotations

import base64
import shlex
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from benchflow.sandbox._base import ExecResult

#: The service caps a single command payload at 64 KB; base64 inflates by
#: 4/3 and the decrypt scaffolding adds overhead, so chunk well inside it.
MAX_INLINE_BYTES = 24 * 1024

SEAL_DIR = "/tmp/.bf_sealed"


class RawExec(Protocol):
    """Env-injection-free command runner provided by the sandbox."""

    async def __call__(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...


@dataclass(frozen=True)
class SealedPayload:
    """Host-side encryption of one payload, ready for command transport."""

    wrapped_key_b64: str  # RSA-OAEP(AES key ‖ MAC key)
    iv_hex: str
    ciphertext_b64: str
    tag_hex: str  # HMAC-SHA256 over the ciphertext


def seal(public_pem: str, data: bytes) -> SealedPayload:
    """Encrypt *data* for the sandbox holding the matching private key."""

    import os as _os

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives import hmac as _hmac
    from cryptography.hazmat.primitives.asymmetric import padding as _pad
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    secret = _os.urandom(64)
    key, mac_key = secret[:32], secret[32:]
    iv = _os.urandom(16)
    public = serialization.load_pem_public_key(public_pem.encode())
    if not isinstance(public, rsa.RSAPublicKey):
        raise RuntimeError(
            "AgentCore sealed upload expected an RSA public key from the "
            f"sandbox, got {type(public).__name__}"
        )
    wrapped = public.encrypt(
        secret,
        _pad.OAEP(
            mgf=_pad.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    tag = _hmac.HMAC(mac_key, hashes.SHA256())
    tag.update(ciphertext)
    return SealedPayload(
        wrapped_key_b64=base64.b64encode(wrapped).decode(),
        iv_hex=iv.hex(),
        ciphertext_b64=base64.b64encode(ciphertext).decode(),
        tag_hex=tag.finalize().hex(),
    )


class SealedChannel:
    """Confidential upload/env-staging over a permanently logged transport."""

    def __init__(self, exec_raw: RawExec, logger: Any) -> None:
        self._exec_raw = exec_raw
        self._logger = logger
        self._public_key: str | None = None

    async def public_key(self) -> str:
        """Generate (once) and return the sandbox's sealing public key."""

        if self._public_key:
            return self._public_key
        d = SEAL_DIR
        result = await self._exec_raw(
            f"mkdir -p {d} && chmod 700 {d} && "
            f"([ -f {d}/key.pem ] || openssl genpkey -algorithm RSA "
            f"-pkeyopt rsa_keygen_bits:2048 -out {d}/key.pem 2>/dev/null) && "
            f"chmod 600 {d}/key.pem && openssl pkey -in {d}/key.pem -pubout",
            timeout_sec=60,
            user="root",
        )
        stdout = result.stdout or ""
        begin = stdout.find("-----BEGIN PUBLIC KEY-----")
        end = stdout.find("-----END PUBLIC KEY-----")
        if result.return_code != 0 or begin == -1 or end == -1:
            raise RuntimeError(
                "AgentCore sealed upload requires `openssl` inside the "
                "runtime image (the generated wrapper installs it when a "
                "package manager exists). Refusing to fall back to plaintext "
                "command uploads: "
                f"{(result.stderr or result.stdout or '')[:300]}"
            )
        self._public_key = stdout[begin : end + len("-----END PUBLIC KEY-----")]
        return self._public_key

    async def upload(
        self,
        data: bytes,
        *,
        target: str | None,
        mode: str | None = None,
        extract_tar: bool = False,
        user: str | None = None,
        timeout_sec: int = 600,
    ) -> None:
        """Deliver *data* through ciphertext-only commands.

        With ``extract_tar`` the plaintext is a gzip tar extracted at ``/``;
        otherwise it is written to ``target`` (with optional chmod ``mode``).
        """

        payload = seal(await self.public_key(), data)

        token = uuid.uuid4().hex[:16]
        staging = f"{SEAL_DIR}/s_{token}.b64"
        keyfile = f"{SEAL_DIR}/k_{token}.bin"
        aesfile = f"{SEAL_DIR}/a_{token}.bin"
        ctfile = f"{SEAL_DIR}/c_{token}.bin"
        chunk = MAX_INLINE_BYTES
        ct_b64 = payload.ciphertext_b64
        # range() yields nothing for an empty payload, which would leave the
        # staging file absent and fail the decrypt; emit one empty write.
        offsets = list(range(0, len(ct_b64), chunk)) or [0]
        for index in offsets:
            piece = ct_b64[index : index + chunk]
            redirect = ">" if index == offsets[0] else ">>"
            result = await self._exec_raw(
                f"printf %s {shlex.quote(piece)} {redirect} {staging}",
                timeout_sec=120,
                user="root",
            )
            if result.return_code != 0:
                await self._exec_raw(f"rm -f {staging}", timeout_sec=30, user="root")
                raise RuntimeError(
                    f"AgentCore sealed staging failed: {(result.stderr or '')[:500]}"
                )

        # Unwrap 64 bytes (AES key ‖ MAC key), verify the ciphertext tag,
        # and only then decrypt. A mismatched tag aborts before any
        # plaintext is produced.
        decrypt = (
            f"openssl pkeyutl -decrypt -inkey {SEAL_DIR}/key.pem "
            f"-in {keyfile} -out {aesfile} "
            f"-pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256 && "
            f"base64 -d {staging} > {ctfile} && "
            f"ENCK=$(od -An -v -tx1 -N32 {aesfile} | tr -d ' \\n') && "
            f"MACK=$(od -An -v -tx1 -j32 -N32 {aesfile} | tr -d ' \\n') && "
            f"ACTUAL=$(openssl dgst -sha256 -mac HMAC -macopt hexkey:$MACK "
            f"-hex < {ctfile} | sed 's/^.*[= ]//') && "
            f'[ "$ACTUAL" = "{payload.tag_hex}" ] && '
            f'openssl enc -d -aes-256-ctr -K "$ENCK" -iv {payload.iv_hex} '
            f"-in {ctfile}"
        )
        if extract_tar:
            sink = " | tar -xzf - -C /"
            prep = ""
            finalize = ""
            run_user = "root"
        else:
            assert target is not None
            quoted = shlex.quote(target)
            prep = f"mkdir -p $(dirname {quoted}) && "
            sink = f" -out {quoted}"
            finalize = f" && chmod {mode} {quoted}" if mode else ""
            run_user = user if user is not None else "root"
        result = await self._exec_raw(
            f"set -o pipefail; "
            f"trap 'rm -f {staging} {keyfile} {aesfile} {ctfile}' EXIT; "
            f"{prep}"
            f"printf %s {shlex.quote(payload.wrapped_key_b64)} "
            f"| base64 -d > {keyfile} && "
            f"{decrypt}{sink}{finalize}",
            timeout_sec=timeout_sec,
            user=run_user,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"AgentCore sealed upload failed: {(result.stderr or '')[:500]}"
            )

    async def stage_env(self, env: dict[str, str]) -> str:
        """Write *env* into a mode-0600 sandbox file over the sealed channel.

        Returns the in-sandbox path of a shell-sourceable file. Only POSIX
        identifier keys are exported, matching the shared env-file helper.
        """

        lines = []
        for key, value in env.items():
            if not key.isidentifier():
                self._logger.warning(
                    "Skipping non-identifier env key %r for AgentCore exec", key
                )
                continue
            lines.append(f"{key}={shlex.quote(str(value))}")
        body = "\n".join(lines) + "\n"
        path = f"{SEAL_DIR}/env_{uuid.uuid4().hex[:16]}.sh"
        await self.upload(body.encode(), target=path, mode="600", timeout_sec=120)
        return path
