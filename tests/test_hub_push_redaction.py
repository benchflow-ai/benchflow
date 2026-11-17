"""``bench hub push`` masks the secrets of every text file it packs (R307B).

``_pack_job`` tarred the job folder as it was on disk: each trial's
``agent/`` files (the agent's own output, ``/logs/agent``) and test.sh's
stdout went up raw, a planted ``password=…`` in plain text, while the
trajectory in the same archive was masked. Every text file is now masked as
``bench bundle`` masks it (``evidence_bundle._redacted_bytes``); binary files
go up as they are; the credential-file and link excludes still hold; a text
file past the one-pass limit is masked line by line, whole lines.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

PASSWORD = "Hu" + "nter2xyz" + "zy-plant" + "ed-7Q"
API_KEY = (
    "sk"
    + "-ant-api"
    + "03-AbCdE"
    + "fGhIjKlM"
    + "nOpQrStU"
    + "vWxYz012"
    + "3456789-"
    + "xyzAA"
)
SECRETS = (PASSWORD.encode(), API_KEY.encode())
MASKED_TRAJECTORY = (
    b'{"type":"tool_call","content":"password=<XXX-benchflow-key-values-XXX>"}\n'
)
BINARY = bytes(range(256)) * 8 + PASSWORD.encode()


def _job(root: Path) -> tuple[Path, Path]:
    job = root / "jobs" / "2026-11-16__23-00-00"
    trial = job / "task__abc123"
    for sub in ("agent", "verifier", "trajectory", "artifacts"):
        (trial / sub).mkdir(parents=True)
    (job / "summary.json").write_text("{}")
    (trial / "agent" / "oracle.txt").write_text(f"export password={PASSWORD}\n")
    (trial / "agent" / "session.log").write_text(f"using {API_KEY}\nok\n")
    (trial / "verifier" / "test-stdout.txt").write_text(
        f"DB_PASSWORD={PASSWORD}\nPASSED\n"
    )
    (trial / "trajectory" / "acp_trajectory.jsonl").write_bytes(MASKED_TRAJECTORY)
    (trial / "artifacts" / "frame.png").write_bytes(BINARY)
    (trial / ".env").write_text(f"PASSWORD={PASSWORD}\n")
    (trial / "auth.json").write_text("{}")
    (trial / "agent" / "link.txt").symlink_to(trial / "agent" / "oracle.txt")
    return job, trial


def _packed(job: Path, archive: Path) -> dict[str, bytes]:
    from benchflow.cli._hub_client import _pack_job

    _pack_job(job, archive)
    files: dict[str, bytes] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is not None:
                files[member.name] = handle.read()
    return files


def test_a_planted_secret_is_masked_in_every_text_file(tmp_path):
    job, _ = _job(tmp_path)
    files = _packed(job, tmp_path / "job.tar.gz")
    hits = [name for name, data in files.items() for s in SECRETS if s in data]
    # The binary frame carries the password inside non-text bytes: as is.
    assert hits == [f"{job.name}/task__abc123/artifacts/frame.png"], hits
    oracle = files[f"{job.name}/task__abc123/agent/oracle.txt"]
    assert b"password=<XXX-benchflow-key-values-XXX>" in oracle, oracle
    stdout = files[f"{job.name}/task__abc123/verifier/test-stdout.txt"]
    assert b"PASSED" in stdout and PASSWORD.encode() not in stdout, stdout


def test_the_masked_trajectory_and_binaries_go_up_unchanged(tmp_path):
    job, _ = _job(tmp_path)
    files = _packed(job, tmp_path / "job.tar.gz")
    trial = f"{job.name}/task__abc123"
    assert files[f"{trial}/trajectory/acp_trajectory.jsonl"] == MASKED_TRAJECTORY
    assert files[f"{trial}/artifacts/frame.png"] == BINARY


def test_credential_files_and_links_stay_out(tmp_path):
    job, _ = _job(tmp_path)
    files = _packed(job, tmp_path / "job.tar.gz")
    trial = f"{job.name}/task__abc123"
    assert f"{trial}/.env" not in files
    assert f"{trial}/auth.json" not in files
    assert f"{trial}/agent/link.txt" not in files
    assert f"{job.name}/summary.json" in files


def test_a_large_log_is_masked_whole_line_by_line(tmp_path, monkeypatch):
    """Past the one-pass masking limit a text file is masked line by line,
    within its time budget: a line the redactor would change is masked
    whole, every line of a private key block too, and the rest is kept. A
    binary file that size goes up as it is."""
    import time

    import benchflow.evidence_bundle as bundle
    from benchflow.publish.redact import REDACTED

    job, trial = _job(tmp_path)
    plain = [f"step {i}: 12 passed" for i in range(40)]
    key = [
        ("--" + "---BEGIN" + " RSA PRI" + "VATE KEY" + "-----"),
        "MIIEowIBAAKCAQEAplantedkeybodylineone0123456789abcdef",
        "planted0key0body0line0two0123456789abcdefABCDEF==",
        "-----END RSA PRIVATE KEY-----",
    ]
    lines = [*plain[:20], f"export password={PASSWORD}", *key, *plain[20:]]
    big_text = trial / "agent" / "big.log"
    big_text.write_text("\n".join(lines) + "\n")
    big_binary = trial / "artifacts" / "video.mp4"
    big_binary.write_bytes(BINARY * 2)
    monkeypatch.setattr(bundle, "_TEXT_LIMIT", 200)
    started = time.monotonic()
    files = _packed(job, tmp_path / "job.tar.gz")
    assert time.monotonic() - started < 60
    name = f"{job.name}/task__abc123"
    packed = files[f"{name}/agent/big.log"].decode().splitlines()
    assert packed == [*plain[:20], REDACTED, *[REDACTED] * 4, *plain[20:]], packed
    assert files[f"{name}/artifacts/video.mp4"] == BINARY * 2


def test_a_large_log_past_its_time_budget_is_left_out_and_named(
    tmp_path, monkeypatch, capsys
):
    """Only when even the line-by-line pass cannot finish is a text file left
    out, and it is named; it is never sent raw."""
    import benchflow.cli._hub_client as hub_client
    import benchflow.evidence_bundle as bundle

    job, trial = _job(tmp_path)
    (trial / "agent" / "big.log").write_text("line\n" * 50 + f"password={PASSWORD}\n")
    monkeypatch.setattr(bundle, "_TEXT_LIMIT", 200)
    monkeypatch.setattr(hub_client, "_LINE_MASK_BUDGET_SEC", -1.0)
    files = _packed(job, tmp_path / "job.tar.gz")
    assert f"{job.name}/task__abc123/agent/big.log" not in files
    assert "Not pushed:" in capsys.readouterr().out


# R307B's re-review of 33325f8ae: five secrets in one log, packed raw when the
# log held a stray non-UTF-8 byte, or a NUL in its first 64 KiB past the limit.
V = (
    "Hunter2-plain-V1q",
    "Hunter2-dbpass-V2q",
    "Hunter2-nextline-V3q",
    "planted0key0body0V4q0123456789abcdefABCDEF",
    "planted-bearer-V5q-0123456789abcdefABCDEFghij",
)
CASE = (
    f"export password={V[0]}\n"
    f"DB_PASSWORD={V[1]}\n"
    "password:\n"
    f"  {V[2]}\n"
    "--" + "---BEGIN" + " RSA PRI" + "VATE KEY" + "-----\n"
    f"MIIEowIBAAKCAQEA{V[3]}\n"
    "-----END RSA PRIVATE KEY-----\n"
    f"Authorization: Bearer {V[4]}\n"
)


def _no_plain_secret(data: bytes) -> list[str]:
    return [v for v in V if v.encode() in data]


def test_a_small_log_with_a_stray_byte_is_masked_not_sent_raw(tmp_path):
    """A stray latin-1 or terminal byte does not make a log binary: it is
    read with surrogateescape, masked, and the byte round-trips."""
    job, trial = _job(tmp_path)
    raw = b"progress \xff done\n" + CASE.encode() + b"tail \x1b[0m\n"
    (trial / "agent" / "case.log").write_bytes(raw)
    files = _packed(job, tmp_path / "job.tar.gz")
    packed = files[f"{job.name}/task__abc123/agent/case.log"]
    assert _no_plain_secret(packed) == [], packed
    assert packed.startswith(b"progress \xff done\n"), packed[:40]
    assert packed.endswith(b"tail \x1b[0m\n"), packed[-40:]


def test_a_large_log_with_a_nul_in_its_head_is_masked_not_sent_raw(
    tmp_path, monkeypatch
):
    """A NUL in a log's first bytes does not make it binary either: past the
    one-pass limit it is masked line by line, a value on the line after its
    ``password:`` included."""
    import benchflow.evidence_bundle as bundle

    job, trial = _job(tmp_path)
    raw = b"start\x00of log\n" + b"step ok\n" * 30 + CASE.encode() + b"end\n"
    (trial / "agent" / "case.log").write_bytes(raw)
    monkeypatch.setattr(bundle, "_TEXT_LIMIT", 200)
    files = _packed(job, tmp_path / "job.tar.gz")
    packed = files[f"{job.name}/task__abc123/agent/case.log"]
    assert _no_plain_secret(packed) == [], packed
    assert packed.startswith(b"start\x00of log\nstep ok\n"), packed[:40]


def test_a_real_binary_with_a_secret_inside_goes_up_unchanged(tmp_path):
    """Only a real binary passes as it is: its magic bytes or extension say
    so and its head is not clean text (a PNG named .bin, a zip, a SQLite
    database)."""
    import io
    import sqlite3
    import zipfile

    job, trial = _job(tmp_path)
    png = b"\x89PNG\r\n\x1a\n" + CASE.encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zipped:
        zipped.writestr("case.log", CASE)
    archive = buffer.getvalue()
    database = trial / "artifacts" / "state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("create table t (v text)")
        connection.execute("insert into t values (?)", (CASE,))
    connection.close()
    stored = database.read_bytes()
    (trial / "artifacts" / "shot.bin").write_bytes(png)
    (trial / "artifacts" / "logs.zip").write_bytes(archive)
    files = _packed(job, tmp_path / "job.tar.gz")
    base = f"{job.name}/task__abc123/artifacts"
    assert files[f"{base}/shot.bin"] == png
    assert files[f"{base}/logs.zip"] == archive
    assert files[f"{base}/state.sqlite"] == stored


def test_a_text_log_that_looks_binary_by_name_or_prefix_is_masked(tmp_path):
    """R307B N1: some magic is plain ASCII (RIFF, ID3), and a name says
    nothing of the bytes: a text log named case.db, or starting RIFF or ID3,
    is masked as text."""
    job, trial = _job(tmp_path)
    (trial / "agent" / "case.db").write_text("log line one\n" + CASE)
    (trial / "agent" / "riff.log").write_text("RIFF log line one\n" + CASE)
    (trial / "agent" / "id3.log").write_text("ID3 log line one\n" + CASE)
    files = _packed(job, tmp_path / "job.tar.gz")
    base = f"{job.name}/task__abc123/agent"
    for name in ("case.db", "riff.log", "id3.log"):
        assert _no_plain_secret(files[f"{base}/{name}"]) == [], name
