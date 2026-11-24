"""BenchFlow masks credentials it staged in everything it records.

BenchFlow knows the exact values it puts into a sandbox: the provider keys
and the proxy key in the agent's environment, the secret fields of each
credentials file it stages, and the keys it uploads with a proxy. Each is
registered when it is staged, and the records written through BenchFlow's
redaction layer (the trajectory, result.json's text, the session-log copies)
replace it, in each form an agent can print it, with ``[masked:<name>]``. The
trial's ``integrity/masking.json`` says how many were masked, by name, never
a value, and which folders this version does not sweep.

The placeholders (``tests/staged_secret_forms.py``) are not shaped like any
known key family, so the pattern redaction alone leaves them: only the
registry masks them.
"""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow._utils import staged_secrets
from tests.staged_secret_forms import (
    BLOCK,
    LONG,
    PROXY_VALUE,
    REFRESH,
    STAGED,
    TOKEN,
    URLSAFE,
    _scan,
    _value_only,
    _wraps,
    printed_forms,
)


def test_the_scan_sees_every_form_it_must():
    """The scan finds each printed form left unmasked, the URL-safe and the
    standard alphabet differ for URLSAFE at every offset, and LONG wraps."""
    for name, value in STAGED.items():
        for form in printed_forms(value):
            assert _scan(form, {name: value}), (name, form)
    for offset in (0, 1, 2):
        assert _value_only(URLSAFE, offset, "url") != _value_only(
            URLSAFE, offset, "std"
        )
    assert all(len(wrap.splitlines()) > 2 for wrap in _wraps(LONG))


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    staged_secrets.register_run("proxy master key", PROXY_VALUE)
    with staged_secrets.trial_registry() as reg:
        for name, value in STAGED.items():
            if name != "proxy master key":
                staged_secrets.register(name, value)
        yield reg


def _printout() -> str:
    lines = []
    for value in STAGED.values():
        lines += printed_forms(value)
    # One line of a multi-line value printed alone (`sed -n 2p key.pem`).
    lines += [f"line: {line}" for line in BLOCK.splitlines()]
    return "\n".join(lines)


def test_a_staged_value_leaves_no_form_in_the_trajectory(registry):
    """The trajectory: an agent's tool output with every form of every
    staged value, written through the ACP trajectory writer."""
    from benchflow.trajectories.types import redact_acp_trajectory_jsonl

    events = [
        {"type": "tool_call_update", "content": [{"text": _printout()}]},
        {"type": "agent_message", "text": f"the token is {TOKEN} ok"},
        # A request body held as JSON text, as a model call's record holds
        # it: the printout's line breaks are escaped twice in the file.
        {"type": "llm_request", "body": json.dumps({"content": _printout()})},
    ]
    written = redact_acp_trajectory_jsonl(events)
    lines = written.splitlines()
    assert _scan(written) == []
    assert _scan(json.loads(lines[0])["content"][0]["text"]) == []
    assert _scan(json.loads(json.loads(lines[2])["body"])["content"]) == []
    assert "[masked:BF_STAGED_TOKEN]" in written
    assert "[masked:proxy master key]" in written


def test_a_staged_value_leaves_no_form_in_result_json_text(registry):
    """result.json's text fields (an agent's error) go through the same
    layer."""
    from benchflow.rollout._results import _masked

    assert _scan(_masked(_printout()) or "") == []


def test_a_staged_value_leaves_no_form_in_a_session_log_copy(registry, tmp_path):
    """A session-log copy, a .jsonl and a plain text file: masked with no
    values of its own list (the registry is what knows them)."""
    from benchflow.rollout._session_log import _redact

    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"type": "tool_result", "content": form})
            for value in STAGED.values()
            for form in printed_forms(value)
        )
        + "\n"
    )
    plain = tmp_path / "session.log"
    plain.write_text(_printout())
    _redact(log, [])
    _redact(plain, [])
    assert _scan(log.read_text()) == []
    assert _scan(plain.read_text()) == []


def test_a_session_log_names_a_staged_value_it_also_was_given(registry, tmp_path):
    """A value on the session log's own list (the agent's env) that BenchFlow
    staged is masked under its name, not as an unnamed ``[redacted]``."""
    from benchflow.rollout._session_log import _redact

    log = tmp_path / "session.jsonl"
    log.write_text(json.dumps({"type": "tool_result", "content": TOKEN}) + "\n")
    _redact(log, [TOKEN])
    text = log.read_text()
    assert "[masked:BF_STAGED_TOKEN]" in text and "[redacted]" not in text


def test_a_staged_value_leaves_no_form_in_the_kill_and_process_records(
    registry, tmp_path
):
    """S-4: a process the agent left with a staged value on its command line
    is named, command line and all, in the stop's kill-after-grading.json and
    in integrity/verifier-inputs.json's process rows. Both are written masked,
    every form, and masking.json lists both. Mutant: a plain json.dumps."""
    from benchflow._utils.kill_record import (
        KILL_AFTER_GRADING_RECORD,
        write_kill_record,
    )
    from benchflow.sandbox._image_manifest import write_detector_record

    argv = f"python3 -c import time; time.sleep(900) rv-argv={TOKEN} {_printout()}"
    write_kill_record(
        tmp_path,
        {"processes": [{"pid": 41, "command": argv}], "killed_total": 1},
        KILL_AFTER_GRADING_RECORD,
    )
    write_detector_record(
        tmp_path,
        {"changes": [{"where": "processes", "after": {"pid": 41, "argv": argv}}]},
    )
    for name in (KILL_AFTER_GRADING_RECORD, "integrity/verifier-inputs.json"):
        text = (tmp_path / name).read_text()
        assert _scan(text) == [], name
        assert "[masked:BF_STAGED_TOKEN]" in text, name
        json.loads(text)
    record = json.loads(staged_secrets.write_record(tmp_path, registry).read_text())
    assert {
        "kill-before-grading.json",
        "kill-after-grading.json",
        "integrity/verifier-inputs.json",
    } <= set(record["applied_to"])
    assert record["masked"]["BF_STAGED_TOKEN"] >= 2


def test_each_mask_names_what_it_masked_and_the_record_counts_it(registry, tmp_path):
    from benchflow.trajectories.types import redact_trajectory_text

    out = redact_trajectory_text(f"a {REFRESH} b {TOKEN} c")
    assert out == "a [masked:auth.json.refresh_token] b [masked:BF_STAGED_TOKEN] c"
    path = staged_secrets.write_record(tmp_path, registry)
    record = json.loads(path.read_text())
    assert path == tmp_path / "integrity" / "masking.json"
    assert record["masked"]["auth.json.refresh_token"] >= 1
    assert record["masked"]["BF_STAGED_TOKEN"] >= 1
    assert sorted(record["registered"]) == sorted(STAGED)
    assert _scan(path.read_text()) == [], "the record holds names and counts only"


def test_the_record_names_what_this_version_does_not_sweep(registry, tmp_path):
    record = json.loads(staged_secrets.write_record(tmp_path, registry).read_text())
    assert {"verifier/", "agent/", "artifacts/"} <= set(record["not_swept"]["folders"])
    assert {
        "run-card.json",
        "summary.json",
        "results.jsonl",
        "adp.jsonl",
        "verifiers.jsonl",
    } <= set(record["not_swept"]["job_files"])
    assert {"trajectory/", "agent/acp_trajectory.jsonl"} <= set(record["applied_to"])


def test_the_minimum_length_is_sixteen(monkeypatch, tmp_path):
    """15 characters are never masked and are named; 16 are masked, however
    plain (a hex key has 16 distinct characters at most)."""
    from benchflow.trajectories.types import redact_trajectory_text

    fifteen, sixteen, plain = "a1b2c3d4e5f6g7h", "a1b2c3d4e5f6g7h8", "0" * 16
    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("FIFTEEN", fifteen)
        staged_secrets.register("SIXTEEN", sixteen)
        staged_secrets.register("PLAIN", plain)
        out = redact_trajectory_text(f"x {fifteen} y {sixteen} z {plain}")
        record = json.loads(staged_secrets.write_record(tmp_path, reg).read_text())
    assert out == f"x {fifteen} y [masked:SIXTEEN] z [masked:PLAIN]"
    assert record["too_short"] == ["FIFTEEN"]
    assert record["registered"] == ["PLAIN", "SIXTEEN"]


def test_a_value_that_was_not_staged_is_left_as_it_is(registry):
    """Only what BenchFlow staged: a look-alike one character off, and the
    token's own halves, stay."""
    from benchflow.trajectories.types import redact_trajectory_text

    lookalike = REFRESH[:-1] + ("A" if REFRESH[-1] != "A" else "B")
    text = f"{lookalike} and {REFRESH[:12]} and {REFRESH[12:]}"
    assert redact_trajectory_text(text) == text


def test_outside_a_trial_nothing_is_registered_or_masked():
    from benchflow.trajectories.types import redact_trajectory_text

    assert staged_secrets.current() is None
    staged_secrets.register("NOT_IN_A_TRIAL", REFRESH)
    assert redact_trajectory_text(f"x {REFRESH} y") == f"x {REFRESH} y"


@pytest.mark.asyncio
async def test_each_trial_masks_its_own_values_and_the_runs(monkeypatch):
    """Two trials at once in one process, as a job runs them: each masks its
    own value and the run's key, never the other's value, and each record
    names only its own and the run's."""
    from benchflow.trajectories.types import redact_trajectory_text

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    staged_secrets.register_run("proxy master key", PROXY_VALUE)
    mine, theirs = "trA1Qw2Er3Ty4Ui5Op6", "trB9Zx8Cv7Bn6Mm5Lk4"

    async def trial(name: str, value: str, other: str):
        with staged_secrets.trial_registry() as reg:
            staged_secrets.register(name, value)
            await asyncio.sleep(0)
            out = redact_trajectory_text(f"{value} {other} {PROXY_VALUE}")
            await asyncio.sleep(0)
            return out, reg.record()

    (a, ra), (b, rb) = await asyncio.gather(
        trial("A_TOKEN", mine, theirs), trial("B_TOKEN", theirs, mine)
    )
    assert a == f"[masked:A_TOKEN] {theirs} [masked:proxy master key]"
    assert b == f"[masked:B_TOKEN] {mine} [masked:proxy master key]"
    assert ra["registered"] == ["A_TOKEN", "proxy master key"]
    assert rb["registered"] == ["B_TOKEN", "proxy master key"]


def test_a_text_redacted_before_its_value_was_staged_is_masked_after():
    """The trial's redaction memo never hands back a copy made before a
    value was registered."""
    from benchflow.trajectories.types import (
        redact_trajectory_text,
        trial_redaction_memo,
    )

    text = f"x {REFRESH} y"
    with trial_redaction_memo(), staged_secrets.trial_registry():
        assert redact_trajectory_text(text) == text
        staged_secrets.register("LATE_TOKEN", REFRESH)
        assert redact_trajectory_text(text) == "x [masked:LATE_TOKEN] y"


def test_a_non_text_copy_keeps_its_length_and_stays_valid(registry, tmp_path):
    """A session-log copy that is not text (a tar, a sqlite file): each
    staged value's bytes are filled in place, every other byte kept, and
    the file still opens."""
    from benchflow.rollout._session_log import _redact

    data = f"token={REFRESH}\n".encode()
    tar_path = tmp_path / "logs.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("a.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    before = tar_path.read_bytes()
    _redact(tar_path, [])
    after = tar_path.read_bytes()
    at, end = (
        before.find(REFRESH.encode()),
        before.find(REFRESH.encode()) + len(REFRESH),
    )
    assert len(after) == len(before)
    assert after[:at] == before[:at] and after[end:] == before[end:]
    with tarfile.open(tar_path) as tar:
        member = tar.extractfile("a.txt")
        assert member is not None
        assert member.read() == b"token=" + b"*" * len(REFRESH) + b"\n"

    db = tmp_path / "state.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table t (v text)")
    con.execute("insert into t values (?)", (REFRESH,))
    con.commit()
    con.close()
    size = db.stat().st_size
    _redact(db, [])
    assert db.stat().st_size == size
    con = sqlite3.connect(db)
    assert con.execute("pragma integrity_check").fetchone()[0] == "ok"
    assert con.execute("select v from t").fetchall() == [("*" * len(REFRESH),)]
    con.close()


# --- where BenchFlow stages a value ----------------------------------------


def test_the_launch_env_registers_its_credentials_not_its_settings(monkeypatch):
    """The agent's launch env registers each credential-bearing value, never
    a model name or a URL."""
    from benchflow.agents.launch_env import agent_launch_env

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    env = {
        "BF_STAGED_TOKEN": TOKEN,
        "ANTHROPIC_MODEL": "claude-placeholder-model-name-1",
        "ANTHROPIC_BASE_URL": "http://proxy.placeholder.example:4000/v1",
    }
    with staged_secrets.trial_registry() as reg:
        agent_launch_env("claude-agent-acp", "anthropic/claude-placeholder", env)
        assert reg.names() == ["BF_STAGED_TOKEN"]


@pytest.fixture
def uploads(monkeypatch):
    from benchflow.agents import credentials

    uploaded: dict[str, str] = {}

    async def upload(env, path, content, *, owner):
        uploaded[path] = content

    monkeypatch.setattr(credentials, "upload_credential", upload)
    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    return uploaded


@pytest.mark.asyncio
async def test_a_staged_credentials_file_registers_its_secret_fields(uploads):
    """The one staging write registers each secret field of a JSON
    credentials file, each by its own name: an account id is no secret, even
    under a field the rule calls a credential."""
    from benchflow.agents import credentials

    content = json.dumps(
        {
            "tokens": {"refresh_token": REFRESH},
            "credentials": {"account_id": "acct-placeholder-0001"},
        }
    )
    with staged_secrets.trial_registry() as reg:
        await credentials.place_credential(
            object(), "/root/.codex/auth.json", content, owner=None, record=None
        )
        assert reg.names() == ["auth.json.tokens.refresh_token"]
    assert uploads == {"/root/.codex/auth.json": content}


@pytest.mark.asyncio
async def test_a_subscription_login_file_registers_its_tokens(
    uploads, tmp_path, monkeypatch
):
    """A host login file uploaded for a subscription run (Claude's
    ``.credentials.json``) registers its access and refresh tokens."""
    from benchflow.agents import credentials
    from benchflow.trajectories.types import redact_trajectory_text

    access, refresh = "acc4Rt7Yu1Io5Pa8Sd2Fg6Hj9", "ref3Lk6Jh9Gf2Ds5Ap8Oi1Uy4"
    login = tmp_path / ".credentials.json"
    login.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": access,
                    "refreshToken": refresh,
                    "expiresAt": 1900000000000,
                    "scopes": ["user:inference"],
                }
            }
        )
    )
    monkeypatch.setattr(credentials, "login_path", lambda path, **kw: login)
    with staged_secrets.trial_registry() as reg:
        await credentials.upload_subscription_auth(
            object(), "claude-agent-acp", "/root"
        )
        names = reg.names()
        out = redact_trajectory_text(login.read_text())
    assert names == [
        ".credentials.json.claudeAiOauth.accessToken",
        ".credentials.json.claudeAiOauth.refreshToken",
    ]
    assert access not in out and refresh not in out
    assert uploads == {"/root/.claude/.credentials.json": login.read_text()}


def _adc(tmp_path: Path, monkeypatch, data: dict) -> None:
    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(json.dumps(data))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())


def test_vertex_credentials_register_their_secret_fields(tmp_path, monkeypatch):
    from benchflow.agents.env import inject_vertex_credentials

    _adc(
        tmp_path,
        monkeypatch,
        {
            "type": "authorized_user",
            "client_id": "placeholder-client.apps.example",
            "client_secret": CLIENT_VALUE,
            "refresh_token": REFRESH,
        },
    )
    env: dict[str, str] = {"GOOGLE_CLOUD_PROJECT": "placeholder-project"}
    with staged_secrets.trial_registry() as reg:
        inject_vertex_credentials(env, "google-vertex/gemini-placeholder")
        names = set(reg.names())
    assert {
        "GOOGLE_APPLICATION_CREDENTIALS_JSON.client_secret",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON.refresh_token",
    } <= names
    assert not any("client_id" in name for name in names)


# The armour is built, so no secret scanner reads this file as holding a key.
ARMOUR = "PRIVATE " + "KEY"
CLIENT_VALUE = "cls7Ty3Ui9Op1As5Df8Gh2Jk"
PEM = (
    f"-----BEGIN {ARMOUR}-----\n"
    + "\n".join(
        [
            "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDdPl4c3eHo7Wq2",
            "Zx9Lm4Rt8Qw3Nf6Bd2Vy5Ui8Op1As4Df7Gh0Jk3Lz6Xc9Vb2Nm5Qw8Er1Ty4Ui7Op",
            "Hq9Wn2Ex5Rc8Tv1Yb4Um7In0Ok3Pl6Qa9Zs2Xd5Cf8Vg1Bh4Nj7Mk0Lo3Pi6Uy9",
        ]
    )
    + f"\n-----END {ARMOUR}-----\n"
)


def test_a_service_account_key_is_masked_however_it_is_printed(tmp_path, monkeypatch):
    """A service-account credentials file: its private key (a multi-line
    PEM, held \\n-escaped in the JSON) is registered, and masked as `cat` of
    the file and `printenv` print it, and each body line printed alone. The
    key's id names it (no secret, by the rule config.json uses), and the
    PEM's armour lines are not registered either."""
    from benchflow.agents.env import inject_vertex_credentials
    from benchflow.trajectories.types import redact_trajectory_text

    key_id = "kid0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c"
    _adc(
        tmp_path,
        monkeypatch,
        {
            "type": "service_account",
            "project_id": "placeholder-project",
            "private_key_id": key_id,
            "private_key": PEM,
            "client_email": "placeholder@placeholder-project.iam.example",
            "client_id": "100000000000000000001",
        },
    )
    env: dict[str, str] = {"GOOGLE_CLOUD_PROJECT": "placeholder-project"}
    body = [line for line in PEM.splitlines() if not line.startswith("-----")]
    with staged_secrets.trial_registry() as reg:
        inject_vertex_credentials(env, "google-vertex/gemini-placeholder")
        names = set(reg.names())
        file_text = env["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
        outs = [
            redact_trajectory_text(file_text),
            redact_trajectory_text(f"GOOGLE_APPLICATION_CREDENTIALS_JSON={file_text}"),
            *(redact_trajectory_text(f"line: {line}") for line in body),
        ]
        armour = redact_trajectory_text(f"-----BEGIN {ARMOUR}----- in the docs")
    assert "GOOGLE_APPLICATION_CREDENTIALS_JSON.private_key" in names
    assert not any(name.endswith("private_key_id") for name in names)
    for out in outs:
        assert not any(line in out for line in body), out
    assert "[masked:" not in armour


@pytest.mark.asyncio
async def test_a_sandbox_proxy_launch_registers_the_keys_it_uploads(monkeypatch):
    """A proxy run inside the sandbox gets the provider's keys, a Vertex
    credentials file and its own master key there: all are registered, in
    the trial's registry, before anything is uploaded."""
    from benchflow.providers import litellm_runtime

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    upstream = "upk5Rt8Yu2Io6Pa9Sd3Fg7Hj1Kl"
    adc = json.dumps({"type": "authorized_user", "refresh_token": REFRESH})
    with staged_secrets.trial_registry() as reg:
        with pytest.raises(AttributeError):  # no route: the launch stops after
            await litellm_runtime._start_sandbox_litellm(
                sandbox=None,
                route=None,
                master_key=PROXY_VALUE,
                agent_env={
                    "OPENAI_API_KEY": upstream,
                    "OPENAI_BASE_URL": "http://upstream.placeholder.example/v1",
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON": adc,
                },
                session_id="",
                agent_name="placeholder-agent",
            )
        assert reg.names() == [
            "GOOGLE_APPLICATION_CREDENTIALS_JSON.refresh_token",
            "OPENAI_API_KEY",
            "proxy master key",
        ]
    assert staged_secrets._RUN.names() == []


@pytest.mark.asyncio
async def test_a_replay_proxy_registers_its_upstream_key(monkeypatch):
    """`bench eval continue`'s replay proxy puts the upstream provider key
    into the sandbox with its config: it is registered first."""
    from benchflow.continue_run.sandbox_proxy import SandboxReplayProxy

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    upstream = "rpk2Wd5Fg8Hj1Kl4Zx7Cv0Bn3"
    with staged_secrets.trial_registry() as reg:
        with pytest.raises(Exception):  # noqa: B017 -- no sandbox: it stops after
            await SandboxReplayProxy.start(
                sandbox=object(),
                recorded=[],
                upstream_url="http://upstream.placeholder.example/v1",
                upstream_api_key=upstream,
                upstream_model="placeholder-model",
            )
        assert reg.names() == ["replay proxy upstream key"]


# --- what a staged value holds ----------------------------------------------


def test_a_json_value_with_no_secret_field_is_registered_whole_and_noted():
    """A staged credential whose JSON has no field on the secret-name list:
    which part is the secret is unknown, so the whole value is registered,
    and each string in it of 16 characters or more, and the record says so.
    Never silent."""
    from benchflow.trajectories.types import redact_trajectory_text

    leaf = "nfv3Gh7Jk1Lz5Xc9Vb2Nm6Qw"
    value = json.dumps({"value": leaf, "kind": "short"})
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("JSON_NOFIELD_TOKEN", value)
        whole = redact_trajectory_text(f"cat: {value}")
        bare = redact_trajectory_text(f"jq: {leaf}")
    assert reg.names() == ["JSON_NOFIELD_TOKEN", "JSON_NOFIELD_TOKEN.value"]
    assert whole == "cat: [masked:JSON_NOFIELD_TOKEN]"
    assert bare == "jq: [masked:JSON_NOFIELD_TOKEN.value]"
    assert reg.record()["notes"] == {"JSON_NOFIELD_TOKEN": staged_secrets.WHOLE_VALUE}


def test_a_json_array_is_registered_whole_and_noted():
    """A staged JSON array, whatever field names its objects hold: the whole
    value, and each string in it, and the record says so."""
    from benchflow.trajectories.types import redact_trajectory_text

    first, second = "arr1Qw2Er3Ty4Ui5Op6", "arr2Zx3Cv4Bn5Mm6Lk7"
    value = json.dumps([first, {"refresh_token": second}])
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("ARRAY_TOKEN", value)
        out = redact_trajectory_text(f"{first} and {second}")
        whole = redact_trajectory_text(value)
    assert reg.names() == [
        "ARRAY_TOKEN",
        "ARRAY_TOKEN.0",
        "ARRAY_TOKEN.1.refresh_token",
    ]
    assert out == "[masked:ARRAY_TOKEN.0] and [masked:ARRAY_TOKEN.1.refresh_token]"
    assert whole == "[masked:ARRAY_TOKEN]"
    assert reg.record()["notes"] == {"ARRAY_TOKEN": staged_secrets.WHOLE_ARRAY}


def test_overlapping_staged_values_are_masked_as_one_run():
    """Two staged values that overlap (one's tail is the other's head) leave
    no part of either, in a text or a non-text copy: one run, named for the
    longer, each value counted once."""
    from benchflow.trajectories.types import redact_trajectory_text

    x, y, z = "ovlA1b2C3d4", "E5f6G7h8I9j0K1", "l2M3n4O5p6Q7R8s9"
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("XY_TOKEN", x + y)
        staged_secrets.register("YZ_TOKEN", y + z)
        out = redact_trajectory_text(f"head {x}{y}{z} tail")
        filled = staged_secrets.mask_bytes(b"\0" + (x + y + z).encode() + b"\0")
    assert out == "head [masked:YZ_TOKEN] tail"
    assert filled == b"\0" + b"*" * len(x + y + z) + b"\0"
    assert reg.record()["masked"] == {"XY_TOKEN": 2, "YZ_TOKEN": 2}


def test_each_url_encoders_form_of_a_staged_value_is_masked():
    """A value with a space and the characters JavaScript's encoders keep
    (``!*'()``), as each encoder writes it, once and twice: Python's
    quote_plus, encodeURIComponent, encodeURI and a form body (WHATWG
    urlencoded: ``*`` kept, ``~`` escaped, a space as ``+``)."""
    from urllib.parse import quote, quote_plus

    from benchflow.trajectories.types import redact_trajectory_text

    value = "plc url'd (Q7)!*~Hq2Vx9Lm4"

    def component(text: str) -> str:  # keeps A-Z a-z 0-9 - _ . ! ~ * ' ( )
        return quote(text, safe="!*'()")

    def uri(text: str) -> str:  # and ; / ? : @ & = + $ , #
        return quote(text, safe="!*'();/?:@&=+$,#")

    def form(text: str) -> str:
        return quote_plus(text, safe="*").replace("~", "%7E")

    forms = [
        encode(once)
        for encode in (quote_plus, component, uri, form)
        for once in (value, encode(value))
    ]
    assert len({*forms, quote(value, safe="")}) > 4, "the encoders differ"
    with staged_secrets.trial_registry():
        staged_secrets.register("URL_TOKEN", value)
        for index, written in enumerate(forms):
            out = redact_trajectory_text(f"sent {written} once")
            assert out == "sent [masked:URL_TOKEN] once", index


def _indexed_database(path: Path, value: str) -> Path:
    """A sqlite file with an index over *value*'s column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("create table t (v text)")
    # The value sorts after these; filled with '*', it would sort before them.
    con.executemany(
        "insert into t values (?)", [(f"a{n:03d}{'m' * 20}",) for n in range(100)]
    )
    con.execute("insert into t values (?)", (value,))
    con.execute("create index ik on t (v)")
    con.commit()
    con.close()
    return path


def test_a_copy_that_the_filler_cannot_fill_safely_is_withheld(tmp_path):
    """A sqlite file with an index over a staged value fails its integrity
    check once the value is filled (the row no longer sorts where the index
    says), and a sqlite log can't be checked on its own: each is replaced by
    a stub that says why, and the trial's record names it by its path in the
    trial folder. A tar and an unindexed sqlite file keep the filler (the
    row above)."""
    from types import SimpleNamespace

    from benchflow.rollout._session_log import (
        WITHHELD_DATABASE,
        WITHHELD_LOG,
        _copy_dest,
        _redact,
        _redact_files,
    )

    trial = tmp_path / "trial"
    # A copy is redacted in a private folder before it is placed: it is
    # named where it lands in the trial folder.
    tree = tmp_path / "private" / "workspace"
    held = _indexed_database(tree / "-app" / "state.sqlite", REFRESH)
    rollout = SimpleNamespace(_rollout_paths=SimpleNamespace(rollout_dir=str(trial)))
    copies = trial / "agent" / "claude-sessions"
    db = _indexed_database(copies / "indexed.sqlite", REFRESH)
    live = tmp_path / "live.sqlite"
    con = sqlite3.connect(live)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma wal_autocheckpoint=0")
    con.execute("create table t (v text)")
    con.execute("insert into t values (?)", (REFRESH,))
    con.commit()
    wal = copies / "copy.sqlite-wal"
    wal.write_bytes((tmp_path / "live.sqlite-wal").read_bytes())
    con.close()
    assert REFRESH.encode() in wal.read_bytes()

    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("auth.json.refresh_token", REFRESH)
        _redact(db, [])
        _redact(wal, [])
        _redact_files([held], [], tree, _copy_dest(rollout, "codex-sessions"))
        staged_secrets.write_trial_record(trial, reg)
    for path, why in (
        (db, WITHHELD_DATABASE),
        (wal, WITHHELD_LOG),
        (held, WITHHELD_DATABASE),
    ):
        stub = path.read_text()
        assert stub.startswith(f"Withheld by BenchFlow: {why}."), stub
        assert "masking.json" in stub
    record = json.loads((trial / "integrity" / "masking.json").read_text())
    assert record["withheld"] == [
        {"path": "agent/claude-sessions/copy.sqlite-wal", "why": WITHHELD_LOG},
        {"path": "agent/claude-sessions/indexed.sqlite", "why": WITHHELD_DATABASE},
        {"path": "agent/codex-sessions/-app/state.sqlite", "why": WITHHELD_DATABASE},
    ]


def test_a_branchs_records_name_each_folders_own_withheld_files(tmp_path):
    """A branch masks its parent and children with one registry. A child's
    withheld copy is named in the child's record by its path from the
    trial's folder, and in no other folder's; the parent's copy at the same
    path, which held nothing staged, is intact and unnamed. Each record says
    its counts are the branch's (``branch_total``)."""
    from benchflow.rollout._session_log import WITHHELD_DATABASE, _redact

    trial = tmp_path / "trial"
    same = Path("agent", "claude-sessions", "state.sqlite")
    mine = _indexed_database(trial / same, "nothing staged in this one")
    child = trial / "branches" / "f1" / "children" / "c1"
    sibling = trial / "branches" / "f1" / "children" / "c2"
    sibling.mkdir(parents=True)
    theirs = _indexed_database(child / same, REFRESH)
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("auth.json.refresh_token", REFRESH)
        _redact(mine, [])
        _redact(theirs, [])
        staged_secrets.write_trial_record(trial, reg)
    records = {
        folder: json.loads((folder / "integrity" / "masking.json").read_text())
        for folder in (trial, child, sibling)
    }
    assert mine.read_bytes().startswith(b"SQLite format 3")
    assert records[trial]["withheld"] == records[sibling]["withheld"] == []
    assert records[child]["withheld"] == [
        {
            "path": "branches/f1/children/c1/agent/claude-sessions/state.sqlite",
            "why": WITHHELD_DATABASE,
        }
    ]
    totals = [record["branch_total"] for record in records.values()]
    assert totals[0]["auth.json.refresh_token"] >= 1
    assert totals == [totals[0]] * 3
    assert not any("masked" in record for record in records.values())


@pytest.mark.asyncio
async def test_the_agents_own_output_log_is_masked(tmp_path):
    """What the agent prints outside the protocol (its debug lines),
    BenchFlow writes to agent/<name> itself: through the redaction layer.
    The record names that file among those masked."""
    from benchflow._utils.artifact_names import agent_log_name
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.acp.container_transport import ContainerTransport

    class Process:
        stderr_tail = ""

        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        async def readline(self) -> bytes:
            return self._lines.pop(0)

        async def close(self) -> None:
            return None

    name = agent_log_name("placeholder-agent")
    lines = [
        f"debug: sending auth {TOKEN}\n".encode(),
        b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n',
    ]
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("BF_STAGED_TOKEN", TOKEN)
        transport = ContainerTransport(
            container_process=Process(lines),  # type: ignore[arg-type]
            command="placeholder-agent",
            agent_log_path=ReachablePath.rollout(tmp_path, "agent") / name,
        )
        assert (await transport.receive())["id"] == 1
        await transport.close()
        staged_secrets.write_trial_record(tmp_path, reg, agent="placeholder-agent")
    log = (tmp_path / "agent" / name).read_text()
    assert "debug: sending auth [masked:BF_STAGED_TOKEN]" in log
    assert _scan(log, {"BF_STAGED_TOKEN": TOKEN}) == []
    record = json.loads((tmp_path / "integrity" / "masking.json").read_text())
    assert f"agent/{name}" in record["applied_to"]


# --- the trial's registry, whoever drives the trial ------------------------


def test_a_driver_binds_its_trials_registry_once_and_its_run_keeps_it():
    """A verb that drives a trial binds the registry for its rollout; the
    rollout's own run() inside keeps that one (owner=), and any other trial
    inside gets its own."""
    rollout, other = object(), object()
    with staged_secrets.trial_registry(owner=rollout) as outer:
        with staged_secrets.trial_registry(owner=rollout) as again:
            assert again is outer
        with staged_secrets.trial_registry(owner=other) as theirs:
            assert theirs is not outer
            assert staged_secrets.current() is theirs
        with staged_secrets.trial_registry() as unowned:
            assert unowned is not outer
        assert staged_secrets.current() is outer
    assert staged_secrets.current() is None


@pytest.mark.asyncio
async def test_after_many_proxy_launches_a_trial_masks_with_its_own_value_alone(
    monkeypatch,
):
    """Each launch of a model proxy (on the host or in the sandbox) mints its
    own master key, staged for that trial. After 400 launches in 400 trials
    the run's registry holds none of them, and the next trial's first mask
    compiles its own value alone."""
    from benchflow.providers import litellm_runtime
    from benchflow.trajectories.types import redact_trajectory_text

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    for index in range(400):
        key = f"mk{index:04d}Zq8Xw2Ce4Rv6Tb"
        with staged_secrets.trial_registry() as reg:
            with pytest.raises(AttributeError):  # no route: the launch stops after
                if index % 2:
                    await litellm_runtime._start_sandbox_litellm(
                        sandbox=None,
                        route=None,
                        master_key=key,
                        agent_env={},
                        session_id="",
                        agent_name="placeholder-agent",
                    )
                else:
                    await litellm_runtime._start_host_litellm(
                        route=None,
                        master_key=key,
                        agent_env={},
                        environment="docker",
                        session_id="",
                        agent_name="placeholder-agent",
                    )
            assert reg.names() == ["proxy master key"]
    assert staged_secrets._RUN.names() == []
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("ONE_TOKEN", REFRESH)
        assert reg.names() == ["ONE_TOKEN"]
        out = redact_trajectory_text(f"x {REFRESH} y")
    assert out == "x [masked:ONE_TOKEN] y"


def _task(root: Path) -> Path:
    task = root / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 900\n')
    (task / "instruction.md").write_text("Do the thing.\n")
    return task


def _continue_folder(root: Path, monkeypatch) -> Path:
    """A run folder `bench eval continue` loads (one recorded exchange), and
    its task."""
    from benchflow.continue_run import orchestrator
    from tests.continue_run._helpers import completion, exchange, write_run_folder

    folder = write_run_folder(
        root / "run", exchanges=[exchange(completion(content="a"))]
    )
    task = _task(root)
    monkeypatch.setattr(orchestrator, "resolve_task_path", lambda *args: task)
    monkeypatch.setattr(orchestrator, "require_replayable_task", lambda run, path: path)
    return folder


def _post(base: str, text: str) -> None:
    import urllib.request

    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(
            {"model": "replay", "messages": [{"role": "user", "content": text}]}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as answer:
        answer.read()


@pytest.mark.asyncio
async def test_a_continue_on_the_host_masks_its_stitched_trajectory(
    tmp_path, monkeypatch
):
    """`bench eval continue` on the host runs the trial through Rollout.run,
    then rewrites trajectory/llm_trajectory.jsonl from the live calls after
    run() returns. What the trial staged is masked there too, and the
    trial's record, written after it, counts those masks."""
    from types import SimpleNamespace

    from benchflow.continue_run import orchestrator
    from benchflow.rollout import Rollout
    from tests.continue_run._helpers import completion

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    folder = _continue_folder(tmp_path, monkeypatch)
    monkeypatch.setattr(
        orchestrator, "_host_proxy_binding", lambda env: ("127.0.0.1", "127.0.0.1")
    )

    class Live:  # the live model, past the recorded turn: it repeats the value
        def __init__(self, model: str) -> None:
            self.model = model

        def __call__(self, body: dict) -> dict:
            return completion(content=f"you printed {TOKEN}")

    monkeypatch.setattr(orchestrator, "LiteLLMLiveForwarder", Live)
    out = tmp_path / "continued-trial"
    out.mkdir()

    async def create(config):
        rollout = Rollout(config)
        rollout._rollout_dir = out

        async def lifecycle():
            # The agent's launch env, staged as agent_launch_env stages it;
            # then the agent prints the value into its next request.
            staged_secrets.register_env({"BF_STAGED_TOKEN": TOKEN})
            base = config.agent_env["LLM_BASE_URL"]
            for _ in range(2):  # the recorded turn, then a live one
                await asyncio.to_thread(_post, base, f"printed {TOKEN}")
            return SimpleNamespace(rewards={"reward": 1.0}, error=None)

        rollout._run_with_scoring = lifecycle
        return rollout

    monkeypatch.setattr(Rollout, "create", create)
    result = await orchestrator.continue_run(
        folder, model="test-model", proxy_mode="host"
    )
    assert result.n_live == 1
    stitched = (out / "trajectory" / "llm_trajectory.jsonl").read_text()
    assert "[masked:BF_STAGED_TOKEN]" in stitched
    assert _scan(stitched, {"BF_STAGED_TOKEN": TOKEN}) == []
    record = json.loads((out / "integrity" / "masking.json").read_text())
    assert record["masked"].get("BF_STAGED_TOKEN", 0) >= 2, record["masked"]
    assert staged_secrets.current() is None


def _sandbox_continue(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str]:
    """A run folder, and `bench eval continue`'s sandbox route ready to run
    on it: the real route and replay proxy; only the sandbox's side is a
    stand-in (its commands succeed, and the proxy's live log holds one live
    call, in which the agent printed the upstream key). The run folder, the
    continued trial's folder and that key."""
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.continue_run import orchestrator, sandbox_proxy
    from benchflow.continue_run.run_folder import load_run_folder
    from benchflow.rollout import Rollout
    from benchflow.trajectories.types import LLMExchange, LLMRequest, LLMResponse
    from tests.continue_run._helpers import completion

    monkeypatch.setattr(staged_secrets, "_RUN", staged_secrets.StagedSecrets())
    folder = _continue_folder(tmp_path, monkeypatch)
    upstream = "rpk2Wd5Fg8Hj1Kl4Zx7Cv0Bn3"
    # The folder the route gives the continued trial.
    run = load_run_folder(folder)
    out = run.path.parent / "continued" / orchestrator.continued_rollout_name(run)
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}")

    async def create(config):
        rollout = Rollout(config)
        rollout._rollout_dir = out
        rollout._rollout_name = out.name
        rollout._started_at = datetime.now()
        rollout._env = SimpleNamespace(exec=AsyncMock())
        for name in ("setup", "start", "install_agent", "cleanup"):
            setattr(rollout, name, AsyncMock())

        async def turn(_steps):
            raise RuntimeError("the agent's turn ends here")

        rollout._run_steps = turn
        monkeypatch.setattr(
            rollout._planes, "resolve_agent_env", lambda *args: {}, raising=False
        )
        return rollout

    monkeypatch.setattr(Rollout, "create", create)
    monkeypatch.setattr(
        "benchflow.providers.runtime.ensure_litellm_runtime",
        AsyncMock(
            return_value=(
                {
                    "LLM_BASE_URL": "http://upstream.placeholder.example/v1",
                    "LLM_API_KEY": upstream,
                    "LLM_MODEL": "placeholder-model",
                },
                object(),
            )
        ),
    )
    monkeypatch.setattr(
        "benchflow.providers.runtime.stop_provider_runtime", AsyncMock()
    )
    # The host route's live model, which continue_run makes before it picks
    # a route; this route's live calls go through the proxy in the sandbox.
    monkeypatch.setattr(orchestrator, "LiteLLMLiveForwarder", lambda model: None)
    live = LLMExchange(
        request=LLMRequest(
            body={"messages": [{"role": "user", "content": f"printed {upstream}"}]}
        ),
        response=LLMResponse(
            status_code=200, body=completion(content=f"you printed {upstream}")
        ),
    )

    async def ran(*args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.trust.state.admit", lambda *args, **kwargs: None)
    monkeypatch.setattr(sandbox_proxy, "exec_raw", ran)
    monkeypatch.setattr(sandbox_proxy, "_upload_text", AsyncMock())
    monkeypatch.setattr(
        sandbox_proxy, "root_python_command", AsyncMock(return_value="probe")
    )
    monkeypatch.setattr(
        sandbox_proxy,
        "_read_remote_text",
        AsyncMock(return_value=live.model_dump_json() + "\n"),
    )
    return folder, out, upstream


@pytest.mark.asyncio
async def test_a_continue_in_the_sandbox_masks_what_its_trial_staged(
    tmp_path, monkeypatch
):
    """`bench eval continue` in the sandbox drives the trial's steps itself.
    The replay proxy registers the upstream key it puts into the sandbox
    while the trial's registry is bound; the live call the agent printed it
    in is masked in the stitched trajectory, and the record is written."""
    from benchflow.continue_run import orchestrator

    folder, out, upstream = _sandbox_continue(tmp_path, monkeypatch)
    await orchestrator.continue_run(folder, model="test-model", proxy_mode="sandbox")
    stitched = (out / "trajectory" / "llm_trajectory.jsonl").read_text()
    assert "[masked:replay proxy upstream key]" in stitched
    assert _scan(stitched, {"replay proxy upstream key": upstream}) == []
    record = json.loads((out / "integrity" / "masking.json").read_text())
    assert "replay proxy upstream key" in record["registered"]
    assert record["masked"].get("replay proxy upstream key", 0) >= 2, record
    assert staged_secrets.current() is None


@pytest.mark.asyncio
async def test_a_continue_in_the_sandbox_that_raises_still_writes_its_record(
    tmp_path, monkeypatch
):
    """The route raising after the agent's turn (its last metadata write
    fails): the trial's record is written all the same."""
    from benchflow.continue_run import orchestrator

    folder, out, _ = _sandbox_continue(tmp_path, monkeypatch)

    def fails(*args, **kwargs):
        raise RuntimeError("the metadata write fails")

    monkeypatch.setattr(orchestrator, "update_continued_metadata", fails)
    with pytest.raises(RuntimeError, match="metadata write fails"):
        await orchestrator.continue_run(
            folder, model="test-model", proxy_mode="sandbox"
        )
    record = json.loads((out / "integrity" / "masking.json").read_text())
    assert "replay proxy upstream key" in record["registered"]
    assert staged_secrets.current() is None


def test_the_bundle_carries_the_trials_masking_record(tmp_path):
    """`bench bundle` carries ``integrity/masking.json``, so a reader of the
    bundle sees what was masked and what was not swept (the bundle carries
    verifier/ and artifacts/, which are not)."""
    import zipfile

    from benchflow.evidence_bundle import build_bundle

    trial = tmp_path / "job" / "task__abc"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    with staged_secrets.trial_registry() as reg:
        staged_secrets.write_record(trial, reg)
    result = build_bundle(trial.parent, tmp_path / "run.zip")
    with zipfile.ZipFile(result.path) as archive:
        names = archive.namelist()
        assert "task__abc/integrity/masking.json" in names, names
        record = json.loads(archive.read("task__abc/integrity/masking.json"))
    assert record["not_swept"]["folders"] == ["verifier/", "agent/", "artifacts/"]


def test_an_external_account_file_registers_its_whole_value_not_its_public_strings():
    """A workload-identity credentials file names no secret field: the whole
    value is registered, but not the strings it holds in the open (its
    type, its URLs, its token-type URN, its token file's path), so ordinary
    output that holds them is left as it is."""
    from benchflow.trajectories.types import redact_trajectory_text

    token_url = "https://sts.placeholder.example/v1/token"
    urn = "urn:ietf:params:oauth:token-type:jwt"
    path = "/var/run/placeholder/token-file"
    value = json.dumps(
        {
            "type": "external_account",
            "audience": "//iam.placeholder.example/projects/0/pools/p/providers/q",
            "subject_token_type": urn,
            "token_url": token_url,
            "credential_source": {"file": path},
            "service_account_impersonation_url": (
                "https://iamcredentials.placeholder.example/v1/accounts/x:token"
            ),
        }
    )
    ordinary = f"external_account via {token_url} with {urn} from {path}"
    with staged_secrets.trial_registry() as reg:
        staged_secrets.register("GOOGLE_APPLICATION_CREDENTIALS_JSON", value)
        whole = redact_trajectory_text(f"cat: {value}")
        left = redact_trajectory_text(ordinary)
    assert reg.names() == ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    assert whole == "cat: [masked:GOOGLE_APPLICATION_CREDENTIALS_JSON]"
    assert left == ordinary
    notes = reg.record()["notes"]
    assert notes == {"GOOGLE_APPLICATION_CREDENTIALS_JSON": staged_secrets.WHOLE_VALUE}
    assert notes["GOOGLE_APPLICATION_CREDENTIALS_JSON"].endswith(
        "except its type, URLs, URNs and file paths"
    )


class _Stop(Exception):
    """Ends a row's setup() right after its registry is claimed."""


def _manual_rollouts(tmp_path: Path, monkeypatch, *names: str) -> list[Any]:
    """Rollouts an SDK caller drives step by step, each with its own folder.
    Each one's setup() stops once its registry is claimed; finalize() takes
    its own seam; ``stage`` is a public step that stages a value and returns
    a text holding it, redacted, as a step's own record would."""
    import benchflow.rollout as rollout_module
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.trajectories.types import redact_trajectory_text

    def stop(_rollout):
        raise _Stop

    monkeypatch.setattr(rollout_module, "require_safe_branch_world", stop)

    class Manual(Rollout):
        @rollout_module._with_staged_registry
        async def stage(self, name: str, value: str) -> str:
            staged_secrets.register(name, value)
            return redact_trajectory_text(f"printed {value}")

        @rollout_module._with_staged_registry
        async def within(self, step):
            return await step()

    task = _task(tmp_path)
    made = []
    for name in names:
        rollout = Manual(RolloutConfig(task_path=task, agent="openhands"))
        rollout._rollout_dir = tmp_path / name
        rollout._rollout_dir.mkdir()
        rollout._completed_result = SimpleNamespace()  # finalize's own seam
        made.append(rollout)
    return made


def _record_of(rollout) -> dict:
    return json.loads((rollout._rollout_dir / "integrity" / "masking.json").read_text())


async def _set_up(rollout, **kwargs) -> None:
    with pytest.raises(_Stop):
        await rollout.setup(**kwargs)


@pytest.mark.asyncio
async def test_rollouts_set_up_together_each_mask_and_record_their_own(
    tmp_path, monkeypatch
):
    """Two rollouts set up before either is driven, then driven and finished
    one after the other: each masks what it staged with its own registry,
    whichever finished first, and each folder has its own record. A third,
    finished having staged nothing, gets no record from its steps."""
    a, b, idle = _manual_rollouts(tmp_path, monkeypatch, "a", "b", "idle")
    await _set_up(a)
    await _set_up(b)
    await _set_up(idle)
    assert a._staged is not b._staged
    assert staged_secrets.current() is None  # no binding outlives a step
    assert await a.stage("A_TOKEN", TOKEN) == "printed [masked:A_TOKEN]"
    await a.finalize()
    assert await b.stage("B_TOKEN", REFRESH) == "printed [masked:B_TOKEN]"
    await b.finalize()
    await idle.finalize()
    assert _record_of(a)["registered"] == ["A_TOKEN"]
    assert _record_of(b)["registered"] == ["B_TOKEN"]
    assert _record_of(b)["masked"] == {"B_TOKEN": 1}
    # One that staged nothing gets no integrity/ folder from its steps.
    assert not (idle._rollout_dir / "integrity").exists()


@pytest.mark.asyncio
async def test_a_rollout_left_unfinished_keeps_its_record_and_masks_no_other(
    tmp_path, monkeypatch
):
    """A rollout set up and driven, then left (no finalize); another set up,
    driven and finished after it: the second has its own registry and
    record, and the first's record is the one its last step left."""
    a, b = _manual_rollouts(tmp_path, monkeypatch, "a", "b")
    await _set_up(a)
    await a.stage("A_TOKEN", TOKEN)
    await _set_up(b)
    assert b._staged is not a._staged
    assert await b.stage("B_TOKEN", REFRESH) == "printed [masked:B_TOKEN]"
    await b.finalize()
    assert _record_of(a)["registered"] == ["A_TOKEN"]
    assert _record_of(a)["masked"] == {"A_TOKEN": 1}
    assert _record_of(b)["registered"] == ["B_TOKEN"]


@pytest.mark.asyncio
async def test_a_branch_child_and_a_verifiers_child_share_their_trials_registry(
    tmp_path, monkeypatch
):
    """Set up inside its parent's step, a branch child (it shares the
    parent's tree) and a verifier's child (set up verifier-only) mask with
    the parent's registry and write no record of their own. A rollout of
    its own set up there gets its own."""
    parent, child, grader, other = _manual_rollouts(
        tmp_path, monkeypatch, "parent", "child", "grader", "other"
    )
    child._tree = parent._tree
    await _set_up(parent)

    async def inside():
        await _set_up(child)
        await _set_up(grader, verifier_only=True)
        await _set_up(other)

    await parent.within(inside)
    assert child._staged is parent._staged and grader._staged is parent._staged
    assert other._staged is not parent._staged
    assert await child.stage("CHILD_TOKEN", TOKEN) == "printed [masked:CHILD_TOKEN]"
    await child.finalize()
    await grader.finalize()
    assert not (child._rollout_dir / "integrity").exists()
    assert not (grader._rollout_dir / "integrity").exists()
    await parent.finalize()
    assert _record_of(parent)["registered"] == ["CHILD_TOKEN"]


@pytest.mark.asyncio
async def test_a_rollout_stepped_under_a_verbs_registry_claims_it(
    tmp_path, monkeypatch
):
    """A verb that binds a registry before it has the rollout, naming none
    (the sandbox route of `bench eval continue`): the rollout's first public
    step claims it, and so does a verifier's child set up there. What a step
    stages lands in it, so the verb's own write after the steps is masked;
    the rollout's steps write no record, which is the verb's to write."""
    from benchflow.trajectories.types import redact_trajectory_text

    rollout, grader = _manual_rollouts(tmp_path, monkeypatch, "trial", "grader")
    with staged_secrets.trial_registry() as verbs:
        await _set_up(rollout)
        assert rollout._staged is verbs
        assert await rollout.stage("A_TOKEN", TOKEN) == "printed [masked:A_TOKEN]"
        await _set_up(grader, verifier_only=True)
        assert grader._staged is verbs
        assert redact_trajectory_text(f"stitched {TOKEN}") == (
            "stitched [masked:A_TOKEN]"
        )
        await rollout.finalize()
    assert verbs.names() == ["A_TOKEN"]
    assert not (rollout._rollout_dir / "integrity").exists()


@pytest.mark.asyncio
async def test_a_rollout_set_up_in_one_task_and_finished_in_another(
    tmp_path, monkeypatch
):
    """A prestart sets a rollout up in a task of its own; the run finishes
    it in another. Nothing is released across them (no binding outlives a
    step): its later steps mask with the registry setup claimed, and
    finalize() writes its record."""
    (rollout,) = _manual_rollouts(tmp_path, monkeypatch, "trial")
    with pytest.raises(_Stop):
        await asyncio.create_task(rollout.setup())
    claimed = rollout._staged
    assert claimed is not None and claimed.owner is rollout
    assert await rollout.stage("A_TOKEN", TOKEN) == "printed [masked:A_TOKEN]"
    await rollout.finalize()
    assert rollout._staged is claimed
    assert _record_of(rollout)["registered"] == ["A_TOKEN"]
    from benchflow._utils.artifact_names import agent_log_name

    assert f"agent/{agent_log_name('openhands')}" in _record_of(rollout)["applied_to"]


@pytest.mark.parametrize("review", ["first", "again"])
def test_a_reviewers_copy_of_the_trial_carries_the_masking_record(tmp_path, review):
    """A withheld file's stub points to integrity/masking.json: a reviewer's
    copy of the trial carries it. A re-review's copy, held to the rollout
    layout, takes it alone from integrity/, and names the rest there as left
    out."""
    from benchflow.review.config import Rubric
    from benchflow.review.wrapper import assemble_review_task

    rubric = Rubric.model_validate(
        {"criteria": [{"name": "sound", "description": "d", "guidance": "PASS"}]}
    )
    trial = tmp_path / "rollout-a"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    solver = {"task_name": "t", "rewards": {"reward": 1.0}, "verifier_error": None}
    result = dict(solver)
    if review == "again":
        (trial / "solver.json").write_text(json.dumps(solver))
        result["scoring"] = {"status": "complete", "rubric_reward": 0.5}
    (trial / "result.json").write_text(json.dumps(result))
    with staged_secrets.trial_registry() as reg:
        staged_secrets.write_record(trial, reg)
    (trial / "integrity" / "other-record.json").write_text("{}")
    dest, _ = assemble_review_task(trial, None, rubric, tmp_path / "w")
    copy = dest / "evidence" / "trial" / "integrity"
    assert json.loads((copy / "masking.json").read_text())["withheld"] == []
    if review == "again":
        assert sorted(p.name for p in copy.iterdir()) == ["masking.json"]
        left = json.loads((dest / "evidence" / "trial-left-out.json").read_text())
        assert "integrity/other-record.json" in left["left_out"]
        assert "integrity" not in left["left_out"]


def test_a_re_reviews_copy_names_a_linked_masking_record_as_left_out(tmp_path):
    """A linked integrity/masking.json is not copied (no link is), and the
    left-out record names it."""
    from benchflow.review.config import Rubric
    from benchflow.review.wrapper import assemble_review_task

    rubric = Rubric.model_validate(
        {"criteria": [{"name": "sound", "description": "d", "guidance": "PASS"}]}
    )
    trial = tmp_path / "rollout-a"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    solver = {"task_name": "t", "rewards": {"reward": 1.0}, "verifier_error": None}
    (trial / "solver.json").write_text(json.dumps(solver))
    result = {**solver, "scoring": {"status": "complete", "rubric_reward": 0.5}}
    (trial / "result.json").write_text(json.dumps(result))
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}")
    (trial / "integrity").mkdir()
    (trial / "integrity" / "masking.json").symlink_to(elsewhere)
    (trial / "integrity" / "other-record.json").write_text("{}")
    dest, _ = assemble_review_task(trial, None, rubric, tmp_path / "w")
    copy = dest / "evidence" / "trial" / "integrity"
    assert not (copy / "masking.json").exists()
    left = json.loads((dest / "evidence" / "trial-left-out.json").read_text())
    assert "integrity/masking.json" in left["left_out"]
    assert "integrity/other-record.json" in left["left_out"]
