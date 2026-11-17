"""config.json and evaluation.json keep no secret-looking config override value.

Guards the audit fix on branch ``audit-config-override-split``
(finding CRED-3; the lead's ruling of 2026-11-13: record the names,
with the values supplied again on regrade, resume and continue, like
agent_env). ``--config-override`` is free-form (a sandbox env, MCP headers),
and config.json stored the whole overlay under ``patch``, a key or password in
it included, because regrade, verifier recovery, continue and resume read that
record back and apply it.

The record now keeps the values that cannot be a credential (``patch``), the
paths of the ones left out (``withheld``) and the form a resume compares
(``compare``). A value is left out when its key path names a credential or a
free-form mapping (env, token, key, password, auth and the like, at any
level), or when the value itself has a secret's shape: the lead's tightening
of 2026-11-13, since shape alone misses a password that looks harmless. A
command that runs part of the job again takes the withheld values from
``BENCHFLOW_CONFIG_OVERRIDE`` and stops with a named error when it needs them
and has none. Every value here is made up.
"""

from __future__ import annotations

import datetime
import json
import re

import pytest

from benchflow._utils.config_override import (
    CONFIG_OVERRIDE_ENV,
    ConfigOverrideNeededError,
    apply_config_override,
    load_config_override,
    overlay_hash,
    overlay_record,
    recorded_overlay,
    validate_overlay,
)
from benchflow.continue_run.orchestrator import build_rollout_config
from benchflow.continue_run.run_folder import RunFolder
from benchflow.errors import UserError
from benchflow.evaluation import (
    Evaluation,
    _check_resume_mismatch,
    _config_from_record,
)
from benchflow.task.config import (
    AgentConfig,
    ArtifactConfig,
    HealthcheckConfig,
    MCPServerConfig,
    SandboxConfig,
    SetupCommandConfig,
    TpuSpec,
)
from tests.test_credential_names import CREDENTIALS, SETTINGS
from tests.test_resume_refuses_changed_settings import (
    _bench,
    _fake_rollouts,
    _refused,
    _resumed,
    _squash,
)
from tests.test_resume_settings_match import (
    _LOGIN_FIX,
    NEW_HF,
    OLD_HF,
    _config,
    _first_job,
    _override,
    _refusal,
    _tasks,
)

SAFE = {"agent": {"timeout_sec": 99}}
# What is left of _override() in the record: nothing under an env is stored.
KEPT = {"sandbox": {"env": {}}}
WITHHELD = ["sandbox.env.HF_TOKEN", "sandbox.env.HUB", "sandbox.env.MY_FLAG"]


@pytest.fixture(autouse=True)
def _no_supplied_overlay(monkeypatch):
    monkeypatch.delenv(CONFIG_OVERRIDE_ENV, raising=False)


def test_the_record_keeps_the_safe_values_and_names_the_others():
    overlay = _override(OLD_HF, "flag-one")

    record = overlay_record(overlay)

    assert OLD_HF not in json.dumps(record) and "flag-one" not in json.dumps(record)
    assert record["patch"] == KEPT
    assert record["withheld"] == WITHHELD
    stored = {name: record[name] for name in ("patch", "withheld", "compare")}
    assert (record["keys"], record["sha256"]) == (["sandbox"], overlay_hash(stored))


@pytest.mark.parametrize(
    "name",
    [
        "api_password",
        "API_KEY",
        "authToken",
        "Cookie",
        "db_passwd",
        "PWD",
        "passphrase",
        "bearer",
        "credentials",
        "client_secret",
        "api_key",
        "X-Auth-Token",
        "githubToken",
        "APIKey",
        "apikey",
        "Authorization",
    ],
)
def test_a_value_under_a_secret_name_is_not_stored_however_harmless_it_looks(name):
    """The key-name rule: the value has no secret's shape, its name does."""
    record = overlay_record({"agent": {name: "tuesday", "timeout_sec": 99}})

    assert "tuesday" not in json.dumps(record)
    assert record["patch"] == SAFE
    assert record["withheld"] == [f"agent.{name}"]
    # Not compared either: supplying another value for it is a rotated key.
    rotated = {"agent": {name: "wednesday", "timeout_sec": 99}}
    assert recorded_overlay(record, rotated, what="regrade") == rotated


def test_a_setting_under_an_env_is_stored_and_compared():
    """The lead's ruling of 2026-11-13 (sixth round), which reverses the
    first rounds' case here: an env entry named like a setting
    (MAX_THINKING_TOKENS, MAX_OUTPUT_TOKENS: tokens beside a companion word)
    is stored in ``patch`` and compared as given, so another value is
    refused by its path. Every other env entry stays out of ``patch``:
    a credential's name is withheld, any other is compared by a digest."""
    overlay = {
        "sandbox": {"env": {"MAX_THINKING_TOKENS": "1024", "MAX_OUTPUT_TOKENS": "4096"}}
    }
    record = overlay_record(overlay)

    assert (record["patch"], record["compare"], record["withheld"]) == (
        overlay,
        overlay,
        [],
    )
    for name in ("MAX_THINKING_TOKENS", "MAX_OUTPUT_TOKENS"):
        changed = {"sandbox": {"env": {**overlay["sandbox"]["env"], name: "8192"}}}
        with pytest.raises(
            ConfigOverrideNeededError, match=re.escape(f"(sandbox.env.{name} changed)")
        ):
            recorded_overlay(record, changed, what="regrade")


def test_a_credential_under_an_env_stays_withheld_beside_a_setting():
    """The lead's ruling of 2026-11-13 (sixth round): MS_GRAPH_TOKEN (token
    with no limit, count or time word) under an env is withheld and a
    rotated one resumes; a key-shaped value under a setting's name is
    withheld too (the shape rule comes first)."""

    def env(graph: str, tokens: str) -> dict:
        return {
            "sandbox": {"env": {"MS_GRAPH_TOKEN": graph, "MAX_OUTPUT_TOKENS": tokens}}
        }

    record = overlay_record(env("tuesday-made-up", "4096"))
    assert record["withheld"] == ["sandbox.env.MS_GRAPH_TOKEN"]
    assert record["patch"] == {"sandbox": {"env": {"MAX_OUTPUT_TOKENS": "4096"}}}
    assert "tuesday-made-up" not in json.dumps(record)
    rotated = env("wednesday-made-up", "4096")
    assert recorded_overlay(record, rotated, what="regrade") == rotated

    shaped = overlay_record(env("tuesday-made-up", SK_ANT))
    assert "sandbox.env.MAX_OUTPUT_TOKENS" in shaped["withheld"]
    assert SK_ANT not in json.dumps(shaped)


def test_nothing_under_a_headers_mapping_is_stored():
    record = overlay_record({"sandbox": {"mcp": {"headers": {"X-Trace": "abc123"}}}})

    assert "abc123" not in json.dumps(record)
    assert record["withheld"] == ["sandbox.mcp.headers.X-Trace"]


@pytest.mark.parametrize(
    "name",
    ["author", "hotkeys", "bypass", "monkey", "keyboard", "passenger", "tokenizer"],
)
def test_a_name_that_only_contains_a_secret_word_is_stored(name):
    """The lead's ruling of 2026-11-13: key names are matched by their words
    (split on _ . - / and camelCase), not by substring."""
    overlay = {"agent": {name: "tuesday", "timeout_sec": 99}}
    record = overlay_record(overlay)

    assert record["withheld"] == []
    assert record["patch"] == overlay


def test_a_secret_name_anywhere_on_the_path_withholds_what_is_below_it():
    record = overlay_record({"agent": {"auth": {"mode": "plain", "user": "sam"}}})

    assert "plain" not in json.dumps(record) and "sam" not in json.dumps(record)
    assert record["withheld"] == ["agent.auth.mode", "agent.auth.user"]


def test_a_secret_shaped_value_under_a_harmless_name_is_not_stored():
    """The shape rule: the name says nothing, the value is a key's shape."""
    record = overlay_record({"agent": {"note": OLD_HF, "timeout_sec": 99}})

    assert OLD_HF not in json.dumps(record)
    assert record["patch"] == SAFE
    assert record["withheld"] == ["agent.note"]


def test_a_flag_under_an_env_is_not_stored_and_a_change_to_it_is_still_seen():
    overlay = {"sandbox": {"env": {"MY_FLAG": "flag-one"}}}
    record = overlay_record(overlay)

    assert "flag-one" not in json.dumps(record)
    assert record["withheld"] == ["sandbox.env.MY_FLAG"]
    assert recorded_overlay(record, overlay, what="regrade") == overlay
    with pytest.raises(ConfigOverrideNeededError, match="is not the recorded one"):
        recorded_overlay(
            record, {"sandbox": {"env": {"MY_FLAG": "flag-two"}}}, what="regrade"
        )


def test_a_password_in_a_plain_field_is_not_stored_either():
    record = overlay_record({"agent": {"note": _LOGIN_FIX, "timeout_sec": 99}})

    assert "admin123" not in json.dumps(record)
    assert record["patch"] == {"agent": {"timeout_sec": 99}}
    assert record["withheld"] == ["agent.note"]


def test_config_json_holds_no_secret_looking_override_value(tmp_path):
    from benchflow.rollout._results import _write_config
    from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="test",
        model=None,
        environment="docker",
        skill_policy=resolve_task_skill_policy(
            task_path=tmp_path / "task",
            skill_mode=SKILL_MODE_NO_SKILL,
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        ),
        sandbox_user="agent",
        context_root=None,
        timeout=300,
        started_at=datetime.datetime.now(),
        agent_env={},
        config_override=_override(OLD_HF, "flag-one"),
    )

    text = (tmp_path / "config.json").read_text()
    assert OLD_HF not in text
    stored = json.loads(text)["config_override"]
    assert (stored["patch"], stored["withheld"]) == (KEPT, WITHHELD)


def _recorded_job(tmp_path, overlay) -> dict:
    """The config record of a job that ran with *overlay*, from evaluation.json."""
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(config_override=overlay),
    )
    job._write_evaluation_record()
    text = (job.job_dir / "evaluation.json").read_text()
    assert OLD_HF not in text
    return json.loads(text)["config"]


def test_evaluation_json_holds_no_secret_looking_override_value(tmp_path):
    recorded = _recorded_job(tmp_path, _override(OLD_HF, "flag-one"))

    assert recorded["config_override"]["patch"] == KEPT
    assert recorded["config_override"]["withheld"] == WITHHELD


def test_nothing_is_asked_for_when_nothing_was_withheld():
    record = overlay_record(SAFE)

    assert record["withheld"] == []
    assert recorded_overlay(record, what="regrade") == SAFE


def test_a_withheld_value_that_is_not_supplied_stops_the_command():
    record = overlay_record(_override(OLD_HF, "flag-one"))

    with pytest.raises(ConfigOverrideNeededError) as refused:
        recorded_overlay(record, what="regrade")

    assert ", ".join(WITHHELD) in str(refused.value)
    assert CONFIG_OVERRIDE_ENV in refused.value.hint
    assert OLD_HF not in str(refused.value) + refused.value.hint


def test_the_values_are_supplied_again_inline_or_from_a_file(tmp_path, monkeypatch):
    overlay = _override(OLD_HF, "flag-one")
    record = overlay_record(overlay)

    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(overlay))
    assert recorded_overlay(record, what="regrade") == overlay

    (tmp_path / "overlay.json").write_text(json.dumps(overlay))
    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, f"@{tmp_path / 'overlay.json'}")
    assert recorded_overlay(record, what="verifier recovery") == overlay


def test_a_rotated_key_is_accepted_and_another_setting_is_not():
    record = overlay_record(_override(OLD_HF, "flag-one"))

    rotated = _override(NEW_HF, "flag-one")
    assert recorded_overlay(record, rotated, what="regrade") == rotated

    with pytest.raises(ConfigOverrideNeededError, match="is not the recorded one"):
        recorded_overlay(record, _override(NEW_HF, "flag-two"), what="regrade")


def test_a_record_from_before_the_split_still_applies():
    overlay = _override(OLD_HF, "flag-one")
    older = {"keys": ["sandbox"], "sha256": overlay_hash(overlay), "patch": overlay}

    assert recorded_overlay(older, what="regrade") == overlay
    # An older evaluation.json kept the overlay itself.
    assert recorded_overlay(overlay, what="a resume") == overlay
    assert recorded_overlay(None, what="regrade") is None


def test_a_resume_needs_the_withheld_values(tmp_path, monkeypatch):
    overlay = _override(OLD_HF, "flag-one")
    recorded = _recorded_job(tmp_path, overlay)

    with pytest.raises(ConfigOverrideNeededError, match=r"sandbox\.env\.HF_TOKEN"):
        _config_from_record(recorded, None)

    rotated = _override(NEW_HF, "flag-one")
    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(rotated))
    assert _config_from_record(recorded, None).config_override == rotated


def test_a_resume_of_a_job_with_no_secret_in_its_overlay_needs_nothing(tmp_path):
    recorded = _recorded_job(tmp_path, SAFE)

    assert _config_from_record(recorded, None).config_override == SAFE


def test_continue_needs_the_withheld_values(tmp_path, monkeypatch):
    overlay = _override(OLD_HF, "flag-one")
    run = RunFolder(
        tmp_path, {"config_override": overlay_record(overlay)}, {}, ["go"], []
    )

    def continued():
        return build_rollout_config(
            run,
            task_path=tmp_path,
            live_model="test",
            agent_env={},
            timeout=30,
            output_dir=tmp_path,
            rollout_name="continued",
        )

    with pytest.raises(
        ConfigOverrideNeededError, match="continue needs the config override values"
    ):
        continued()

    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(overlay))
    assert continued().config_override == overlay


# --- review of PR #145, 2026-11-13 -------------------------------------------


def test_a_resume_with_another_token_budget_is_refused(tmp_path):
    """Review of PR #145, 2026-11-13: ``agent.token_budget`` splits into the
    words token and budget, so it was withheld and left out of what a resume
    compares, and a resume at budget 5 of a job run at 100000 was accepted.
    A field the task config declares is stored and compared."""
    old = _first_job(
        tmp_path, job_name="named", config_override={"agent": {"token_budget": 100000}}
    )
    record = json.loads((old / "evaluation.json").read_text())["config"]
    stored = record["config_override"]
    assert (stored["patch"], stored["withheld"]) == (
        {"agent": {"token_budget": 100000}},
        [],
    )
    same = _config(config_override={"agent": {"token_budget": 100000}})
    assert _check_resume_mismatch(old, same) is None

    refusal = _refusal(old, _config(config_override={"agent": {"token_budget": 5}}))

    assert "config_override agent.token_budget changed" in refusal


def _fields(model) -> dict[str, int]:
    """Every field of *model* that is not a mapping, set to a number."""
    return {
        name: 7
        for name, field in model.model_fields.items()
        if "dict[" not in str(field.annotation)
    }


def test_every_field_the_task_config_declares_is_stored():
    """Review of PR #145, 2026-11-13: the fields of the models behind the
    overlay's sections are the schema's names, not a credential's, whatever
    words they hold (token_budget; a setup command's capture_dir_b64_env).
    The rule reads them from the models, so a field added later is covered."""
    overlay = {
        "agent": _fields(AgentConfig),
        "sandbox": {
            **_fields(SandboxConfig),
            "healthcheck": _fields(HealthcheckConfig),
            "tpu": _fields(TpuSpec),
            "mcp_servers": [_fields(MCPServerConfig)],
            "setup_commands": [_fields(SetupCommandConfig)],
        },
        "artifacts": [_fields(ArtifactConfig)],
    }

    record = overlay_record(overlay)

    assert record["withheld"] == []
    assert record["patch"] == overlay


def _in_lists(trace: str) -> dict:
    """An MCP server with headers and a setup command with an env, each in
    the list the task config declares for it."""
    return {
        "sandbox": {
            "mcp_servers": [{"name": "gh", "headers": {"X-Trace": trace}}],
            "setup_commands": [{"command": "make", "env": {"MY_FLAG": "flag-one"}}],
        }
    }


def test_an_env_or_headers_mapping_in_a_list_is_not_stored():
    """Review of PR #145, 2026-11-13: an MCP server's headers and a setup
    command's env are declared mappings, so what they hold is not stored,
    though the server or command sits in a list: the list is withheld whole,
    and a change in it is still a change."""
    record = overlay_record(_in_lists("abc123"))

    assert "abc123" not in json.dumps(record)
    assert "flag-one" not in json.dumps(record)
    assert record["withheld"] == ["sandbox.mcp_servers", "sandbox.setup_commands"]
    assert recorded_overlay(record, _in_lists("abc123"), what="regrade")
    with pytest.raises(ConfigOverrideNeededError, match="is not the recorded one"):
        recorded_overlay(record, _in_lists("def456"), what="regrade")


@pytest.mark.parametrize(
    ("value", "changed"), [(100000, 5), (True, False), ([1, 2], [1, 3])]
)
def test_a_value_under_a_count_name_is_stored_and_compared(value, changed):
    """Review of PR #145, 2026-11-13: under a secret name a number, a switch
    or a list of numbers was not stored and not compared, so a changed
    budget resumed. The lead's ruling of 2026-11-13 replaced the digest the
    second round compared it by: max_tokens names a count (its other word
    is max), not a credential, so its value is stored and compared as it
    is, and a value under a credential's name is withheld, a number too
    (test_a_number_under_a_credential_key_is_withheld_with_no_digest)."""
    overlay = {"metadata": {"max_tokens": value}}
    record = overlay_record(overlay)

    assert record["patch"] == overlay
    assert record["withheld"] == []
    assert recorded_overlay(record, overlay, what="regrade") == overlay
    changes = r"\(metadata\.max_tokens changed\)"
    with pytest.raises(ConfigOverrideNeededError, match=changes):
        recorded_overlay(record, {"metadata": {"max_tokens": changed}}, what="regrade")


@pytest.mark.parametrize(
    "name",
    [
        "authtoken",
        "accesstoken",
        "refreshtoken",
        "apitoken",
        "privatekey",
        "secretkey",
        "clientsecret",
        "xapikey",
        "GHTOKEN",
        "session",
        "sessionid",
        "JSESSIONID",
        "sid",
        "connect.sid",
    ],
)
def test_a_credential_name_written_as_one_word_is_not_stored(name):
    """Review of PR #145, 2026-11-13: a credential's name written as one
    word (authtoken, GHTOKEN) and a session cookie's name (JSESSIONID,
    connect.sid) were stored."""
    record = overlay_record({"agent": {name: "tuesday", "timeout_sec": 99}})

    assert "tuesday" not in json.dumps(record)
    assert record["patch"] == SAFE
    assert record["withheld"] == [f"agent.{name}"]


@pytest.mark.parametrize("name", ["sidebar", "inside", "obsession", "secretary"])
def test_a_name_that_only_contains_a_session_or_compound_word_is_stored(name):
    """Review of PR #145, 2026-11-13: sid and session match as whole words,
    and a run-together name only when it ends in token, secret or password."""
    overlay = {"agent": {name: "tuesday", "timeout_sec": 99}}

    assert overlay_record(overlay)["patch"] == overlay


def test_a_bare_bearer_value_is_not_stored_under_any_name():
    """Review of PR #145, 2026-11-13: "Bearer <token>" under a harmless name
    had no secret's shape (an Authorization header holding it has) and was
    stored. It is compared as a digest, as other secret-shaped values outside
    an env are; a sentence that mentions Bearer is stored."""
    overlay = {"metadata": {"note": "Bearer q7W2e9R4t6Y1", "say": "Use a Bearer."}}
    record = overlay_record(overlay)

    assert "q7W2e9R4t6Y1" not in json.dumps(record)
    assert record["withheld"] == ["metadata.note"]
    assert record["patch"] == {"metadata": {"say": "Use a Bearer."}}


def _server(*args: str) -> dict:
    return {
        "sandbox": {"mcp_servers": [{"name": "gh", "command": "npx", "args": [*args]}]}
    }


# Made up, shaped like a key, so the redactor digests the whole item.
OLD_SK = "sk-" + "a1B2c3D4" * 4
NEW_SK = "sk-" + "z9Y8x7W6" * 4


@pytest.mark.parametrize(
    ("flag", "old", "new"),
    [
        ("sid", "wednesday", "friday"),
        ("api-key", OLD_SK, NEW_SK),
        ("token", OLD_SK, NEW_SK),
    ],
)
def test_the_value_after_a_secret_flag_in_a_list_is_not_stored(flag, old, new):
    """Review of PR #145, 2026-11-13: MCP args ["--api-key", "<v>"] held the
    key as a plain list item, and the list was stored. The value of a
    --<name> <v> or --<name>=<v> flag whose name is a credential's is left
    out of what a resume compares and the list is withheld whole: a rotated
    key resumes, another change to the list is a change.

    Third review of PR #145, 2026-11-13: --api-key=<v> with a value shaped
    like a key was digested whole before the flag was seen, so a rotated
    key was a change. The api-key and token cases guard that; sid's value
    is harmless-shaped and never met the redactor."""
    overlay = _server(
        "-y", "srv", "--api-key", "tuesday", f"--{flag}={old}", "--lang", "en"
    )
    record = overlay_record(overlay)

    assert "tuesday" not in json.dumps(record)
    assert old not in json.dumps(record)
    assert record["withheld"] == ["sandbox.mcp_servers"]
    rotated = _server(
        "-y", "srv", "--api-key", "thursday", f"--{flag}={new}", "--lang", "en"
    )
    assert recorded_overlay(record, rotated, what="regrade") == rotated
    other = _server(
        "-y", "srv", "--api-key", "tuesday", f"--{flag}={old}", "--lang", "fr"
    )
    with pytest.raises(ConfigOverrideNeededError, match="is not the recorded one"):
        recorded_overlay(record, other, what="regrade")
    plain = _server("-y", "srv", "--keyboard", "us")
    assert overlay_record(plain)["patch"] == plain


# --- third review of PR #145, 2026-11-13 --------------------------------------


def _job_env(token: str | None = OLD_HF, **more: str) -> dict:
    """A sandbox env with a credential (GH_TOKEN) and a setting (MY_FLAG)."""
    env = {"MY_FLAG": "flag-one", **more}
    if token is not None:
        env["GH_TOKEN"] = token
    return {"sandbox": {"env": env}}


# The overlay a resume is given in place of _job_env(), and what the
# refusal names.
_DROPPED_OR_ADDED = [
    (_job_env(token=None), "sandbox.env.GH_TOKEN removed"),
    (_job_env(NEW_TOKEN=NEW_HF), "sandbox.env.NEW_TOKEN added"),
]
_DROPPED_OR_ADDED_IDS = ["dropped", "added"]


@pytest.mark.parametrize(
    ("supplied", "said"), _DROPPED_OR_ADDED, ids=_DROPPED_OR_ADDED_IDS
)
def test_a_supplied_overlay_must_withhold_the_same_keys(supplied, said):
    """Third review of PR #145, 2026-11-13: only ``compare`` was checked,
    and a secret-looking entry has no place in it, so a regrade, recovery or
    continue given sandbox.env without GH_TOKEN was accepted and ran
    without it; one given a new secret-looking key too. The withheld paths
    must be the job's, named by path; a rotated key still is accepted."""
    record = overlay_record(_job_env())

    with pytest.raises(ConfigOverrideNeededError) as refused:
        recorded_overlay(record, supplied, what="regrade")

    assert said in str(refused.value)
    for value in (OLD_HF, NEW_HF, "flag-one"):
        assert value not in str(refused.value) + refused.value.hint
    rotated = _job_env(token=NEW_HF)
    assert recorded_overlay(record, rotated, what="regrade") == rotated


@pytest.mark.parametrize(
    ("supplied", "said"), _DROPPED_OR_ADDED, ids=_DROPPED_OR_ADDED_IDS
)
def test_a_resume_that_drops_or_adds_a_withheld_key_is_refused(
    tmp_path, monkeypatch, supplied, said
):
    """Third review of PR #145, 2026-11-13: a resume with GH_TOKEN dropped
    from the job's sandbox.env, or a new secret-looking key added, matched
    the job, through _check_resume_mismatch (Evaluation.resume with
    config_override=..., a named rerun) and through BENCHFLOW_CONFIG_OVERRIDE
    (Evaluation.resume without one). Both refuse it now, naming the path."""
    old = _first_job(tmp_path, job_name="named", config_override=_job_env())

    refusal = _refusal(old, _config(config_override=supplied))
    assert f"ran config_override {said}" in refusal
    resumed = _refused(_resumed(old, config_override=supplied))
    assert resumed.changed == [f"config_override {said}"]
    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(supplied))
    with pytest.raises(ConfigOverrideNeededError, match=said):
        Evaluation.resume(old)
    for value in (OLD_HF, NEW_HF, "flag-one"):
        assert value not in refusal + str(resumed)

    rotated = _job_env(token=NEW_HF)
    assert _check_resume_mismatch(old, _config(config_override=rotated)) is None
    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(rotated))
    assert Evaluation.resume(old)._config.config_override == rotated


def test_the_command_line_refuses_a_dropped_or_added_withheld_key(
    tmp_path, monkeypatch
):
    """Third review of PR #145, 2026-11-13: `bench eval resume` takes the
    overlay from BENCHFLOW_CONFIG_OVERRIDE, and a rerun of `bench run --job`
    from --config-override; both accepted one without GH_TOKEN or with a new
    secret-looking key. They exit 2 now, naming the path, and run nothing; a
    rotated key resumes."""
    ran = _fake_rollouts(monkeypatch)
    tasks, jobs = _tasks(tmp_path), tmp_path / "jobs"
    line = (
        "run",
        str(tasks),
        "--agent",
        "oracle",
        "--jobs-dir",
        str(jobs),
        "--job",
        "j1",
    )
    first = _bench(*line, "--config-override", json.dumps(_job_env()))
    assert first.exit_code == 0, first.output
    job = jobs / "j1"

    for supplied, said in _DROPPED_OR_ADDED:
        monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(supplied))
        resumed = _bench("eval", "resume", str(job))
        assert resumed.exit_code == 2, resumed.output
        assert f"is not the recorded one ({said})" in _squash(resumed.output)
        monkeypatch.delenv(CONFIG_OVERRIDE_ENV)
        rerun = _bench(*line, "--config-override", json.dumps(supplied))
        assert rerun.exit_code == 2, rerun.output
        assert f"in it ran config_override {said}" in _squash(rerun.output)
        for value in (OLD_HF, NEW_HF, "flag-one"):
            assert value not in resumed.output + rerun.output
    assert len(ran) == 2

    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(_job_env(token=NEW_HF)))
    rotated = _bench("eval", "resume", str(job))
    assert rotated.exit_code == 0, rotated.output
    assert len(ran) == 2


def _server_env(env: dict) -> dict:
    return {
        "sandbox": {
            "mcp_servers": [
                {"name": "gh", "transport": "stdio", "command": "npx", "env": env}
            ]
        }
    }


@pytest.mark.parametrize(
    "env",
    [
        {"MY_FLAG": "flag-one"},
        {"GH_TOKEN": OLD_HF, "MY_FLAG": "flag-one", "NEW_TOKEN": OLD_HF},
    ],
    ids=["dropped", "added"],
)
def test_a_key_dropped_or_added_inside_a_withheld_list_is_a_change(tmp_path, env):
    """Third review of PR #145, 2026-11-13: an MCP server's env sits in a
    list, which is withheld whole, so its withheld path is the same when a
    secret-looking key inside it is dropped or added, and ``compare`` held
    no trace of that key. ``compare`` now names every key, with
    ``<withheld>`` where it does not compare the value, so it is a change
    there; a rotated key is not."""
    job_overlay = _server_env({"GH_TOKEN": OLD_HF, "MY_FLAG": "flag-one"})
    record = overlay_record(job_overlay)
    assert record["withheld"] == ["sandbox.mcp_servers"]
    assert OLD_HF not in json.dumps(record)

    with pytest.raises(
        ConfigOverrideNeededError, match=r"\(sandbox\.mcp_servers changed\)"
    ):
        recorded_overlay(record, _server_env(env), what="regrade")
    old = _first_job(tmp_path, job_name="named", config_override=job_overlay)
    refusal = _refusal(old, _config(config_override=_server_env(env)))
    assert "ran config_override sandbox.mcp_servers changed" in refusal

    rotated = _server_env({"GH_TOKEN": NEW_HF, "MY_FLAG": "flag-one"})
    assert recorded_overlay(record, rotated, what="regrade") == rotated
    assert _check_resume_mismatch(old, _config(config_override=rotated)) is None


def _stdio(*args: str) -> dict:
    """An MCP server a run can load (stdio, with a command) with *args*."""
    server = {"name": "gh", "transport": "stdio", "command": "npx", "args": [*args]}
    return {"sandbox": {"mcp_servers": [server]}}


def test_a_count_flag_in_a_list_is_stored_and_compared(tmp_path):
    """Third review of PR #145, 2026-11-13: --max-tokens 4096 in MCP args
    resumed at 8192. The lead's ruling of 2026-11-13: --max-tokens names a
    count, not a credential (its other word is max), so the list is stored
    in ``patch`` as given and 8192 is a change, named by the list's path.
    (The third round's digest of a number after a secret flag is gone.)"""
    overlay = _stdio("--lang", "en", "--max-tokens", "4096")
    record = overlay_record(overlay)

    assert (record["patch"], record["withheld"]) == (overlay, [])
    more = _stdio("--lang", "en", "--max-tokens", "8192")
    with pytest.raises(
        ConfigOverrideNeededError, match=r"\(sandbox\.mcp_servers changed\)"
    ):
        recorded_overlay(record, more, what="regrade")
    old = _first_job(tmp_path, job_name="named", config_override=overlay)
    refusal = _refusal(old, _config(config_override=more))
    assert "ran config_override sandbox.mcp_servers changed" in refusal
    assert _check_resume_mismatch(old, _config(config_override=overlay)) is None


def _no_trace_of(value: str, record: dict) -> None:
    """Neither *value* nor a digest of it (as a string or a number, the
    forms a record could have digested) is anywhere in *record*."""
    from benchflow.evaluation import _redacted

    text = json.dumps(record)
    assert value not in text
    for form in (value, json.dumps(value), json.dumps(int(value))):
        assert _redacted(form) not in text


def test_a_number_after_a_credential_flag_in_a_list_is_withheld_with_no_digest(
    tmp_path,
):
    """The lead's ruling of 2026-11-13: a short digest of --password 123456
    could be guessed. The value after a credential's flag, numeric or not,
    is withheld with no digest anywhere in the record, and supplied again:
    another value on the flag resumes, dropping the flag does not."""
    overlay = _stdio("--lang", "en", "--password", "123456")
    record = overlay_record(overlay)

    args = record["compare"]["sandbox"]["mcp_servers"][0]["args"]
    assert args == ["--lang", "en", "--password", "<withheld>"]
    assert record["withheld"] == ["sandbox.mcp_servers"]
    _no_trace_of("123456", record)
    with pytest.raises(ConfigOverrideNeededError, match=r"sandbox\.mcp_servers"):
        recorded_overlay(record, what="regrade")
    other = _stdio("--lang", "en", "--password", "654321")
    assert recorded_overlay(record, other, what="regrade") == other
    dropped = _stdio("--lang", "en")
    with pytest.raises(
        ConfigOverrideNeededError, match=r"\(sandbox\.mcp_servers changed\)"
    ):
        recorded_overlay(record, dropped, what="regrade")
    old = _first_job(tmp_path, job_name="named", config_override=overlay)
    assert _check_resume_mismatch(old, _config(config_override=other)) is None
    refusal = _refusal(old, _config(config_override=dropped))
    assert "ran config_override sandbox.mcp_servers changed" in refusal


def test_a_number_under_a_credential_key_is_withheld_with_no_digest(tmp_path):
    """The lead's ruling of 2026-11-13, for a key: metadata.password 123456
    was compared by a short digest, which could be guessed. It is withheld,
    with no digest in the record, and supplied again: another value
    resumes, dropping the key is refused, by its path."""
    overlay = {"metadata": {"password": 123456, "note": "plain"}}
    record = overlay_record(overlay)

    assert record["patch"] == {"metadata": {"note": "plain"}}
    assert record["withheld"] == ["metadata.password"]
    assert record["compare"]["metadata"]["password"] == "<withheld>"
    _no_trace_of("123456", record)
    with pytest.raises(ConfigOverrideNeededError, match=r"metadata\.password"):
        recorded_overlay(record, what="regrade")
    other = {"metadata": {"password": 654321, "note": "plain"}}
    assert recorded_overlay(record, other, what="regrade") == other
    dropped = {"metadata": {"note": "plain"}}
    with pytest.raises(
        ConfigOverrideNeededError, match=r"\(metadata\.password removed\)"
    ):
        recorded_overlay(record, dropped, what="regrade")
    old = _first_job(tmp_path, job_name="named", config_override=overlay)
    assert _check_resume_mismatch(old, _config(config_override=other)) is None
    refusal = _refusal(old, _config(config_override=dropped))
    assert "ran config_override metadata.password removed" in refusal


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


@pytest.mark.parametrize("name", SETTINGS)
def test_a_credential_word_beside_a_companion_word_is_a_setting(name):
    """The lead's rulings of 2026-11-13 (fifth and sixth rounds): a setting
    that is not compared lets a resume mix runs. A name the shared
    classifier calls a setting (tests/test_credential_names.py: tokens
    beside a companion word, token or session beside a limit, count or time
    word, a tokenizer's token, a token id, credentials beside a count word),
    as a key and as a flag in a list: its value is in ``patch`` and in
    ``compare`` as given, nothing is withheld, and another value is
    refused, named by its path."""
    key = {"metadata": {name: 4096}}
    record = overlay_record(key)
    assert (record["patch"], record["compare"], record["withheld"]) == (key, key, [])
    with pytest.raises(
        ConfigOverrideNeededError, match=re.escape(f"(metadata.{name} changed)")
    ):
        recorded_overlay(record, {"metadata": {name: 8192}}, what="regrade")

    flag = _stdio(_flag(name), "4096")
    record = overlay_record(flag)
    assert (record["patch"], record["compare"], record["withheld"]) == (flag, flag, [])
    with pytest.raises(
        ConfigOverrideNeededError, match=re.escape("(sandbox.mcp_servers changed)")
    ):
        recorded_overlay(record, _stdio(_flag(name), "8192"), what="regrade")


# Run together the other way round (a credential's word, then a companion):
# one word with no credential word in it, so stored and compared as any name.
# The lead's review of 2c568af4c listed them as settings; the shared rule
# gives None (tests/test_credential_names.py does not pin them).
_GLUED_AFTER = ["tokenlimit", "tokenbudget", "sessiontimeout", "sessionttl"]


@pytest.mark.parametrize("name", _GLUED_AFTER)
def test_a_companion_glued_after_a_credential_word_is_stored_both_ways(name):
    """Review of PR #145 2c568af4c, 2026-11-13: these names lost their
    tests when the name lists moved to the shared table. As a key and as a
    flag in a list, the value is in ``patch`` and ``compare`` as given and
    another value is refused, named by its path."""
    from benchflow._utils.credential_names import name_kind

    assert name_kind(name) is None
    key = {"metadata": {name: 4096}}
    record = overlay_record(key)
    assert (record["patch"], record["compare"], record["withheld"]) == (key, key, [])
    with pytest.raises(
        ConfigOverrideNeededError, match=re.escape(f"(metadata.{name} changed)")
    ):
        recorded_overlay(record, {"metadata": {name: 8192}}, what="regrade")
    flag = _stdio(_flag(name), "4096")
    record = overlay_record(flag)
    assert (record["patch"], record["compare"], record["withheld"]) == (flag, flag, [])
    with pytest.raises(
        ConfigOverrideNeededError, match=re.escape("(sandbox.mcp_servers changed)")
    ):
        recorded_overlay(record, _stdio(_flag(name), "8192"), what="regrade")


@pytest.mark.parametrize("name", CREDENTIALS)
def test_a_credential_name_is_withheld_both_ways(name):
    """The lead's rulings of 2026-11-13 (fifth and sixth rounds): a name the
    shared classifier calls a credential (password, key, auth, credential,
    cookie and bearer always; token or session with no limit, count or time
    word; MS_GRAPH_TOKEN, max_api_key, session_id, ...), as a key and as a
    flag in a list: its value, a number too, is withheld, is not in
    ``compare``, has no digest in the record, must be supplied again, and
    another value resumes."""
    key = {"metadata": {name: 123456, "note": "plain"}}
    record = overlay_record(key)
    assert record["withheld"] == [f"metadata.{name}"]
    assert record["compare"]["metadata"][name] == "<withheld>"
    _no_trace_of("123456", record)
    with pytest.raises(ConfigOverrideNeededError, match=re.escape(f"metadata.{name}")):
        recorded_overlay(record, what="regrade")
    other = {"metadata": {name: 654321, "note": "plain"}}
    assert recorded_overlay(record, other, what="regrade") == other

    flag = _stdio(_flag(name), "123456")
    record = overlay_record(flag)
    assert record["withheld"] == ["sandbox.mcp_servers"]
    args = record["compare"]["sandbox"]["mcp_servers"][0]["args"]
    assert args == [_flag(name), "<withheld>"]
    _no_trace_of("123456", record)
    with pytest.raises(ConfigOverrideNeededError, match=r"sandbox\.mcp_servers"):
        recorded_overlay(record, what="regrade")
    rotated = _stdio(_flag(name), "654321")
    assert recorded_overlay(record, rotated, what="regrade") == rotated


def test_a_resume_with_another_max_output_tokens_is_refused(tmp_path):
    """The lead's ruling of 2026-11-13 (fifth round): max_output_tokens was
    withheld and not compared, so a resume at 8192 of a job run at 4096
    was accepted. Through the same resume path as the token_budget case
    (test_a_resume_with_another_token_budget_is_refused)."""
    overlay = {"metadata": {"max_output_tokens": 4096}}
    old = _first_job(tmp_path, job_name="named", config_override=overlay)
    stored = json.loads((old / "evaluation.json").read_text())["config"]
    assert (
        stored["config_override"]["patch"],
        stored["config_override"]["withheld"],
    ) == (
        overlay,
        [],
    )
    assert _check_resume_mismatch(old, _config(config_override=overlay)) is None

    more = {"metadata": {"max_output_tokens": 8192}}
    refusal = _refusal(old, _config(config_override=more))

    assert "config_override metadata.max_output_tokens changed" in refusal


def test_a_setting_name_still_withholds_a_value_shaped_like_a_key():
    """The lead's ruling of 2026-11-13 (fifth round): the shape rule comes
    before the name rule, so a key-shaped value is withheld under a
    setting's name too."""
    record = overlay_record({"metadata": {"max_output_tokens": SK_ANT}})

    assert record["withheld"] == ["metadata.max_output_tokens"]
    assert SK_ANT not in json.dumps(record)


def test_a_secret_entry_after_a_free_form_flag_in_a_list_has_no_digest():
    """Fourth round of PR #145, 2026-11-13 (the same objection as the
    lead's ruling): the item after --env or --header was compared by a
    digest, DB_PASSWORD=<v> and Authorization: <v> too. Such an entry is
    left out as an env or headers mapping's would be; a setting there is
    still compared by a digest."""
    overlay = _stdio("--env", "DB_PASSWORD=123456", "--header", "X-Trace: abc")
    record = overlay_record(overlay)

    args = record["compare"]["sandbox"]["mcp_servers"][0]["args"]
    assert args[:3] == ["--env", "<withheld>", "--header"]
    assert args[3].startswith("<redacted sha256:")
    _no_trace_of("123456", record)
    for given in ("DB_PASSWORD=123456", "X-Trace: abc"):
        assert given not in json.dumps(record)


def test_a_secret_flag_taken_as_a_switch_still_hides_the_next_value():
    """Third review of PR #145, 2026-11-13 (found while fixing the number
    case): --no-auth took the next item, --api-key, as its value, so the key
    after --api-key was compared, and stored in ``compare``, as plain text.
    A value that is itself a secret-named flag makes the item after it a
    value too."""
    record = overlay_record(_server("--no-auth", "--api-key", "tuesday"))

    assert "tuesday" not in json.dumps(record)
    rotated = _server("--no-auth", "--api-key", "thursday")
    assert recorded_overlay(record, rotated, what="regrade") == rotated


def test_the_record_hash_holds_no_withheld_value():
    """Third review of PR #145, 2026-11-13: ``sha256`` hashed the whole
    overlay, withheld values included, so a guessed weak password could be
    checked against config.json. It hashes what the record stores (patch,
    withheld and compare): the same for two passwords, and still another
    hash for a changed setting."""
    guessed = overlay_record({"agent": {"password": "hunter2"}})
    real = overlay_record({"agent": {"password": "letmein1"}})

    assert guessed["sha256"] == real["sha256"]
    assert guessed["sha256"] != overlay_hash({"agent": {"password": "hunter2"}})
    flag_two = overlay_record(_job_env(MY_FLAG="flag-two"))
    assert flag_two["sha256"] != overlay_record(_job_env())["sha256"]


def test_an_older_record_whose_hash_is_the_whole_overlays_still_resumes(tmp_path):
    """Third review of PR #145, 2026-11-13: a record written before
    ``withheld`` and ``compare`` existed has the whole overlay's hash and
    keeps the overlay in ``patch``. Nothing compares that hash with a new
    record's, so the same overlay, or a rotated key, still resumes it, and a
    dropped withheld key is still named."""
    old = _first_job(tmp_path, job_name="named", config_override=_job_env())
    path = old / "evaluation.json"
    stored = json.loads(path.read_text())
    older = {
        "keys": ["sandbox"],
        "sha256": overlay_hash(_job_env()),
        "patch": _job_env(),
    }
    stored["config"]["config_override"] = older
    path.write_text(json.dumps(stored))

    assert _check_resume_mismatch(old, _config(config_override=_job_env())) is None
    rotated = _config(config_override=_job_env(token=NEW_HF))
    assert _check_resume_mismatch(old, rotated) is None
    assert recorded_overlay(older, _job_env(token=NEW_HF), what="regrade")
    refusal = _refusal(old, _config(config_override=_job_env(token=None)))
    assert "ran config_override sandbox.env.GH_TOKEN removed" in refusal


# --- fourth round of PR #145, 2026-11-13: the override in messages, logs and
# saved configs -----------------------------------------------------------------

# Made up: an sk-ant-shaped key, and a harmless-shaped password that pydantic
# would quote whole (it cuts a long value but keeps its start).
SK_ANT = "sk-ant-api03-" + "Q7w2E9r4T6y1U3i5" * 4
PASSWORD = "hunter2-made-up"


def _nowhere(*texts: str) -> None:
    """Neither made-up value, nor the start of the key, nor pydantic's
    quoting of an input, is in any of *texts*."""
    joined = "\n".join(texts)
    for shown in (SK_ANT, SK_ANT[:11], PASSWORD, "input_value"):
        assert shown not in joined


def test_an_override_that_does_not_parse_names_where_not_what(tmp_path):
    """Fourth round of PR #145, 2026-11-13 (reviewer, item 2): the parse
    error printed `begins <the first 80 characters>`, an @file's too, so a
    key in a malformed override was printed raw. It says where the parser
    stopped and, for a file, which file, never what the text was."""
    broken = '{"sandbox": {"env": {"GH_TOKEN": "' + SK_ANT + '", "PW": "' + PASSWORD
    with pytest.raises(ValueError) as inline:
        load_config_override(broken)
    assert "inline --config-override" in str(inline.value)
    assert "line 1" in str(inline.value)
    (tmp_path / "o.json").write_text(broken)
    with pytest.raises(ValueError) as from_file:
        load_config_override(f"@{tmp_path / 'o.json'}")
    assert str(tmp_path / "o.json") in str(from_file.value)
    _nowhere(str(inline.value), str(from_file.value))

    tasks = _tasks(tmp_path)
    for given in (broken, f"@{tmp_path / 'o.json'}"):
        line = ("run", str(tasks), "--jobs-dir", str(tmp_path / "jobs"))
        line += ("--agent", "oracle", "--config-override", given)
        result = _bench(*line, "--dry-run")
        assert result.exit_code == 2, result.output
        _nowhere(result.stdout, result.stderr)


def test_the_timeout_multipliers_check_names_keys_never_values():
    """Fourth round of PR #145, 2026-11-13 (reviewer, item 2: every message
    that formats the override): a refused timeout_multipliers printed the
    mapping, and a refused multiplier its value, with repr."""
    for scales in ({"api_key": SK_ANT}, {"verifier": PASSWORD}):
        with pytest.raises(ValueError) as refused:
            validate_overlay({"timeout_multipliers": scales})
        _nowhere(str(refused.value))


def _rollout_task(root) -> object:
    """A task a run lists (as tests/test_cli_verbs.py's): task.md, an image,
    an oracle and a verifier."""
    task = root / "hello"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho hello\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n"
    )
    (task / "task.md").write_text("---\nname: hello\n---\nSay hello.\n")
    return task


@pytest.mark.parametrize(
    ("overlay", "said"),
    [
        (
            {"sandbox": {"evn": {"GH_TOKEN": SK_ANT, "PW": PASSWORD}}},
            "sandbox.evn: is not a field there",
        ),
        (
            {"sandbox": {"env": {"GH_TOKEN": [SK_ANT, PASSWORD]}}},
            "sandbox.env.GH_TOKEN: takes a string",
        ),
        (
            {"agent": {"timeout_sec": {"password": PASSWORD, "key": SK_ANT}}},
            "agent.timeout_sec: takes a number",
        ),
    ],
    ids=["mistyped-key", "wrong-type", "wrong-type-holding-a-key"],
)
def test_an_override_the_task_config_refuses_names_paths_not_values(
    tmp_path, caplog, capsys, overlay, said
):
    """Fourth round of PR #145, 2026-11-13 (reviewer, item 3): pydantic's
    `input_value={'GH_TOKEN': ...}` reached the `Run failed:` line, the
    trial's error and its result. The overlay's error names each path and
    what the field takes, holds no input, and chains nothing; it is still a
    ValidationError, and a user's error, so no traceback is written."""
    import asyncio
    import logging

    from pydantic import ValidationError

    from benchflow.rollout import Rollout
    from benchflow.rollout._config import RolloutConfig
    from benchflow.task.config import TaskConfig

    task = _rollout_task(tmp_path)
    with pytest.raises(ValidationError) as refused:
        apply_config_override(TaskConfig(), overlay)
    assert isinstance(refused.value, UserError)
    assert said in str(refused.value)
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    config = RolloutConfig.from_legacy(
        task_path=task,
        agent="oracle",
        environment="docker",
        jobs_dir=tmp_path / "jobs",
        config_override=overlay,
    )
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(Rollout(config).run())

    assert said in result.error
    files = [
        path.read_text(errors="replace")
        for path in (tmp_path / "jobs").rglob("*")
        if path.is_file()
    ]
    assert not list((tmp_path / "jobs").rglob("error-traceback.txt"))
    out = capsys.readouterr()
    _nowhere(
        str(refused.value),
        repr(refused.value),
        result.error,
        caplog.text,
        out.out,
        out.err,
        *files,
    )


@pytest.mark.parametrize(
    ("overlay", "said"),
    [
        (
            {"sandbox": {"evn": {"GH_TOKEN": SK_ANT, "PW": PASSWORD}}},
            "sandbox.evn: is not a field there",
        ),
        (
            {"sandbox": {"env": {"GH_TOKEN": [SK_ANT, PASSWORD]}}},
            "sandbox.env.GH_TOKEN: takes a string",
        ),
    ],
    ids=["mistyped-key", "wrong-type"],
)
def test_a_dry_run_refuses_an_override_the_task_config_refuses(
    tmp_path, caplog, overlay, said
):
    """Fourth round of PR #145, 2026-11-13 (reviewer, item 3): the run card
    applies the override to each task and logged pydantic's text, input
    and all, at debug. Sixth-round review nit: the dry run then exited 0;
    it exits 2 with the message a real run's trials give, naming the task
    and the path, never a value."""
    import logging

    tasks = tmp_path / "tasks"
    _rollout_task(tasks)
    with caplog.at_level(logging.DEBUG):
        result = _bench(
            "run", str(tasks), "--agent", "oracle", "--jobs-dir",
            str(tmp_path / "jobs"), "--config-override", json.dumps(overlay),
            "--dry-run",
        )  # fmt: skip
    assert result.exit_code == 2, result.output
    text = _squash(result.stderr)
    assert f"hello: the config override does not fit the task's config: {said}" in text
    assert "--dry-run" in text  # the hint names a command that checks it
    _nowhere(result.stdout, result.stderr, caplog.text)


def test_a_saved_config_holds_no_withheld_value_unless_asked(tmp_path, monkeypatch):
    """Fourth round of PR #145, 2026-11-13 (reviewer, item 4):
    Evaluation.to_dict()/to_yaml() and RolloutConfig's wrote the override
    whole. As for agent_env, its values that may be a credential are left
    out unless include_config_override=True: the override's record is
    written, and loading it needs BENCHFLOW_CONFIG_OVERRIDE again or stops
    with ConfigOverrideNeededError, never runs without them."""
    from benchflow.rollout._config import RolloutConfig

    overlay = {"sandbox": {"env": {"GH_TOKEN": SK_ANT, "PW": PASSWORD}}}
    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(config_override=overlay),
    )
    rollout = RolloutConfig.from_legacy(
        task_path=_rollout_task(tmp_path),
        agent="oracle",
        jobs_dir=tmp_path / "jobs",
        config_override=overlay,
    )
    saved = job.to_yaml(tmp_path / "job.yaml")
    rollout_saved = rollout.to_yaml(tmp_path / "rollout.yaml")
    _nowhere(
        json.dumps(job.to_dict()),
        json.dumps(rollout.to_dict()),
        saved.read_text(),
        rollout_saved.read_text(),
    )
    assert job.to_dict()["config_override"]["withheld"] == [
        "sandbox.env.GH_TOKEN",
        "sandbox.env.PW",
    ]
    assert job.to_dict(include_config_override=True)["config_override"] == overlay
    assert rollout.to_dict(include_config_override=True)["config_override"] == overlay

    with pytest.raises(ConfigOverrideNeededError, match=r"sandbox\.env\.GH_TOKEN"):
        Evaluation.from_yaml(saved)
    with pytest.raises(ConfigOverrideNeededError, match=r"sandbox\.env\.GH_TOKEN"):
        RolloutConfig.from_yaml(rollout_saved)
    monkeypatch.setenv(CONFIG_OVERRIDE_ENV, json.dumps(overlay))
    assert Evaluation.from_yaml(saved)._config.config_override == overlay
    assert RolloutConfig.from_yaml(rollout_saved).config_override == overlay


def test_a_shards_worker_payload_keeps_the_overrides_record():
    """Fourth round of PR #145, 2026-11-13 (found checking every place the
    override is written): shard-NNN/worker_payload.json, kept in the job
    folder beside the worker's private copy, held the override whole."""
    from benchflow.eval_sharding import _worker_payload_artifact

    overlay = {"sandbox": {"env": {"GH_TOKEN": SK_ANT, "PW": PASSWORD}}}
    artifact = _worker_payload_artifact({"config": {"config_override": overlay}})

    _nowhere(json.dumps(artifact))
    assert artifact["config"]["config_override"]["withheld"] == [
        "sandbox.env.GH_TOKEN",
        "sandbox.env.PW",
    ]


def test_the_override_messages_read_for_a_job_and_for_a_saved_config(
    tmp_path, monkeypatch
):
    """Sixth-round review of PR #145, 2026-11-13 (a nit): the messages said
    "this job ran with" for a saved config too. They speak of the record,
    which a job and a saved config both have."""
    record = overlay_record(_job_env())
    for what in ("a resume", "loading this config"):
        with pytest.raises(ConfigOverrideNeededError) as needed:
            recorded_overlay(record, what=what)
        assert str(needed.value).startswith(
            f"{what} needs the config override values its record left out"
        )
        with pytest.raises(ConfigOverrideNeededError) as other:
            recorded_overlay(record, _job_env(token=None), what=what)
        assert str(other.value).startswith(
            f"the config override given to {what} is not the recorded one "
            "(sandbox.env.GH_TOKEN removed); the record sets "
        )
        for error in (needed.value, other.value):
            assert "job ran" not in str(error) + error.hint

    job = Evaluation(
        tasks_dir=_tasks(tmp_path),
        jobs_dir=tmp_path / "jobs",
        config=_config(config_override=_job_env()),
    )
    saved = job.to_yaml(tmp_path / "job.yaml")
    with pytest.raises(ConfigOverrideNeededError) as loading:
        Evaluation.from_yaml(saved)
    assert str(loading.value).startswith(
        "loading this config needs the config override values its record left out"
    )
