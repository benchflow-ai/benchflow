"""Secret shapes and record fields the write-time redactor used to miss.

Guards the audit fix on branch ``audit-redaction-gaps``
(findings CRED-2, CRED-7, RECORDS-7). Every value here is made up.

Before the fix: result.json kept error text raw; error-traceback.txt masked
only the host's own secret values; and the shapes below passed through
``redact_trajectory_text`` unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow.rollout import write_error_traceback
from benchflow.trajectories.types import redact_trajectory_text

MASK = "***REDACTED***"

MISSED = [
    ("Proxy-Authorization: Basic dXNlcjpGQUtFcGFzczEyMw==", "dXNlcjpGQUtFcGFzczEyMw"),
    (
        '"proxy-authorization": "Bearer FAKEtoken0123456789abcd"',
        "FAKEtoken0123456789abcd",
    ),
    ("Cookie: session=FAKEsess0123456789; theme=dark", "FAKEsess0123456789"),
    (
        "Set-Cookie: sid=FAKEsid0123456789abcdef; Path=/; HttpOnly",
        "FAKEsid0123456789abcdef",
    ),
    (
        ("token ya" + "29.a0FAK" + "EFAKEFAK" + "EFAKEFAK" + "EFAKE123" + "4 expired"),
        ("ya" + "29.a0FAK" + "EFAKEFAK" + "EFAKEFAK" + "EFAKE123" + "4"),
    ),
    (
        "key AQ.FAKEFAKEFAKEFAKEFAKEFAKE_1234 refused",
        "AQ.FAKEFAKEFAKEFAKEFAKEFAKE_1234",
    ),
    (
        ("git clone with gl" + "pat-FAKE" + "FAKEFAKE" + "FAKEFAKE" + "12"),
        ("gl" + "pat-FAKE" + "FAKEFAKE" + "FAKEFAKE" + "12"),
    ),
    (
        ("stripe sk" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
        ("sk" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
    ),
    ("DB_PASS=FAKEhunter2345", "FAKEhunter2345"),
    ("MYSQL_PWD=FAKEhunter2345", "FAKEhunter2345"),
    ('"SMTP_PASS": "FAKEhunter2345"', "FAKEhunter2345"),
    ("https://FAKEtok0123456789abcdef@gw.example.com/v1", "FAKEtok0123456789abcdef"),
    ("wss://bob:FAKEpass123456@stream.example.com/ws", "FAKEpass123456"),
    ("ssh://git:FAKEpass123456@git.example.com/repo.git", "FAKEpass123456"),
    # A second Authorization name the one-word branch's word
    # took; the rc leaked these too.
    ('Proxy-Authorization: none;(Authorization: "FAKEtook0123"', "FAKEtook0123"),
    ('Proxy-Authorization: none%20Authorization: "FAKEtook0123"', "FAKEtook0123"),
    ('Proxy-Authorization: none+Authorization="FAKEtook0123"', "FAKEtook0123"),
    ('Proxy-Authorization: none=>HTTP_AUTHORIZATION= "FAKEtook0123"', "FAKEtook0123"),
    ('Proxy-Authorization: none->Authorization%3A%20"FAKEtook0123"', "FAKEtook0123"),
    ("Proxy-Authorization: none:=Authorization: 'FAKEtook0123'", "FAKEtook0123"),
    (
        'Proxy-Authorization: noneX_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA_AUTHORIZATION : "FAKEtook0123"',
        "FAKEtook0123",
    ),
    # Round 2: a name glued to the header's word after A.2's words, a
    # shell reference, a name with more after the header's word, a code name and
    # an escaped quote; a key-shaped literal in code; R192C's B-4 and p_ma5 shapes.
    ('Authorization: please retryAuthorization: "FAKEround2a0123"', "FAKEround2a0123"),
    ("${HTTP_AUTHORIZATION}=FAKEround2a0123", "FAKEround2a0123"),
    ("${HTTP_AUTHORIZATION:-x}=FAKEround2a0123", "FAKEround2a0123"),
    ('AuthorizationHeader: "FAKEround2a0123"', "FAKEround2a0123"),
    ('Authorization: none authorization_header="FAKEround2a0123"', "FAKEround2a0123"),
    ('has_authorization: noneAuthorization=\\"FAKEround2a0123\\"', "FAKEround2a0123"),
    ('has_authorization: f("FAKEround2a0123xyz")', "FAKEround2a0123xyz"),
    ('Authorization: "none_proxyAuthorization": "FAKEround2a0123"', "FAKEround2a0123"),
    ('HTTP_AUTHORIZATION="noneAuthorization": "FAKEround2a0123"', "FAKEround2a0123"),
    ('Authorization: ""proxyAuthorization""=FAKEround2a0123', "FAKEround2a0123"),
    (
        "x-proxy-authorization: &quot;proxyAuthorization&quot;: FAKEround2a0123",
        "FAKEround2a0123",
    ),
    # Review of PR #192 (R192C, code-escq, code-unclosed): a code Authorization
    # name, then a quote left open or closed another way.
    ('has_authorization=\\"FAKEescq0123', "FAKEescq0123"),
    ('copy_source_authorization: \\"FAKEescq0123 retry', "FAKEescq0123"),
    ('has_authorization: "FAKEescq0123', "FAKEescq0123"),
    ('requires_authorization="FAKEescq0123\\"', "FAKEescq0123"),
    ("has_authorization: " + "\\" * 8 + '"FAKEescq0123', "FAKEescq0123"),
    ('has_authorization: "FAKEescq0123' + "x" * 5000, "FAKEescq0123"),
    ('has_authorization: "!@#$%^&*()_+"', "!@#$%^&*()_+"),
    # Review of PR #192 (R192C, B-2 and B-3): a quoted camel name left open; a name
    # after a joiner run, or quoted, that another carrier reads.
    ('Authorization: "proxyAuthorization=FAKEedge0123', "FAKEedge0123"),
    ('Proxy-Authorization: \\"proxyAuthorization:FAKEedge0123', "FAKEedge0123"),
    ("Authorization: _authorization: FAKEedge0123", "FAKEedge0123"),
    ('Authorization: "noneAuthorization": "FAKEedge0123"', "FAKEedge0123"),
    # Review of PR #192 (R192C, B-1): one word glued to an Authorization name
    # no carrier reads, after a header name.
    ("Authorization: proxyAuthorization=FAKEglued0123", "FAKEglued0123"),
    ("HTTP_AUTHORIZATION=proxyAuthorization=FAKEglued0123", "FAKEglued0123"),
    ("Proxy-Authorization: noneAuthorization=FAKEglued0123", "FAKEglued0123"),
    ("Authorization: retryAuthorization=FAKEglued0123 please retry", "FAKEglued0123"),
    ("Authorization: setAuthorization=FAKEglued0123", "FAKEglued0123"),
    ("Proxy-Authorization: none2authorization=FAKEglued0123", "FAKEglued0123"),
    ("authorization%3A%20xAuthorization%3DFAKEglued0123", "FAKEglued0123"),
    ("Authorization: tokenAuthorization:FAKEglued0123 retry", "FAKEglued0123"),
    (
        json.dumps({"output": "Proxy-Authorization: noneAuthorization=FAKEglued0123"}),
        "FAKEglued0123",
    ),
    # Review of PR #192 (R192B, M-A): the second name after a punctuation mark
    # inside the first one's words, and after a prefix over 64 characters.
    (
        'Authorization: please retry;HTTP_AUTHORIZATION="FAKEsecond0123"',
        "FAKEsecond0123",
    ),
    (
        'Authorization: please retry (Proxy-Authorization: "FAKEsecond0123")',
        "FAKEsecond0123",
    ),
    ('Authorization: see [Authorization: "FAKEsecond0123"]', "FAKEsecond0123"),
    ('Authorization: none {Proxy-Authorization: "FAKEsecond0123"}', "FAKEsecond0123"),
    ('Authorization: none <Authorization: "FAKEsecond0123">', "FAKEsecond0123"),
    ('Authorization: none /Proxy-Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none |Authorization="FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none #HTTP_AUTHORIZATION="FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none $HTTP_AUTHORIZATION="FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none @Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none :Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    (
        'error: Authorization: header rejected (Proxy-Authorization: "FAKEsecond0123")',
        "FAKEsecond0123",
    ),
    (
        'HTTP_AUTHORIZATION=please retry;Proxy-Authorization="FAKEsecond0123"',
        "FAKEsecond0123",
    ),
    (
        'Authorization: none X_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA_AUTHORIZATION="FAKEsecond0123"',
        "FAKEsecond0123",
    ),
    # Review of PR #192 (R192, M2): a second Authorization-family name after
    # the first one's words, its value quoted.
    ('Authorization: none, Proxy-Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: missing Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: none has_authorization="FAKEsecond0123"', "FAKEsecond0123"),
    (
        'Authorization: please retry; HTTP_AUTHORIZATION="FAKEsecond0123"',
        "FAKEsecond0123",
    ),
    ('Proxy-Authorization: none, Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    (
        'HTTP_AUTHORIZATION=please retry Proxy-Authorization="FAKEsecond0123"',
        "FAKEsecond0123",
    ),
    ('Authorization: none x-proxy-authorization: "FAKEsecond0123"', "FAKEsecond0123"),
    ('Authorization: Proxy-Authorization: "FAKEsecond0123"', "FAKEsecond0123"),
]


@pytest.mark.parametrize("text,secret", MISSED)
def test_the_shape_is_masked(text, secret):
    masked = redact_trajectory_text(text)

    assert secret not in masked
    assert MASK in masked


KEPT = [
    "PWD=/app",
    "OLDPWD=/home/agent",
    "BYPASS=1",
    "first_pass=True, second_pass=False",
    "https://git@github.com/org/repo.git",
    "ssh://git@github.com/org/repo.git",
    "https://example.com/users/alice@example.org",
    "the cookie jar is empty",
    "authorization failed for user alice",
    ("sk" + "_test_no" + "taliveke" + "y"),
]


@pytest.mark.parametrize("text", KEPT)
def test_ordinary_text_is_left_alone(text):
    assert redact_trajectory_text(text) == text


@pytest.mark.parametrize("name", ["MYSQL_PWD", "DB_PASS", "SMTP_PASS", "pg_pass"])
def test_a_password_variable_stays_out_of_config_json(name):
    assert _should_record_env_entry(name, "FAKEhunter2") is False


@pytest.mark.parametrize(
    "name,value", [("PWD", "/app"), ("BYPASS", "1"), ("COMPASS", "n")]
)
def test_a_look_alike_variable_is_still_recorded(name, value):
    assert _should_record_env_entry(name, value) is True


ERROR = (
    'upstream said: {"refresh_token": "1//0gFAKEFAKEFAKEFAKEFAKE", '
    '"x-api-key": "sk' + "-ant-api" + "03-FAKEF" + "AKEFAKEF" + 'AKE1234"} '
    "master=sk-runtime-FAKEAbCdEfGhIjKlMnOpQrStUvWx "
    "url=https://FAKEtok0123456789abcdef@gw.example.com"
)
ERROR_SECRETS = (
    "1//0gFAKEFAKEFAKEFAKEFAKE",
    ("sk" + "-ant-api" + "03-FAKEF" + "AKEFAKEF" + "AKE1234"),
    "sk-runtime-FAKEAbCdEfGhIjKlMnOpQrStUvWx",
    "FAKEtok0123456789abcdef",
)


def test_error_traceback_txt_masks_secret_shapes(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    try:
        raise RuntimeError(ERROR)
    except RuntimeError as exc:
        path = write_error_traceback(tmp_path, exc)

    assert path is not None
    text = path.read_text()
    assert "RuntimeError" in text
    for secret in ERROR_SECRETS:
        assert secret not in text


def test_result_json_masks_error_text(tmp_path):
    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout._results import _build_rollout_result

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do nothing.\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    now = datetime.now()

    result = _build_rollout_result(
        trial,
        task_name="task",
        rollout_name="task__1",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=["x"],
        error=ERROR,
        verifier_error=f"verifier crashed: {ERROR}",
        trajectory=[],
        partial_trajectory=False,
        rewards=None,
        started_at=now,
        timing={},
        diagnostics=RolloutDiagnostics(),
    )

    saved = json.loads((trial / "result.json").read_text())
    for field in ("error", "verifier_error"):
        assert MASK in saved[field]
        for secret in ERROR_SECRETS:
            assert secret not in saved[field], field
    # The run's own result keeps the text it classified.
    assert result.error == ERROR
    assert not list(Path(trial).glob("*.tmp"))


def test_a_trial_without_a_folder_masks_its_error_text(tmp_path):
    """Review of PR #134, 2026-11-13: a task that fails before its rollout
    folder exists gets its result.json from ``_record_trial_without_folder``,
    which wrote ``rollout_result_payload``'s ``error``, ``verifier_error`` and
    ``export_error`` raw. All three are masked now."""
    import asyncio
    from unittest.mock import AsyncMock

    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from benchflow.models import RolloutResult

    agent = "claude-agent-acp"
    tasks = tmp_path / "tasks"
    (tasks / "typo").mkdir(parents=True)
    (tasks / "typo" / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent=agent, concurrency=1, retry=RetryConfig(max_retries=0)
        ),
        job_name="j",
        preflight=False,
    )
    # Refused before any rollout folder exists.
    job._run_single_task = AsyncMock(
        return_value=RolloutResult(
            task_name="typo",
            agent=agent,
            error=ERROR,
            verifier_error=f"verifier crashed: {ERROR}",
            export_error=f"export failed: {ERROR}",
        )
    )
    asyncio.run(job.run())

    folder = next(job.job_dir.glob("typo__*"))
    saved = json.loads((folder / "result.json").read_text())
    assert saved["no_rollout_folder"] is True
    for field in ("error", "verifier_error", "export_error"):
        assert MASK in saved[field], field
        for secret in ERROR_SECRETS:
            assert secret not in saved[field], field


# Review of PR #134, 2026-11-13: three rules masked harmless text. Each case
# below must come through unchanged, and the secret beside it in the same
# line must still be masked.
OVER_MASKED = [
    # A *_PASS name holding a count or a switch word is a test result.
    "TOTAL_PASS=12",
    "ALL_TESTS_PASS: true",
    '{"ALL_TESTS_PASS": false, "N_PASS": 3}',
    # A Set-Cookie header named in prose holds no cookie.
    "Set-Cookie: header too large",
    "upstream sent too big header: Set-Cookie: header too large",
    # sk_live_/rk_live_ inside a longer word is no Stripe key.
    ("mask" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
    ("work" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
]


@pytest.mark.parametrize("text", OVER_MASKED)
def test_review_over_masked_text_is_left_alone(text):
    """Review of PR #134, 2026-11-13: harmless text the new rules masked
    (``TOTAL_PASS=12``, ``ALL_TESTS_PASS: true``, ``Set-Cookie: header too
    large``, ``mask_live_...``) comes through unchanged."""
    assert redact_trajectory_text(text) == text


# The one expected over-mask class (the lead's ruling, part A.2,
# 2026-11-14): "a deliberate over-mask of prose after an Authorization name;
# one expected class". After the Authorization header's own name, or an
# upper-case env name that ends in it, a value of two words or more on the
# name's line is masked from the second word on, whatever the first word
# is; a single prose word stays (`Authorization: required`,
# `HTTP_AUTHORIZATION=required`). The last four are the escape-boundary
# test's rows with its " done" suffix (SECOND_WORD_MASKED there).
# Since, the lead's ruling of 2026-11-16 08:46Z, every word is masked, the first prose word too: a bare
# value under an Authorization name is a credential, as 0.9.3 masks it.
EXPECTED_OVER_MASKED = [
    ("HTTP_AUTHORIZATION=please sign in first", f"HTTP_AUTHORIZATION={MASK}"),
    ("Authorization: please sign in first", f"Authorization: {MASK}"),
    ("Authorization: required done", f"Authorization: {MASK}"),
    ("authorization: none. done", f"authorization: {MASK}"),
    ("Proxy-Authorization: Missing done", f"Proxy-Authorization: {MASK}"),
    ("authorization: none done", f"authorization: {MASK}"),
]
ONE_PROSE_WORD = [
    ("Authorization: required", f"Authorization: {MASK}"),
    ("HTTP_AUTHORIZATION=required", f"HTTP_AUTHORIZATION={MASK}"),
    ("Proxy-Authorization: Missing", f"Proxy-Authorization: {MASK}"),
]


@pytest.mark.parametrize("text,expected", EXPECTED_OVER_MASKED)
def test_review_the_one_expected_over_mask_class(text, expected):
    """Prose after an Authorization name is masked whole, the one over-mask
    the sweep expects (the lead's ruling of 2026-11-16 08:46Z, its A.2 ruling kept the first word); a
    second pass masks nothing more."""
    assert redact_trajectory_text(text) == expected
    assert redact_trajectory_text(expected) == expected


@pytest.mark.parametrize("text,expected", ONE_PROSE_WORD)
def test_review_one_prose_word_after_an_authorization_name_is_masked(text, expected):
    """A single prose word is the value (the lead's ruling of 2026-11-16 08:46Z, kept it)."""
    assert redact_trajectory_text(text) == expected
    assert redact_trajectory_text(expected) == expected


BESIDE = [
    ("TOTAL_PASS=12 DB_PASS=FAKEhunter2345", f"TOTAL_PASS=12 DB_PASS={MASK}"),
    (
        "ALL_TESTS_PASS: true, SMTP_PASS: true9FAKE2345",
        f"ALL_TESTS_PASS: true, SMTP_PASS: {MASK}",
    ),
    (
        '{"ALL_TESTS_PASS": true, "DB_PASS": "FAKEhunter2345"}',
        f'{{"ALL_TESTS_PASS": true, "DB_PASS": "{MASK}"}}',
    ),
    # Six digits or more may be a numeric PIN: still masked.
    ("N_PASS=3 MYSQL_PWD=48213907", f"N_PASS=3 MYSQL_PWD={MASK}"),
    (
        "Set-Cookie: sid=FAKEsid0123456789abcdef; Path=/; HttpOnly then header "
        "too large",
        f"Set-Cookie: {MASK} then header too large",
    ),
    (
        "Cookie: theme=dark; session=FAKEsess0123456789 sent twice",
        f"Cookie: {MASK} sent twice",
    ),
    (
        '"set-cookie": "sid=FAKEsid0123456789abcdef; Path=/", "status": 431',
        f'"set-cookie": "{MASK}", "status": 431',
    ),
    (
        (
            "mask"
            + "_live_FA"
            + "KEFAKEFA"
            + "KEFAKE12"
            + "34 key=sk"
            + "_live_FA"
            + "KEFAKEFA"
            + "KEFAKE12"
            + "34"
        ),
        ("mask" + f"_live_FAKEFAKEFAKEFAKE1234 key={MASK}"),
    ),
    (
        (
            "work"
            + "_live_FA"
            + "KEFAKEFA"
            + "KEFAKE12"
            + "34 'rk"
            + "_live_FA"
            + "KEFAKEFA"
            + "KEFAKE12"
            + "34'"
        ),
        ("work" + f"_live_FAKEFAKEFAKEFAKE1234 '{MASK}'"),
    ),
]


@pytest.mark.parametrize("text,expected", BESIDE)
def test_review_the_secret_beside_harmless_text_is_still_masked(text, expected):
    """Review of PR #134, 2026-11-13: narrowing the ``*_PASS``, cookie and
    ``[sr]k_live_`` rules does not reopen the leak: the real secret shape in
    the same line is masked, the harmless text beside it kept."""
    assert redact_trajectory_text(text) == expected


# Review of PR #134, 2026-11-13, follow-ups: three rules still masked
# harmless text. Each case below comes through unchanged.
KEPT_AFTER_REVIEW = [
    # `AQ.` after a letter or a digit ends a word: no Google AI Studio key.
    "see docs/FAQ.troubleshooting-network-errors for help",
    "Read the FAQ.ProxySetupAndCertificates_v2 page",
    "IRAQ.population_statistics_by_region",
    # A count after a *_PASS name that ends a sentence or a bracket, or is
    # one of two.
    "TOTAL_PASS=12.",
    "(TOTAL_PASS=12)",
    "TESTS_PASS=12/15",
    "TOTAL_PASS=12",
    "ALL_TESTS_PASS: true",
    "[N_PASS=3]; ALL_TESTS_PASS: true.",
    '{"TOTAL_PASS": "12.", "TESTS_PASS": "12/15"}',
    # A user name without a password in a URL.
    "ssh://deploy-automation@host",
    "git+ssh://git@github.com/org/repo",
    "https://user@host/path",
    "ssh://build-service-account@ci.example.com/repo.git",
]


@pytest.mark.parametrize("text", KEPT_AFTER_REVIEW)
def test_review_follow_up_harmless_text_is_left_alone(text):
    """Review of PR #134, 2026-11-13, follow-ups: ``FAQ.`` followed by a long
    word became ``F***``; ``TOTAL_PASS=12.``, ``(TOTAL_PASS=12)`` and
    ``TESTS_PASS=12/15`` were masked; so was ``ssh://deploy-automation@host``.
    Each comes through unchanged."""
    assert redact_trajectory_text(text) == text


AQ_KEY = "AQ.FAKEFAKEFAKEFAKEFAKEFAKE_1234"
FAQ = "FAQ.troubleshooting-network-errors"
STILL_MASKED_AFTER_REVIEW = [
    # The AQ. key at a word start, after `=`, `:`, a quote or whitespace.
    (f"{AQ_KEY} refused, see {FAQ}", f"{MASK} refused, see {FAQ}"),
    (f"{FAQ} key={AQ_KEY}", f"{FAQ} key={MASK}"),
    (f"{FAQ} key:{AQ_KEY}", f"{FAQ} key:{MASK}"),
    (f'{FAQ} "{AQ_KEY}"', f'{FAQ} "{MASK}"'),
    (f"{FAQ} '{AQ_KEY}'", f"{FAQ} '{MASK}'"),
    (f"{FAQ}\t{AQ_KEY}", f"{FAQ}\t{MASK}"),
    # A password beside a count, or a value that only starts like one.
    ("TOTAL_PASS=12. DB_PASS=FAKEhunter2345", f"TOTAL_PASS=12. DB_PASS={MASK}"),
    ("(TOTAL_PASS=12) DB_PASS=FAKEhunter2345", f"(TOTAL_PASS=12) DB_PASS={MASK}"),
    ("TESTS_PASS=12/15 SMTP_PASS=12/FAKEhunter", f"TESTS_PASS=12/15 SMTP_PASS={MASK}"),
    ("TOTAL_PASS=12. DB_PASS=12.FAKEhunter", f"TOTAL_PASS=12. DB_PASS={MASK}"),
    ("TOTAL_PASS=12) DB_PASS=12)FAKEhunter", f"TOTAL_PASS=12) DB_PASS={MASK}"),
    (
        '{"TOTAL_PASS": "12.", "DB_PASS": "FAKEhunter2345"}',
        f'{{"TOTAL_PASS": "12.", "DB_PASS": "{MASK}"}}',
    ),
    # Six digits or more may be a numeric PIN, punctuation or not.
    ("TOTAL_PASS=12. MYSQL_PWD=48213907.", f"TOTAL_PASS=12. MYSQL_PWD={MASK}"),
    ("(N_PASS=3) MYSQL_PWD=482139)", f"(N_PASS=3) MYSQL_PWD={MASK}"),
    ("TESTS_PASS=12/15 DB_PASS=482139/7", f"TESTS_PASS=12/15 DB_PASS={MASK}"),
    # A password in the userinfo, the same user name's or another's.
    (
        (
            "ssh://deploy-automation@host ht"
            + "tps://us"
            + "er:FAKEp"
            + "ass0123@"
            + "host/x"
        ),
        f"ssh://deploy-automation@host https://{MASK}@host/x",
    ),
    ("ssh://deploy-automation:FAKEpass0123@host", f"ssh://{MASK}@host"),
    # A token alone as the userinfo: digits, capital letters, or both.
    (
        "git+ssh://git@github.com/org/repo "
        "https://FAKEtok0123456789abcdef@gw.example.com/v1",
        f"git+ssh://git@github.com/org/repo https://{MASK}@gw.example.com/v1",
    ),
    (
        "https://user@host/path "
        "https://0123456789abcdef0123456789abcdef01234567@github.com/org/repo",
        f"https://user@host/path https://{MASK}@github.com/org/repo",
    ),
    (
        "ssh://deploy-automation@host https://FAKEtokenAbCdEfGhIjKlMn@gw.example.com",
        f"ssh://deploy-automation@host https://{MASK}@gw.example.com",
    ),
]


@pytest.mark.parametrize("text,expected", STILL_MASKED_AFTER_REVIEW)
def test_review_follow_up_secret_beside_harmless_text_is_still_masked(text, expected):
    """Review of PR #134, 2026-11-13, follow-ups: the ``AQ.`` boundary, the
    count after a ``*_PASS`` name and the user name in a URL reopen no leak:
    the key, the password or the token beside the harmless text is masked."""
    assert redact_trajectory_text(text) == expected


@pytest.mark.parametrize(
    "stderr,returncode,fields",
    [
        (
            f"Aborted rollout due to AuthenticationError: {ERROR}",
            0,
            ("error", "verifier_error"),
        ),
        (f"Traceback (most recent call last):\nRuntimeError: {ERROR}", 1, ("error",)),
    ],
)
def test_a_hosted_run_result_json_masks_its_error_text(
    tmp_path, monkeypatch, stderr, returncode, fields
):
    """Review of PR #134, 2026-11-13: a hosted environment run wrote
    result.json's ``error`` and ``verifier_error`` raw
    (``hosted_env._write_run_artifacts``), a key in the provider's error
    included. Both are masked as a native trial's are; the run still counts
    as errored and keeps the unmasked text in memory.

    Review of PR #148, 2026-11-13: ``hosted_env/hosted_run.json`` kept the
    same text raw in its ``error`` and ``verifiers_error``. Neither file
    holds any of the four secrets now."""
    from tests.test_hosted_env_rollout_contract import (
        _classify,
        _invoke,
        _patch_run_custom,
    )

    _patch_run_custom(monkeypatch, stdout="", stderr=stderr, returncode=returncode)
    result = _invoke(tmp_path)

    saved = json.loads((result.run_dir / "result.json").read_text())
    for field in fields:
        assert "upstream said" in saved[field], field
        assert MASK in saved[field], field
        for secret in ERROR_SECRETS:
            assert secret not in saved[field], field
    hosted = json.loads((result.run_dir / "hosted_env" / "hosted_run.json").read_text())
    hosted_fields = {"error": "error", "verifier_error": "verifiers_error"}
    for field in fields:
        assert "upstream said" in hosted[hosted_fields[field]], field
        assert MASK in hosted[hosted_fields[field]], field
    for name in ("result.json", "hosted_env/hosted_run.json"):
        text = (result.run_dir / name).read_text()
        assert [secret for secret in ERROR_SECRETS if secret in text] == [], name
    assert _classify(result) == "errored"
    assert ERROR_SECRETS[1] in result.error


@pytest.mark.parametrize(
    "text",
    [
        "see docs/FAQ.troubleshooting-network-errors for help",
        "Read the FAQ.ProxySetupAndCertificates_v2 page",
        "IRAQ.population_statistics_by_region",
    ],
)
def test_the_export_redactor_keeps_faq_text_too(text):
    """Review of PR #148, 2026-11-13: ``publish/redact.py`` has its own
    ``AQ.`` rule, with no left boundary, so the text the trajectory redactor
    now keeps was still masked at upload."""
    from benchflow.publish.redact import redact_value

    assert redact_value(text) == (text, 0)
    assert redact_value({"note": text, "steps": [text]}) == (
        {"note": text, "steps": [text]},
        0,
    )


@pytest.mark.parametrize(
    "text",
    [
        f"{AQ_KEY} refused, see {FAQ}",
        f"key={AQ_KEY}",
        f"key: {AQ_KEY}",
        f'"{AQ_KEY}"',
        f"({AQ_KEY})",
        f"see {FAQ}\n{AQ_KEY}",
    ],
)
def test_the_export_redactor_still_masks_an_aq_key_beside_faq_text(text):
    """The other half of the pair: a made-up Google AI Studio key at a word
    start is still masked at upload, and the FAQ text beside it is kept."""
    from benchflow.publish.redact import redact_value

    redacted, count = redact_value(text)

    assert AQ_KEY not in redacted and count >= 1
    assert redacted.count(FAQ) == text.count(FAQ)
