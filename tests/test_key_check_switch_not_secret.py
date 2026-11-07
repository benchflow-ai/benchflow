"""The key check's opt-out switch is not a credential (merge of PT's key check
with integ's redaction of credential-named variables): ``BENCHFLOW_KEY_CHECK=0``
must not turn every "0" in BenchFlow's messages into ``***``."""

from benchflow.cli._errors import _redact, _secrets
from benchflow.doctor import secret_values


def test_the_key_check_switch_is_not_redacted(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_KEY_CHECK", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-0000")
    assert "0" not in secret_values({"BENCHFLOW_KEY_CHECK": "0"})
    text = "jobs/run/20261017-230510 with sk-test-0000"
    assert _redact(text, _secrets([])) == "jobs/run/20261017-230510 with ***"
