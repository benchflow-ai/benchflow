"""Guards escaped credential redaction at PY8 base 363d815b."""

import json

import pytest

from benchflow.cli._errors import run_cli
from benchflow.cli.main import app
from benchflow.doctor import redact_exact, redacted_traceback

VALUES = [
    "demo\\secret-0123456789",
    "demo\nsecret-0123456789",
    "demo\tsecret-'\"-0123456789",
    "demo-秘密-0123456789",
]
FORMS = [str, repr, ascii, json.dumps, lambda v: json.dumps(v, ensure_ascii=False)]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("form", FORMS)
def test_complete_serialized_credentials_are_redacted(value, form):
    """Guards 363d815b across raw, repr, ASCII and both JSON encodings."""
    rendered = form(value)
    masked = redact_exact(f"unrelated detail: {rendered}", [value])
    assert "***" in masked
    assert "unrelated detail:" in masked
    assert form(value) not in masked
    assert "***" in redacted_traceback(ValueError(rendered), [value])


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("json_mode", [False, True])
def test_explain_never_formats_a_known_credential_into_an_error(
    tmp_path, monkeypatch, capsys, value, json_mode
):
    """Guards 363d815b's concrete explain repr boundary in text and JSON."""
    monkeypatch.setenv("D2_PASSWORD", value)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["explain", value, *(["--json"] if json_mode else [])])
    assert caught.value.code != 0
    out, err = capsys.readouterr()
    output = str(json.loads(out)) if json_mode else out + err
    assert "***" in output
    for form in FORMS:
        assert form(value) not in output
