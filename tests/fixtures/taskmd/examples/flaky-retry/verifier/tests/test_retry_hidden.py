"""Hidden retry tests. Mounted only in the verifier's sandbox, at /verifier/tests."""

from pathlib import Path

import pytest

from src.http.client import HttpClient, RetryError


class Flaky:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def send(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError(f"timeout {self.calls}")
        return "ok"


def test_retry_error_cause(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RetryError) as err:
        HttpClient(Flaky(failures=10)).get("/status")
    assert isinstance(err.value.__cause__, TimeoutError)
    assert str(err.value.__cause__) == "timeout 4"  # the last of four attempts


def test_delays_double(monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    HttpClient(Flaky(failures=3)).get("/status")
    assert len(delays) == 3
    for got, base in zip(delays, (0.5, 1.0, 2.0)):
        assert base <= got <= base * 1.2


def test_no_new_dependencies():
    base = Path(__file__).resolve().parent.parent / "fixtures" / "requirements.txt"
    assert Path("requirements.txt").read_text().split() == base.read_text().split()
