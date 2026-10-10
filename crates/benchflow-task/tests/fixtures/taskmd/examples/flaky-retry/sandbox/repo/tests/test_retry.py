from src.http.client import HttpClient


class Flaky:
    """A transport that times out a set number of times, then answers."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def send(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError(f"timeout {self.calls}")
        return "ok"


def test_timeout_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    transport = Flaky(failures=3)
    assert HttpClient(transport).get("/status") == "ok"
    assert transport.calls == 4  # the first attempt and three retries
