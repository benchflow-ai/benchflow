"""Tests for the readiness-probe loop."""

from benchflow.environment.readiness import wait_for_readiness


async def test_ready_when_all_http_probes_pass():
    async def probe(url: str) -> bool:
        return True

    result = await wait_for_readiness(
        http=["http://x/health", "http://y/health"],
        tcp=[],
        timeout_sec=5,
        _http_check=probe,
        _tcp_check=None,
    )
    assert result.ready is True
    assert result.error is None
    assert set(result.checked) == {"http://x/health", "http://y/health"}


async def test_times_out_when_a_probe_never_passes():
    async def probe(url: str) -> bool:
        return False

    result = await wait_for_readiness(
        http=["http://x/health"],
        tcp=[],
        timeout_sec=1,
        poll_interval=0.1,
        _http_check=probe,
        _tcp_check=None,
    )
    assert result.ready is False
    assert "timed out" in (result.error or "")


async def test_becomes_ready_after_initial_failures():
    calls = {"n": 0}

    async def probe(url: str) -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    result = await wait_for_readiness(
        http=["http://x/health"],
        tcp=[],
        timeout_sec=5,
        poll_interval=0.01,
        _http_check=probe,
        _tcp_check=None,
    )
    assert result.ready is True
    assert calls["n"] >= 3


async def test_tcp_probe_checked():
    async def tcp(port: int) -> bool:
        return port == 9001

    result = await wait_for_readiness(
        http=[],
        tcp=[9001],
        timeout_sec=2,
        poll_interval=0.1,
        _http_check=None,
        _tcp_check=tcp,
    )
    assert result.ready is True
    assert "9001" in result.checked[0]


async def test_empty_probes_ready_immediately():
    result = await wait_for_readiness(http=[], tcp=[], timeout_sec=2)
    assert result.ready is True
    assert result.checked == []
