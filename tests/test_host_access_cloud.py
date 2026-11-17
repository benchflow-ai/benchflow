"""Host access on a cloud VM needs a second opt-in (EN-609).

A task whose compose gives it this machine (the host's network, the Docker
socket, a privileged container: Toolathlon's k8s tasks) is admitted only
with BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1. On a cloud VM that hands the agent
the VM's cloud credentials through the metadata server, so there it also
needs BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD=1, and the refusal says why.
"""

from __future__ import annotations

import pytest

from benchflow.task import compose_env as ce


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.delenv(ce.ALLOW_HOST_ACCESS_ON_CLOUD_VAR, raising=False)
    monkeypatch.setattr(ce, "_CLOUD_PROBE", [True])
    return monkeypatch


def test_on_a_cloud_vm_the_first_opt_in_is_not_enough(cloud):
    cloud.setenv(ce.ALLOW_HOST_ACCESS_VAR, "1")
    assert ce.host_access_allowed() is False
    remedy = ce._host_remedy()
    assert "this VM's cloud credentials" in remedy
    assert ce.ALLOW_HOST_ACCESS_ON_CLOUD_VAR in remedy


def test_both_opt_ins_admit_host_access_on_a_cloud_vm(cloud):
    cloud.setenv(ce.ALLOW_HOST_ACCESS_VAR, "1")
    cloud.setenv(ce.ALLOW_HOST_ACCESS_ON_CLOUD_VAR, "1")
    assert ce.host_access_allowed() is True


def test_off_a_cloud_the_first_opt_in_is_enough(monkeypatch):
    monkeypatch.delenv(ce.ALLOW_HOST_ACCESS_ON_CLOUD_VAR, raising=False)
    monkeypatch.setattr(ce, "_CLOUD_PROBE", [False])
    monkeypatch.setenv(ce.ALLOW_HOST_ACCESS_VAR, "1")
    assert ce.host_access_allowed() is True
    monkeypatch.delenv(ce.ALLOW_HOST_ACCESS_VAR)
    assert ce.host_access_allowed() is False
    assert ce._host_remedy() == ce._HOST_REMEDY


def test_the_probe_sends_no_metadata_header_and_no_token_path(monkeypatch):
    sent = {}

    class Conn:
        def __init__(self, host, port, timeout):
            sent["where"] = (host, port, timeout)

        def request(self, method, path, headers=None, **_):
            sent["request"] = (method, path, dict(headers or {}))

        def getresponse(self):
            return type("R", (), {"status": 403})()

        def close(self):
            pass

    import http.client

    monkeypatch.setattr(http.client, "HTTPConnection", Conn)
    monkeypatch.setattr(ce, "_CLOUD_PROBE", [])
    assert ce.cloud_metadata_server_answers() is True
    method, path, headers = sent["request"]
    assert method == "GET" and path == "/computeMetadata/v1/instance/id"
    assert "token" not in path
    assert not any(h.lower() == "metadata-flavor" for h in headers)
    assert sent["where"][0] == "169.254.169.254"


def test_no_answer_is_not_a_cloud(monkeypatch):
    import http.client

    class Down:
        def __init__(self, *a, **k):
            raise OSError("no route")

    monkeypatch.setattr(http.client, "HTTPConnection", Down)
    monkeypatch.setattr(ce, "_CLOUD_PROBE", [])
    assert ce.cloud_metadata_server_answers() is False
