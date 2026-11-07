"""``bench eval run``'s engine path loads only what a run uses.

An oracle or nop trial on Docker starts no ACP agent, no model proxy and no
integrity audit, yet the run path imported all three: the ACP SDK's schema
(about 0.2 s), httpx with its command-line tool (pygments), the integrity
pipeline and the summary-export models, before the first trial started. The
engine bench measured 1.36 s of every Python trial outside the engine's own
timing. Each check below runs in a fresh interpreter, imports the modules a
run imports and makes the calls an oracle trial makes, then reads what was
loaded. The lazy names keep resolving to the same objects.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

import pytest

_RUN_PATH = """
import json, sys, asyncio
import benchflow.cli.main, benchflow.cli.doctor, benchflow.eval_plan
import benchflow.evaluation, benchflow.rollout, benchflow.rollout_planes
import benchflow.sandbox.docker, benchflow.task.formats
from benchflow.evaluation import EvaluationConfig
from benchflow.providers.runtime import extract_usage, stop_provider_runtime

EvaluationConfig(agent="oracle", environment="docker")
from benchflow.usage_tracking import usage_unavailable
assert extract_usage(None) == usage_unavailable()
asyncio.run(stop_provider_runtime(None))
print(json.dumps(sorted(sys.modules)))
"""

# Loaded only by an ACP agent, a model proxy, an HTTP readiness probe, an
# integrity audit or --summary-out.
_NOT_ON_THE_RUN_PATH = (
    "acp",
    "acp.schema",
    "benchflow.acp.client",
    "benchflow.acp.runtime",
    "httpx",
    "pygments",
    "benchflow.providers.litellm_runtime",
    "benchflow.integrity.contract",
    "benchflow.integrity.emit",
    "benchflow.job_export",
)


def _loaded(code: str, tmp_path) -> set[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_the_run_path_loads_no_acp_proxy_or_audit_modules(tmp_path):
    loaded = _loaded(_RUN_PATH, tmp_path)
    assert "benchflow.rollout" in loaded  # the check ran the real path
    assert sorted(m for m in _NOT_ON_THE_RUN_PATH if m in loaded) == []


@pytest.mark.parametrize(
    "package",
    [
        "benchflow.acp",
        "benchflow.acp.types",
        "benchflow.contracts",
        "benchflow.integrity",
    ],
)
def test_every_public_name_still_resolves(package):
    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, name


def test_lazy_names_are_the_objects_they_name():
    from acp import meta, schema

    import benchflow.acp as acp_pkg
    import benchflow.contracts as contracts
    import benchflow.integrity as integrity
    from benchflow.acp import client, session
    from benchflow.acp import types as acp_types
    from benchflow.agents import protocol
    from benchflow.integrity import trial, verdict

    assert acp_types.TextContent is schema.TextContentBlock
    assert acp_types.PromptResult is schema.PromptResponse
    assert acp_types.ClientInfo is schema.Implementation
    assert acp_types.AgentCapabilities is schema.AgentCapabilities
    assert acp_types.ACP_PROTOCOL_VERSION == meta.PROTOCOL_VERSION
    assert acp_types.ContentBlock == (
        schema.TextContentBlock | schema.ImageContentBlock | schema.ResourceContentBlock
    )
    assert acp_pkg.ACPClient is client.ACPClient
    assert acp_pkg.ACPSession is session.ACPSession
    assert contracts.StopReason is protocol.StopReason is acp_types.StopReason
    assert contracts.Agent is protocol.Agent
    assert integrity.audit_trial is trial.audit_trial
    assert integrity.IntegrityVerdict is verdict.IntegrityVerdict
    assert integrity.normalize_integrity_mode is trial.normalize_integrity_mode
    with pytest.raises(AttributeError):
        _ = acp_types.NoSuchName  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = contracts.NoSuchName  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("value", "mode"),
    [(None, "off"), (False, "off"), (True, "audit"), (" Strict ", "strict")],
)
def test_integrity_mode_normalizes_as_before(value, mode):
    from benchflow.integrity.constants import normalize_integrity_mode

    assert normalize_integrity_mode(value) == mode
    with pytest.raises(ValueError, match="integrity must be one of"):
        normalize_integrity_mode("loud")
