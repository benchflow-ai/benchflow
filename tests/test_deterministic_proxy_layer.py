"""The in-sandbox proxy from its cached layer, on a real sandbox (PY8 U2b).

The deterministic tier's allowlisted Codex task (proxy in the sandbox) runs
three times through ``bench eval run``: with ``--fresh-install`` (the proxy
installs in the trial), then twice with layers on (the default: the first
may build the proxy layer, the second reuses it). Each trial's config.json
says what happened, and one fixed request through the fake provider gives
usage records with the same schema and token fields whether the proxy ran
from the layer or was installed in the trial.

On Docker the three trials share one task image, built here once. A layer's
key holds the task image's content digest, and a rebuild from the Dockerfile
gives the same bytes only while the daemon's BuildKit cache still holds every
step. When the daemon drops those records between trials (its garbage
collection, other builds on the host), the rebuild runs ``apt-get`` again,
writes new bytes, and the second trial rightly misses (batch 46 on VM 1);
the reuse check would then test the daemon, not BenchFlow.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.integration.deterministic import harness as h
from tests.test_deterministic_integration import (  # noqa: F401  (fixtures)
    ALLOWLIST_VARIANT,
    SANDBOX,
    _run,
    _usage_record,
    det_root,
    fake_url,
    needs_sandbox,
    require_sandbox,
)

pytestmark = pytest.mark.deterministic


@pytest.fixture(scope="session")
def proxy_layer_trials(det_root, fake_url, request) -> dict[str, Path]:  # noqa: F811
    image = _pinned_task_image(det_root, request) if SANDBOX == "docker" else None
    runs = {}
    for name, extra in (
        ("fresh", ["--fresh-install"]),
        ("layer-1", []),
        ("layer-2", []),
    ):
        route = "codex-in-sandbox"
        extra = ["--retry-attempts", "0", "--harness", "acp", *extra]
        if image is None:
            run = _run(
                det_root,
                f"proxy-layer-{name}",
                ["eval", "run"],
                [ALLOWLIST_VARIANT],
                fake_url,
                route=route,
                extra=extra,
            )
        else:
            root = det_root / f"proxy-layer-{name}"
            task = h.materialize_task(ALLOWLIST_VARIANT, root / "tasks")
            task_md = task / "task.md"
            text = task_md.read_text()
            assert "sandbox:\n" in text
            task_md.write_text(
                text.replace("sandbox:\n", f"sandbox:\n  docker_image: {image}\n", 1)
            )
            run = h.run_bench(
                ["eval", "run"],
                tasks_dir=task,
                jobs_dir=root / "jobs",
                sandbox=SANDBOX,
                host_fake_url=fake_url,
                route=route,
                extra=extra,
            )
        runs[name] = run.trial("codex-allowlist")
    return runs


def _pinned_task_image(base: Path, request) -> str:
    """The template task's image, built once from its Dockerfile: its ID."""
    root = base / "proxy-layer-image"
    context = h.materialize_task(ALLOWLIST_VARIANT, root) / "environment"
    tag = f"benchflow-test-proxy-layer-{uuid.uuid4().hex[:12]}"
    iidfile = root / "task-image.id"
    subprocess.run(
        ["docker", "build", "--iidfile", str(iidfile), "-t", tag, str(context)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    request.addfinalizer(
        lambda: subprocess.run(
            ["docker", "image", "rm", tag], check=True, capture_output=True, timeout=60
        )
    )
    return iidfile.read_text().strip()


def _layers(trial: Path) -> dict[str, Any]:
    return h.read_json(trial / "config.json").get("layers") or {}


@needs_sandbox
def test_a_fresh_install_says_so_and_installs_in_the_trial(proxy_layer_trials):
    trial = proxy_layer_trials["fresh"]
    assert h.read_json(trial / "result.json")["rewards"]["reward"] == 1.0
    layers = _layers(trial)
    assert layers["proxy"] == {"status": "fresh", "reason": "fresh install asked for"}
    assert layers["agent"]["status"] == "fresh"
    assert h.read_json(trial / "timing.json")["litellm_install"] > 1.0


@needs_sandbox
def test_the_second_trial_reuses_the_proxy_layer(proxy_layer_trials):
    first = _layers(proxy_layer_trials["layer-1"])["proxy"]
    second = _layers(proxy_layer_trials["layer-2"])["proxy"]
    assert first["status"] in {"hit", "miss"}
    assert first["install"] == "from layer"
    assert second["status"] == "hit"
    assert second["install"] == "from layer"
    assert second["key"] == first["key"]
    assert second["image"] == f"benchflow-layer:proxy-{first['key']}"
    for name in ("layer-1", "layer-2"):
        trial = proxy_layer_trials[name]
        assert h.read_json(trial / "result.json")["rewards"]["reward"] == 1.0
    # This task provisions an environment before installation. The proxy
    # still reuses its layer, but the fix must install the agent
    # after that mutable phase rather than trusting its recipe stamp.
    agent = _layers(proxy_layer_trials["layer-2"])["agent"]
    assert agent["status"] == "fresh"
    assert "provisioning" in agent["reason"]


@needs_sandbox
def test_usage_records_match_from_the_layer_and_from_a_fresh_install(
    proxy_layer_trials,
):
    fresh = _usage_record(proxy_layer_trials["fresh"])
    warm = _usage_record(proxy_layer_trials["layer-2"])
    assert warm["record_keys"] == fresh["record_keys"]
    assert warm["usage"] == fresh["usage"]
    assert warm["agent_result"] == fresh["agent_result"]
