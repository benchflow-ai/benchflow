"""Live utilization warnings during a job (benchflow._utils.utilization)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils import live_activity, utilization
from benchflow._utils.utilization import Window, summarize
from benchflow.providers.litellm_runtime import HostLiteLLMProcess
from benchflow.rollout import Rollout


def _window(samples: list[list[tuple[str, int | None]]], waiting: int = 0) -> Window:
    window = Window()
    for trials in samples:
        window.add(trials, waiting)
    return window


def test_agents_that_leave_the_model_idle_are_called_underused():
    # The DrugTargetWorld launch: 8 sessions on 4 GPUs, the agents in their own
    # tools most of the time; here a request is in flight in 2 of 8 agents.
    busy_two = [("connected", 1)] * 2 + [("connected", 0)] * 6
    line, warnings = summarize(_window([busy_two] * 4, waiting=40), slots=8, minutes=5)
    assert line == (
        "Utilization, last 5 min (4 samples): 8.0 of 8 slots running (0.0 in "
        "setup, 8.0 with the agent); 40 tasks waiting; a model request in flight "
        "25% of the agents' time (averaging 2.0 at once over the window)"
    )
    assert len(warnings) == 1
    assert "in flight only 25% of their time" in warnings[0]
    assert "if the server is yours and has room" in warnings[0]


def test_slots_empty_while_tasks_wait_are_called_underused():
    half = [("setup", None)] + [("connected", 1)] * 2
    line, warnings = summarize(_window([half] * 3, waiting=10), slots=8, minutes=5)
    assert "3.0 of 8 slots running (1.0 in setup, 2.0 with the agent)" in line
    assert any("only 3.0 of 8 slots ran while tasks waited" in w for w in warnings)


def test_a_setup_bound_window_is_flagged_and_gets_no_concurrency_advice():
    """With most slots in setup on a busy Docker daemon, the
    line advised more sessions; and the parts did not add up."""
    slow = [
        ("setup", None),
        ("image-building", None),
        ("connected", 0),
        ("verifying", None),
    ]
    line, warnings = summarize(_window([slow] * 4, waiting=2), slots=4, minutes=5)
    assert "(2.0 in setup, 1.0 with the agent, 1.0 grading or cleaning up)" in line
    assert any("2.0 of the 4.0 running slots were in setup" in w for w in warnings)
    assert not any("--concurrency" in w for w in warnings)
    # The first window is setup by nature: no setup warning there.
    _line, first = summarize(
        _window([slow] * 4, waiting=2), slots=4, minutes=2, first=True
    )
    assert not any("running slots were in setup" in w for w in first)


def test_a_busy_run_has_no_warning_and_no_proxy_is_said():
    full = [("connected", 2)] * 4
    _line, warnings = summarize(_window([full] * 2, waiting=3), slots=4, minutes=5)
    assert warnings == []
    unseen = [("connected", None)] * 4
    line, warnings = summarize(_window([unseen], waiting=0), slots=4, minutes=5)
    assert "model load not visible (no BenchFlow model proxy on this machine)" in line
    assert warnings == []
    starting = [("setup", None)] * 4
    line, _warnings = summarize(_window([starting], waiting=2), slots=4, minutes=1)
    # Setup: no agent yet, so no word on the model.
    assert line.endswith("4.0 in setup, 0.0 with the agent); 2 tasks waiting")
    # An oracle's solve runs in its installed sandbox and never connects: the
    # agent's time, with no model to speak of (a benchmark's hour-long oracle read as
    # an hour of setup).
    oracle = [("installed", None)]
    line, _warnings = summarize(_window([oracle] * 3), slots=1, minutes=5)
    assert line.endswith(
        "1.0 of 1 slot running (0.0 in setup, 1.0 with the agent); 0 tasks waiting"
    )


@pytest.mark.asyncio
async def test_the_watcher_logs_a_warning_and_keeps_it_for_the_dashboard(
    monkeypatch, caplog
):
    monkeypatch.setattr(utilization, "_latest", None)
    monkeypatch.setattr(
        live_activity, "utilization_sample", lambda: [("connected", 0)] * 4
    )
    with caplog.at_level(logging.INFO, logger="benchflow.utilization"):
        task = asyncio.ensure_future(
            utilization.watch(
                4, lambda: 7, sample_sec=0.001, first_report_sec=0.0, report_sec=60.0
            )
        )
        for _ in range(200):
            await asyncio.sleep(0.005)
            if utilization.latest() is not None:
                break
        task.cancel()
    line, warnings = utilization.latest()
    assert line.startswith("Utilization, last 0 min (")
    assert warnings and "in flight only 0% of their time" in warnings[0]
    assert "Underused:" in caplog.text


@pytest.mark.asyncio
async def test_the_job_s_end_reports_its_last_stretch(monkeypatch, caplog):
    monkeypatch.setattr(utilization, "_latest", None)
    monkeypatch.setattr(utilization, "_LAST_LINE_SEC", 0.004)
    monkeypatch.setattr(
        live_activity, "utilization_sample", lambda: [("connected", 1)] * 2
    )
    with caplog.at_level(logging.INFO, logger="benchflow.utilization"):
        task = asyncio.ensure_future(
            utilization.watch(2, lambda: 0, sample_sec=0.001, first_report_sec=3600)
        )
        await asyncio.sleep(0.05)
        assert utilization.latest() is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    line, warnings = utilization.latest()
    assert "2.0 of 2 slots running" in line and warnings == []
    assert "a model request in flight 100% of the agents' time" in caplog.text


@pytest.mark.asyncio
async def test_a_sample_that_fails_never_stops_the_watcher(monkeypatch):
    def broken():
        raise RuntimeError("registry mid-update")

    monkeypatch.setattr(live_activity, "utilization_sample", broken)
    task = asyncio.ensure_future(
        utilization.watch(2, lambda: 0, sample_sec=0.001, first_report_sec=0.0)
    )
    await asyncio.sleep(0.02)
    assert not task.done()
    task.cancel()


def test_the_sample_reads_each_live_rollout_s_phase_and_requests():
    live_activity.register(
        "a", SimpleNamespace(_phase="connected", model_requests_in_flight=lambda: 2)
    )
    live_activity.register("b", SimpleNamespace(_phase="setup"))
    try:
        sample = dict(live_activity.utilization_sample())
    finally:
        live_activity.unregister("a")
        live_activity.unregister("b")
    assert sample == {"connected": 2, "setup": None}


def test_requests_in_flight_are_the_proxy_s_taken_less_its_logged(tmp_path: Path):
    log = tmp_path / "litellm.jsonl"
    proxy = SimpleNamespace(log_path=log)
    assert HostLiteLLMProcess.requests_in_flight(proxy) == 0
    Path(f"{log}.requests").write_text("1\n2\n3\n")
    Path(f"{log}.logged").write_text("1\n")
    assert HostLiteLLMProcess.requests_in_flight(proxy) == 2
    rollout = SimpleNamespace(
        _usage_runtime=SimpleNamespace(
            server=SimpleNamespace(
                requests_in_flight=lambda: HostLiteLLMProcess.requests_in_flight(proxy)
            )
        )
    )
    assert Rollout.model_requests_in_flight(rollout) == 2
    # No proxy on this machine (a sandbox-side gateway, or none at all).
    assert (
        Rollout.model_requests_in_flight(SimpleNamespace(_usage_runtime=None)) is None
    )


def _seen(**bodies: object) -> dict:
    import json
    from collections import Counter

    return {
        key: Counter({json.dumps(value, sort_keys=True): 12})
        for key, value in bodies.items()
    }


def test_the_sampling_sent_is_checked_against_the_vendor_and_the_reference():
    """The Nov 3 launch sent thinking off at temperature 1.0 to Qwen3.8-27B;
    the first utilization report now says so, two minutes in."""
    from benchflow.model_settings import vendor_setting
    from benchflow.run_card import sampling_check

    card = {
        "model": {"vendor_setting": vendor_setting("Qwen/Qwen3.8-27B").as_record()},
        "reference": {
            "deviations": [
                {"key": "temperature", "reference": 0.6, "run": None, "kind": "check"},
                {"key": "thinking", "reference": True, "run": None, "kind": "check"},
            ]
        },
    }
    seen = _seen(temperature=1.0, chat_template_kwargs={"enable_thinking": False})
    line, differs = sampling_check(card, seen)
    assert differs
    assert line.startswith(
        "Sampling sent so far (the first 12 requests): temperature 1.0; "
        'chat_template_kwargs {"enable_thinking": false}; thinking off.'
    )
    assert (
        "Against Qwen's setting for Qwen3.8-27B with thinking off: temperature 0.7 "
        "(sent 1); top_p 0.8 (not sent: the server's default applies)"
    ) in line
    assert (
        "Against the reference: temperature 0.6 (sent 1); thinking true (sent off)"
        in line
    )
    matching = _seen(
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
    )
    line, differs = sampling_check({"model": card["model"]}, matching)
    assert not differs and line.endswith("with thinking on: it matches.")


@pytest.mark.asyncio
async def test_the_watcher_says_the_sampling_once(monkeypatch, tmp_path, caplog):
    import json

    trial = tmp_path / "t__1"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(
        json.dumps({"request": {"body": {"temperature": 1.0}}}) + "\n"
    )
    monkeypatch.setattr(utilization, "_latest", None)
    monkeypatch.setattr(live_activity, "utilization_sample", lambda: [("connected", 1)])
    monkeypatch.setattr(live_activity, "trial_dirs", lambda: [trial])
    with caplog.at_level(logging.INFO, logger="benchflow.utilization"):
        task = asyncio.ensure_future(
            utilization.watch(
                1, lambda: 0, sample_sec=0.001, first_report_sec=0.0, report_sec=0.0
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
    said = [
        r for r in caplog.records if r.getMessage().startswith("Sampling sent so far")
    ]
    assert len(said) == 1
    assert (
        said[0].getMessage()
        == "Sampling sent so far (the first request): temperature 1.0."
    )


def test_the_live_trial_folders_are_listed():
    live_activity.register("a", SimpleNamespace(_rollout_dir=Path("/jobs/j/a__1")))
    live_activity.register("b", SimpleNamespace(_rollout_dir=None))
    try:
        assert live_activity.trial_dirs() == [Path("/jobs/j/a__1")]
    finally:
        live_activity.unregister("a")
        live_activity.unregister("b")
