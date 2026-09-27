"""network_mode: allowlist on Daytona, through bench eval run with a real ACP agent.

The oracle runs without a network policy (by design), so the probe is a
``claude-agent-acp`` Bash tool call scripted by the fake model: it fetches
one listed and one unlisted host as the sandbox user and writes the HTTP
codes to /app/net.txt, which the verifier reads.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e import harness as h

PROBE = (
    "for u in https://example.com/ https://www.wikipedia.org/; do "
    "code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' \"$u\" || true); "
    'echo "$u $code"; done > /app/net.txt; cat /app/net.txt'
)
SCRIPTS = {
    "net-probe": [
        {
            "text": "Probing.",
            "tool": "Bash",
            "input": {"command": PROBE, "description": "Probe"},
        },
        {"text": "Probed."},
    ]
}
NET_TEST = """#!/bin/bash
cat /app/net.txt
allowed=$(awk '$1=="https://example.com/"{print $2}' /app/net.txt)
denied=$(awk '$1=="https://www.wikipedia.org/"{print $2}' /app/net.txt)
if [ "$allowed" = "200" ] && [ "$denied" != "200" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def test_allowlist_admits_listed_host_only(
    sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    task = h.write_fake_llm_task(
        tasks_root / "network", "e2e-allowlist", script="net-probe", scripts=SCRIPTS,
        test=NET_TEST, extra_packages=("iptables", "openssl"),
        frontmatter={"agent": {"timeout_sec": 180, "network_mode": "allowlist",
                               "allowed_hosts": ["example.com"]}},
    )  # fmt: skip
    run = h.bench("tasks", "check", task, "--sandbox", sandbox)
    h.assert_exit(run, 0)
    job = jobs_root / "allowlist"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, *h.fake_agent_args("proxy"), "--sandbox", sandbox,
        "--jobs-dir", jobs_root, "--job-name", job.name,
        "--max-sandbox-seconds", str(h.cap_seconds()), log=jobs_root / "allowlist.log",
    )  # fmt: skip
    ledger.record("network_mode allowlist on Daytona (agent probe)", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    trial = h.trial_of(job, "e2e-allowlist")
    stdout = (trial / "verifier" / "test-stdout.txt").read_text()
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}, (stdout, result.get("error"))
    assert "https://example.com/ 200" in stdout
    assert (
        "https://www.wikipedia.org/ 403" in stdout or "wikipedia.org/ 000" in stdout
    ), stdout
    # The refusal is in the block log downloaded with the trial.
    blocked = h.read_jsonl(trial / "trajectory" / "egress_denylist.jsonl")
    assert any(
        b["rule"] == "not-allowlisted" and b["url"].startswith("www.wikipedia.org")
        for b in blocked
    ), blocked
    assert not any("example.com" in b["url"] for b in blocked), blocked
