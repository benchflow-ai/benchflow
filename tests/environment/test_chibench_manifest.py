"""The chi-bench manifest parses and declares the expected environment.

chi-bench is the canonical ``owns_lifecycle = true`` single-image SMSB with
``env_var`` task selection — the architecture's external proof that a heavy
stateful benchmark onboards via a ~25-line manifest, environment untouched.
"""

from pathlib import Path

from benchflow.environment.manifest import load_manifest

MANIFEST_PATH = Path("benchmarks/chi-bench/environment.toml")


def test_chibench_manifest_loads():
    m = load_manifest(MANIFEST_PATH)
    assert m.name == "chi-bench"
    assert m.image == "chi-bench:latest"
    assert m.base_image is None
    assert m.isolation == "per_task"


def test_chibench_manifest_owns_its_lifecycle():
    m = load_manifest(MANIFEST_PATH)
    # The image's tini entrypoint starts cb serve itself, so the framework
    # declares no [[services]] — owns_lifecycle = true forbids them.
    assert m.owns_lifecycle is True
    assert m.services == []


def test_chibench_manifest_uses_env_var_task_selection():
    m = load_manifest(MANIFEST_PATH)
    # One image serves every task; the task id is injected at runtime via
    # the env var docker/entrypoint.sh reads.
    assert m.task_selection.mechanism == "env_var"
    assert m.task_selection.key == "CHI_BENCH_TASK_ID"
    # It must reach PID 1, not just a docker exec call.
    assert m.task_selection.inject_into == "entrypoint"


def test_chibench_manifest_declares_the_four_exposed_ports():
    m = load_manifest(MANIFEST_PATH)
    # Unified HTTP :8023 + provider :8020 / payer :8100 / cm :8200 FastMCP.
    assert m.all_ports == [8020, 8023, 8100, 8200]


def test_chibench_manifest_gates_on_the_real_health_endpoint():
    m = load_manifest(MANIFEST_PATH)
    # The unified FastAPI server's GET /health on :8023; with no [[services]]
    # the readiness probe must be declared explicitly.
    assert m.readiness.http == ["http://localhost:8023/health"]
    assert m.effective_http == ["http://localhost:8023/health"]
    assert m.readiness.timeout_sec == 120


def test_chibench_manifest_forwards_the_anthropic_key():
    m = load_manifest(MANIFEST_PATH)
    # The verifier's WorkspaceJudge shells out to the Claude Code CLI.
    assert m.forward_env.keys == ["ANTHROPIC_API_KEY"]


def test_chibench_manifest_is_stateless_until_state_table_lands():
    m = load_manifest(MANIFEST_PATH)
    # No [environment.state] table yet — snapshot/restore unsupported.
    assert m.state is None
