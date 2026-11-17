"""OpenCode's pinned model catalog: one models.dev snapshot that the dry run
reads and every OpenCode launch reads through OPENCODE_MODELS_PATH, the
effort table generated from it, and the refusal of a model it lacks.

R383E round 7 found OpenCode's Gemini levels in the cache its background
refresh fills from models.opencode.ai at launch and every 60 minutes (else the
catalog it bundles), so a level could change between a dry run and a run with
no pin change, and a closed network got the bundled catalog.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.agents import opencode_catalog, registry
from benchflow.agents.launch_env import agent_launch_env
from benchflow.effort_request import EffortRequestError, settle_effort
from tests.test_effort_request import _bench_run, _shell
from tools import opencode_catalog as tool

COMMIT = opencode_catalog.SOURCE["commit"]
NODE_PREFIX = registry._BENCHFLOW_NODE_PREFIX
MISSING = (
    "gemini/gemini-9-nope is not in BenchFlow's pinned OpenCode catalog "
    f"{opencode_catalog.DATE} (models.dev {COMMIT[:12]}), the only one opencode reads"
)


def test_the_snapshot_is_models_dev_at_its_commit_under_its_licence():
    """The lead's condition 1 and 4: models.dev's MIT licence and an
    attribution ship beside the snapshot, whose source commit and date are
    recorded; the snapshot, its metadata and the table agree."""
    source = opencode_catalog.SOURCE
    assert source["source"] == "https://github.com/anomalyco/models.dev"
    assert len(COMMIT) == 40 and source["date"] == source["committed"][:10]
    assert opencode_catalog.LICENSE.read_text().startswith(
        "MIT License\n\nCopyright (c) 2025 models.dev\n"
    )
    readme = (opencode_catalog.SNAPSHOT.parent / "README.md").read_text()
    assert "MIT" in readme and "LICENSE.models-dev" in readme
    assert "tools/opencode_catalog.py build" in readme
    assert tool.problems() == []


def test_a_snapshot_refreshed_without_measuring_again_fails_the_check(
    tmp_path, monkeypatch
):
    """Condition 4: a refresh regenerates the table, or `check` fails."""
    for name in ("SNAPSHOT", "METADATA", "EFFORTS", "LICENSE"):
        copy = tmp_path / getattr(tool, name).name
        shutil.copy(getattr(tool, name), copy)
        monkeypatch.setattr(tool, name, copy)
    catalog = json.loads(tool.SNAPSHOT.read_text())
    catalog["google"]["models"].pop("gemini-4-argon")
    text = tool.snapshot_text(catalog)
    tool.SNAPSHOT.write_text(text)
    metadata = json.loads(tool.METADATA.read_text())
    metadata["sha256"] = opencode_catalog.sha256(text.encode())
    tool.METADATA.write_text(json.dumps(metadata))
    assert tool.problems() == [
        "gemini-efforts.json was measured on another snapshot: measure again",
        "gemini-efforts.json's models are not the snapshot's google models",
    ]


def test_the_effort_table_was_generated_from_the_snapshot_on_the_pinned_opencode():
    table = opencode_catalog.efforts()
    assert table["opencode"] == registry.CACHEABLE_INSTALLS["opencode"]
    assert table["catalog_sha256"] == opencode_catalog.sha256(
        opencode_catalog.snapshot_bytes()
    )
    assert set(table["models"]) == opencode_catalog.provider_models("google")
    assert set(
        registry.OPENCODE_GEMINI_EFFORTS
    ) | registry.OPENCODE_GEMINI_DROPPED == set(table["models"])


@pytest.mark.parametrize("agent", ["opencode", "acpx:opencode", "acpx/opencode"])
@pytest.mark.parametrize(
    "model", ["gemini/gemini-4-argon", "anthropic/claude-opus-5-5", None]
)
def test_every_opencode_launch_env_points_at_the_pinned_catalog(agent, model):
    """Condition 2: on every route, whatever the run's env says."""
    env = agent_launch_env(agent, model, {"OPENCODE_MODELS_PATH": "/elsewhere.json"})
    assert env["OPENCODE_MODELS_PATH"] == registry.OPENCODE_MODELS_PATH


def test_no_other_agent_gets_the_catalog_path():
    assert "OPENCODE_MODELS_PATH" not in agent_launch_env(
        "codex-acp", "openai/gpt-5.5", {}
    )


@pytest.mark.parametrize(
    "environment", ["docker", "remote-docker", "daytona", "modal", "apple-container"]
)
async def test_each_sandbox_launches_opencode_with_the_pinned_catalog(
    tmp_path, environment
):
    """Condition 2, per launch path: the env the ACP transport starts OpenCode
    with, on each sandbox (the sandbox's own transport is handed it)."""
    from benchflow.acp.client import ACPClient
    from benchflow.acp.runtime import connect_acp
    from benchflow.acp.session import ACPSession

    session = ACPSession("s1")
    session.model_state = {
        "currentModelId": "google/gemini-4-argon",
        "availableModels": [{"modelId": "google/gemini-4-argon"}],
    }
    session.config_options = []
    acp = AsyncMock(spec=ACPClient)
    acp.initialize = AsyncMock(return_value=MagicMock(agent_info=None))
    acp.session_new = AsyncMock(return_value=session)
    transport = MagicMock()
    with (
        patch("benchflow.acp.runtime.ContainerTransport", transport),
        patch("benchflow.acp.runtime.ACPClient", return_value=acp),
    ):
        await connect_acp(
            env=AsyncMock(),
            agent="opencode",
            agent_launch="opencode",
            agent_env={"OPENCODE_MODELS_PATH": "/elsewhere.json"},
            sandbox_user=None,
            model="gemini/gemini-4-argon",
            rollout_dir=tmp_path,
            environment=environment,
            agent_cwd="/app",
            reasoning_effort=None,
        )
    assert (
        transport.call_args.kwargs["env"]["OPENCODE_MODELS_PATH"]
        == registry.OPENCODE_MODELS_PATH
    )


def test_the_install_writes_the_snapshot_where_every_launch_points():
    """In the trial's install, the cached Docker and Daytona install layer
    (the same command) and acpx's wrapped install."""
    step = opencode_catalog.install_cmd(
        registry.OPENCODE_MODELS_PATH, f"{NODE_PREFIX}/bin/node"
    )
    assert step in registry.AGENTS["opencode"].install_cmd
    assert step in registry.cacheable_install("opencode")[1]
    assert step in registry._acpx_wrap(registry.AGENTS["opencode"]).install_cmd


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_install_step_writes_the_snapshots_bytes_and_refuses_any_other(tmp_path):
    out = tmp_path / "share/opencode/catalog.json"
    step = opencode_catalog.install_cmd(str(out), shutil.which("node"))
    subprocess.run(["sh", "-c", step], check=True)
    assert out.read_bytes() == opencode_catalog.snapshot_bytes()
    assert out.stat().st_mode & 0o777 == 0o644
    tampered = step.replace(
        opencode_catalog.sha256(opencode_catalog.snapshot_bytes()), "0" * 64
    )
    out.unlink()
    ran = subprocess.run(["sh", "-c", tampered], capture_output=True, text=True)
    assert ran.returncode == 1 and not out.exists()
    assert "the pinned OpenCode catalog decoded to sha256" in ran.stderr


def _wrapper(tmp_path, models_path):
    from tests.test_opencode_family_proxy_tracking import _wrapper_script

    script = _wrapper_script("opencode", "opencode-proxy").replace(
        registry.OPENCODE_MODELS_PATH, str(models_path)
    )
    path = tmp_path / "opencode-proxy"
    path.write_text(script)
    return path


@pytest.mark.parametrize("given", ["unset", "elsewhere", "missing"])
def test_the_wrapper_refuses_to_launch_without_the_pinned_catalog(tmp_path, given):
    """OpenCode reads its bundled catalog when OPENCODE_MODELS_PATH's file is
    not there (1.19.5 and 1.18.11: ``readJson(OPENCODE_MODELS_PATH)`` failing
    falls through to it), so no catalog means no launch."""
    pinned = tmp_path / "models-dev.json"
    if given != "missing":
        pinned.write_bytes(opencode_catalog.snapshot_bytes())
    elsewhere = tmp_path / "other.json"
    elsewhere.write_text("{}")
    env = {"PATH": "/usr/bin:/bin"}
    if given != "unset":
        env["OPENCODE_MODELS_PATH"] = str(elsewhere if given == "elsewhere" else pinned)
    ran = subprocess.run(
        ["sh", str(_wrapper(tmp_path, pinned)), "acp"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert ran.returncode == 1
    assert (
        "OpenCode's pinned model catalog is not at OPENCODE_MODELS_PATH" in ran.stderr
    )


def test_the_wrapper_goes_on_to_launch_with_the_pinned_catalog(tmp_path):
    pinned = tmp_path / "models-dev.json"
    pinned.write_bytes(opencode_catalog.snapshot_bytes())
    ran = subprocess.run(
        ["sh", str(_wrapper(tmp_path, pinned)), "acp"],
        env={"PATH": "/usr/bin:/bin", "OPENCODE_MODELS_PATH": str(pinned)},
        capture_output=True,
        text=True,
    )
    # It execs the agent, which isn't installed here.
    assert "pinned model catalog" not in ran.stderr
    assert ran.returncode != 1


@pytest.mark.parametrize("effort", [None, "high"])
def test_a_mis_cased_model_id_is_refused_before_launch(effort):
    """R600 L1: OpenCode looks a model up by its exact id, so
    gemini/GEMINI-4-ARGON passed the dry run (the check lowercased it) and
    the run exited 1 ("model not found")."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/GEMINI-4-ARGON", effort, {})
    assert str(caught.value) == MISSING.replace(
        "gemini/gemini-9-nope", "gemini/GEMINI-4-ARGON"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_install_step_leaves_the_catalog_readable_under_a_strict_umask(tmp_path):
    """R600 L2: the folders the step creates took the sandbox's umask (700
    under 077), so a non-root agent couldn't read the catalog and the wrapper
    refused after a passing dry run."""
    out = tmp_path / "share/opencode/catalog.json"
    step = opencode_catalog.install_cmd(str(out), shutil.which("node"))
    subprocess.run(["sh", "-c", f"umask 077; {step}"], check=True)
    for folder in (out.parent, out.parent.parent):
        assert folder.stat().st_mode & 0o555 == 0o555, oct(folder.stat().st_mode)
    assert out.stat().st_mode & 0o777 == 0o644
    assert out.read_bytes() == opencode_catalog.snapshot_bytes()


def _wrapper_step(target, models_path):
    """OpenCode's -proxy wrapper install step, as the registry builds it, with
    the wrapper written to *target* and checking for *models_path*."""
    real = f"{registry._BENCHFLOW_BIN_PREFIX}/opencode-proxy"
    step = registry._opencode_family_proxy_wrapper_install(
        "opencode", ".config/opencode/opencode.json", str(models_path)
    )
    assert step.count(real) == 2
    return step.replace(real, str(target))


def test_the_wrapper_install_leaves_it_executable_under_a_strict_umask(tmp_path):
    """R600 L3: ``chmod +x`` honoured the umask, so under 077 the wrapper was
    700 and a non-root agent got "Permission denied" after a passing dry run."""
    target = tmp_path / "opencode-proxy"
    step = _wrapper_step(target, tmp_path / "models-dev.json")
    subprocess.run(["sh", "-c", f"umask 077; {step}"], check=True)
    assert target.stat().st_mode & 0o777 == 0o755


def _sudo_to_nobody() -> bool:
    if shutil.which("sudo") is None:
        return False
    ran = subprocess.run(["sudo", "-n", "-u", "nobody", "true"], capture_output=True)
    root = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    return ran.returncode == 0 and root.returncode == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.skipif(
    not _sudo_to_nobody(), reason="needs passwordless sudo to root and nobody"
)
def test_a_non_root_user_launches_the_wrapper_after_a_strict_umask_install():
    """R600 L2 and L3, as R600 checked them by hand: root installs the catalog
    and the wrapper under umask 077, and ``nobody`` launches the wrapper. It
    starts, reads the catalog (its check passes) and execs the agent, which
    is not installed here (127)."""
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        root.chmod(0o755)
        catalog = root / "share/opencode/models-dev.json"
        wrapper = root / "bin/opencode-proxy"
        steps = " && ".join(
            [
                f"mkdir -p {root}/bin",
                opencode_catalog.install_cmd(str(catalog), shutil.which("node")),
                _wrapper_step(wrapper, catalog),
            ]
        )
        subprocess.run(["sudo", "-n", "sh", "-c", f"umask 077; {steps}"], check=True)
        subprocess.run(["sudo", "-n", "chmod", "755", f"{root}/bin"], check=True)
        read = subprocess.run(
            ["sudo", "-n", "-u", "nobody", "cat", str(catalog)], capture_output=True
        )
        assert read.returncode == 0 and read.stdout == opencode_catalog.snapshot_bytes()
        ran = subprocess.run(
            [
                "sudo",
                "-n",
                "-u",
                "nobody",
                "env",
                "-i",
                "PATH=/usr/bin:/bin",
                f"OPENCODE_MODELS_PATH={catalog}",
                str(wrapper),
                "acp",
            ],
            capture_output=True,
            text=True,
        )
        assert "Permission denied" not in ran.stderr, ran.stderr
        assert "pinned model catalog" not in ran.stderr, ran.stderr
        assert ran.returncode == 127, (ran.returncode, ran.stderr)
    finally:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(root)], check=False)


@pytest.mark.parametrize("effort", [None, "high"])
def test_a_model_the_pinned_catalog_lacks_is_refused_before_launch(effort):
    """Condition 3: never a silent fallback to a live or bundled catalog."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/gemini-9-nope", effort, {})
    assert str(caught.value) == MISSING
    assert "tools/opencode_catalog.py" in caught.value.hint


def test_a_deprecated_model_the_pinned_opencode_drops_is_refused_before_launch():
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/gemini-2.5-flash-image", None, {})
    assert str(caught.value) == (
        "gemini/gemini-2.5-flash-image is deprecated in BenchFlow's pinned OpenCode "
        f"catalog {opencode_catalog.DATE} (models.dev {COMMIT[:12]}), and OpenCode "
        f"{registry.CACHEABLE_INSTALLS['opencode']} drops a deprecated model, so opencode "
        "can't run it"
    )


def test_the_default_reviewer_runs_from_the_pinned_catalog():
    """bench review's default reviewer (opencode, no -m) is a model the
    catalog holds and the pinned OpenCode runs, or a refresh would refuse
    every default review."""
    from benchflow.cli.review import DEFAULT_REVIEW_MODEL

    assert settle_effort("opencode", DEFAULT_REVIEW_MODEL, None, {}).model
    assert DEFAULT_REVIEW_MODEL.split("/", 1)[1] in registry.OPENCODE_GEMINI_EFFORTS


@pytest.mark.parametrize(
    ("agent", "model"),
    [
        ("opencode", "gemini/gemini-4-argon"),
        ("opencode", "anthropic/claude-opus-5-5"),
        ("codex-acp", "gemini/gemini-9-nope"),
    ],
)
def test_a_model_the_catalog_holds_or_does_not_read_runs(agent, model):
    assert settle_effort(agent, model, None, {}).model == model


@pytest.mark.parametrize(
    ("argv", "words"),
    [
        (("-a", "opencode", "-m", "gemini/gemini-9-nope"), MISSING),
        (("-a", "opencode", "-m", "gemini/gemini-9-nope", "-e", "high"), MISSING),
        (
            ("-a", "opencode", "-m", "gemini/GEMINI-4-ARGON"),
            MISSING.replace("gemini/gemini-9-nope", "gemini/GEMINI-4-ARGON"),
        ),
        (
            ("-a", "acpx/opencode", "-m", "gemini/gemini-9-nope"),
            MISSING.replace("opencode reads", "acpx/opencode reads"),
        ),
    ],
)
def test_the_refusal_is_the_dry_runs_and_the_runs(tmp_path, monkeypatch, argv, words):
    _shell(monkeypatch, tmp_path, GEMINI_API_KEY="fake-not-a-key")
    texts = []
    for flags in ([], ["--dry-run"]):
        refused, seen = _bench_run(
            tmp_path / ("dry" if flags else "run"), monkeypatch, *argv, *flags
        )
        assert refused.exit_code == 2, refused.output
        assert seen == {"preflight": [], "configs": []}
        texts.append(" ".join(refused.output.split()))
    assert f"Refused before launch: {words}" in texts[0]
    assert texts[0] == texts[1]


BINARY = os.environ.get("BENCHFLOW_OPENCODE_BIN")
needs_binary = pytest.mark.skipif(
    not BINARY, reason="set BENCHFLOW_OPENCODE_BIN to the pinned OpenCode binary"
)


@needs_binary
def test_the_effort_table_is_what_the_pinned_opencode_sends_whatever_it_fetches():
    """Condition 2: the pinned OpenCode (opencode-linux-x64
    CACHEABLE_INSTALLS["opencode"], bin/opencode) over ACP, as a trial runs
    it, with the snapshot pinned and its live catalog fetch at a stub serving
    another catalog: the levels are the table's. Unpinned, the same fetch
    moves them, so the stub's catalog did reach OpenCode."""
    table = tool.measure_table(BINARY)
    assert table["opencode"] == registry.CACHEABLE_INSTALLS["opencode"]
    assert table["models"] == opencode_catalog.efforts()["models"]
    catalog = opencode_catalog.catalog()
    watched = ["gemini-4-argon", "gemini-4-hydrogen", "gemini-3-flash-preview"]
    unpinned = tool.measure(
        BINARY, watched, models_path=None, live=tool.hostile(catalog)
    )
    assert unpinned == {
        "gemini-4-argon": {"sent": ["low"]},
        "gemini-4-hydrogen": {"sent": ["low", "medium"]},
        "gemini-3-flash-preview": None,
    }


@needs_binary
@pytest.mark.parametrize("network", ["closed", False])
def test_hydrogen_and_argon_run_from_the_pinned_catalog_on_a_closed_network(network):
    """R383E: on a closed network (OpenCode's catalog fetch fails, or is off),
    OpenCode reads the catalog it bundles, which has no gemini-4-hydrogen
    (1.19.5 and 1.18.11) and no gemini-4-argon (1.18.11). Pinned, both send
    the table's levels."""
    models = ["gemini-4-argon", "gemini-4-hydrogen"]
    table = opencode_catalog.efforts()["models"]
    pinned = tool.measure(BINARY, models, models_path=str(tool.SNAPSHOT), fetch=network)
    assert pinned == {m: table[m] for m in models}
    unpinned = tool.measure(BINARY, models, models_path=None, fetch=network)
    assert unpinned["gemini-4-hydrogen"] is None
    bundles_argon = registry.CACHEABLE_INSTALLS["opencode"] != "1.18.11"
    assert (unpinned["gemini-4-argon"] is not None) is bundles_argon


def test_the_install_payload_is_the_snapshot_gzipped():
    """The snapshot travels in the install command, so a refreshed snapshot
    is a new install-layer key (sandbox.env_layers hashes the command)."""
    import gzip
    import shlex

    step = opencode_catalog.install_cmd(registry.OPENCODE_MODELS_PATH, "node")
    payload = shlex.split(step)[3]
    assert (
        gzip.decompress(base64.b64decode(payload)) == opencode_catalog.snapshot_bytes()
    )
