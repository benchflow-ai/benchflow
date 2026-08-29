"""T2 oracle-invariant proofs for composed checkpoints, against real Docker.

Guards the composed-checkpoint layer ("feat(branch): compose sandbox +
environment checkpoints in the branch engine"; docs/rollout-branching-rfc.md
WS-1, T2 tier per WS-3/§5; FrontierPhysics#73). PR number to be added on
submission.

The rollout-branching RFC's validation plan (docs/rollout-branching-rfc.md §5)
names three tiers; ``tests/test_branch_composed.py`` is T1 (fakes). This file
is the T2 tier: an executable proof on a live ``DockerSandbox`` that the
composed checkpoint→restore of RFC §3.1 is lossless — zero delta ⇒ identical
observable state — and that a known-bad delta is detectable:

1. ``test_container_layer_zero_delta_round_trip`` — the container layer alone
   (``Sandbox.snapshot``/``restore``, i.e. ``docker commit`` → ``docker run``)
   round-trips a file-tree digest computed inside the sandbox, and a
   post-restore mutation moves the digest again (negative control).
2. ``test_composed_two_layer_round_trip`` — ``checkpoint_composed`` /
   ``restore_composed`` over a real ``ManifestEnvironment`` (``[environment.
   state] kind=sqlite``) plus the sandbox layer: file digest AND a sqlite
   query both return to their pre-checkpoint values, and the recorded
   ``StageSnapshot`` carries both layer refs.
3. ``test_zero_delta_reward_proxy_invariant`` — a deterministic in-sandbox
   verifier (the reward function in miniature) reads PASS before checkpoint,
   FAIL after a destructive mutation, PASS again after restore: the
   reward-equality argument of RFC §5 (a) executed end-to-end.

Marking: ``pytest.mark.live`` — the repo's marker for tests needing a real
Docker daemon (pyproject registers it as "requires real Anthropic API and
Docker daemon"; this file needs Docker only, no API keys — there is no
narrower docker-only marker, and ``integration`` implies Gemini/Daytona
credentials, so ``live`` is the closest fit). The default addopts deselect it;
when selected without a reachable daemon, the module-scoped ``docker_prereqs``
fixture skips every test with a specific reason instead of erroring.

Construction path: ``DockerSandbox`` is built exactly the way
``benchflow.sandbox.setup._create_sandbox_environment`` builds it for
``sandbox_type="docker"`` (a tiny environment dir + ``SandboxConfig``), with a
throwaway Dockerfile (alpine + the ``sqlite3`` CLI that
``ManifestEnvironment``'s sqlite backup/restore shells out to). Teardown
always runs: compose down via ``Sandbox.stop`` (with its force-kill-by-label
fallback), a container sweep by compose-project label, and ``docker rmi`` of
every ``bf-snap-<env>-*`` image this file's snapshots created (the tag
pattern from ``DockerSandbox.snapshot``). The built ``bf__<env>`` base image
is deliberately kept so repeat runs hit the Docker build cache.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from benchflow.branch import StageSnapshot, checkpoint_composed, restore_composed
from benchflow.environment.manifest import EnvironmentManifest, StateSpec
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox.docker import (
    DockerSandbox,
    _sanitize_docker_compose_project_name,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.trajectories.tree import RolloutTree

pytestmark = [pytest.mark.live]

# One environment name for the whole file: the image builds once
# (``DockerSandbox._image_build_locks`` + the Docker build cache) and the
# snapshot tag prefix below stays precise to this file's snapshots.
_ENV_NAME = "branch-composed-proof"
# Tag pattern from DockerSandbox.snapshot():
#   _sanitize_docker_image_name(f"bf-snap-{environment_name}-{suffix}")
_SNAP_IMAGE_PREFIX = f"bf-snap-{_ENV_NAME}-"

# sqlite3 CLI is required by ManifestEnvironment's `.backup`-based
# snapshot/restore; busybox provides find/sort/xargs/stat/sha256sum.
_DOCKERFILE = """\
FROM alpine:3.20
RUN apk add --no-cache sqlite
WORKDIR /app
"""

# Marker prefix for value-bearing exec output. `_run_docker_compose_command`
# merges stderr into stdout, so compose warnings (e.g. "Found orphan
# containers") can pollute the stream; values are extracted by prefix match
# instead of trusting raw stdout.
_MARKER = "BFPROOF:"


def _docker_unavailable_reason() -> str | None:
    """Return why real-Docker tests cannot run here, or None if they can."""
    if shutil.which("docker") is None:
        return "docker CLI not installed"
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"docker daemon unreachable: {exc}"
    if result.returncode != 0:
        return "docker daemon not running (`docker info` failed)"
    return None


@pytest.fixture(scope="module")
def docker_prereqs() -> None:
    """Skip (never error) when the Docker daemon is absent.

    Deferred to fixture time — like ``test_smoke.smoke_prereqs`` — so the
    subprocess only fires when a live test is actually selected.
    """
    reason = _docker_unavailable_reason()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="module")
def environment_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal sandbox environment definition: just a Dockerfile."""
    env_dir = tmp_path_factory.mktemp("branch-composed-env")
    (env_dir / "Dockerfile").write_text(_DOCKERFILE)
    return env_dir


def _sweep_project_containers(session_id: str) -> None:
    """Best-effort removal of any container left carrying our project label.

    ``DockerSandbox.restore`` re-creates ``main`` outside compose (raw
    ``docker run`` with compose labels); if ``compose down`` misses it, this
    sweep — the same label filter ``_force_kill_project`` uses — catches it.
    """
    project = _sanitize_docker_compose_project_name(session_id)
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    container_ids = listed.stdout.split()
    if container_ids:
        subprocess.run(
            ["docker", "rm", "-f", "-v", *container_ids],
            capture_output=True,
            timeout=60,
            check=False,
        )
    subprocess.run(
        ["docker", "network", "rm", f"{project}_default"],
        capture_output=True,
        timeout=30,
        check=False,
    )


def _remove_snapshot_images() -> None:
    """``docker rmi`` every snapshot image this file's tests created.

    Matches on the exact ``bf-snap-<env-name>-`` prefix so unrelated images
    (including other tests' ``bf-snap-*``) are never touched.
    """
    listed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    for repository in listed.stdout.split():
        if repository.startswith(_SNAP_IMAGE_PREFIX):
            subprocess.run(
                ["docker", "rmi", "-f", repository],
                capture_output=True,
                timeout=60,
                check=False,
            )


@asynccontextmanager
async def _live_sandbox(
    environment_dir: Path, rollout_dir: Path
) -> AsyncIterator[DockerSandbox]:
    """Start a real DockerSandbox; guarantee teardown of containers + snapshots.

    Mirrors the ``sandbox_type == "docker"`` branch of
    ``benchflow.sandbox.setup._create_sandbox_environment``. One sandbox per
    test keeps every round trip starting from a clean compose-managed
    ``main`` container.
    """
    session_id = f"bf-branch-composed-{uuid.uuid4().hex[:8]}"
    rollout_paths = RolloutPaths(rollout_dir=rollout_dir)
    rollout_paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment_dir,
        environment_name=_ENV_NAME,
        session_id=session_id,
        rollout_paths=rollout_paths,
        task_env_config=SandboxConfig(),
    )
    try:
        await sandbox.start(force_build=False)
        yield sandbox
    finally:
        try:
            # delete=False: plain `compose down` — containers and network go,
            # the built bf__<env> image stays cached for the next run.
            await sandbox.stop(delete=False)
        finally:
            _sweep_project_containers(session_id)
            _remove_snapshot_images()


async def _exec_ok(sandbox: DockerSandbox, command: str) -> None:
    """Exec a state-mutating command and assert it succeeded."""
    result = await sandbox.exec(command, timeout_sec=120)
    assert result.return_code == 0, (
        f"in-sandbox command failed (rc={result.return_code}): {command!r}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


async def _exec_value(sandbox: DockerSandbox, command: str) -> str:
    """Exec a value-producing command; return its single-line output.

    The value is echoed behind a unique marker prefix and extracted by prefix
    match, so merged compose warnings cannot corrupt digest comparisons.
    """
    wrapped = f'__bf_out="$({command})" && echo "{_MARKER}${{__bf_out}}"'
    result = await sandbox.exec(wrapped, timeout_sec=120)
    assert result.return_code == 0, (
        f"in-sandbox command failed (rc={result.return_code}): {command!r}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    values = [
        line[len(_MARKER) :].strip()
        for line in (result.stdout or "").splitlines()
        if line.startswith(_MARKER)
    ]
    assert len(values) == 1, (
        f"expected exactly one {_MARKER} line for {command!r}, "
        f"got {values!r}\nstdout: {result.stdout}"
    )
    return values[0]


def _digest_command(exclude_glob: str | None = None) -> str:
    """One deterministic line summarizing /app: file contents + tree + modes.

    Content hashes plus a ``stat`` listing of names and permission bits, all
    digested by a final ``sha256sum``. ``exclude_glob`` drops files by
    basename (used for the sqlite DB, whose ``.backup``-restored copy is
    logically — not necessarily byte — identical; the DB is asserted through
    a SQL query instead).

    Delegates to the src helper so the proofs measure with the same oracle
    production records — the reviewer's P1-A repro (PR #1046 second review)
    drove *this* helper's original ``find | sort | xargs`` pipeline, which
    word-split legal filenames and emitted a successful wrong digest; the
    null-safe fail-closed form now lives in exactly one place.
    """
    from benchflow.sandbox.workspace_digest import workspace_digest_command

    return workspace_digest_command("/app", exclude_basename=exclude_glob)


async def test_container_layer_zero_delta_round_trip(
    docker_prereqs: None, environment_dir: Path, tmp_path: Path
) -> None:
    """Sandbox.snapshot → destructive mutation → Sandbox.restore is lossless.

    Zero delta ⇒ identical state digest; and the digest is a real oracle —
    a post-restore mutation (the negative control) moves it again.
    """
    async with _live_sandbox(environment_dir, tmp_path) as sandbox:
        await _exec_ok(
            sandbox,
            "mkdir -p /app/t1/nested"
            " && echo alpha > /app/t1/state.txt"
            " && echo beta > /app/t1/nested/deep.txt"
            " && chmod 640 /app/t1/state.txt"
            " && chmod 700 /app/t1/nested",
        )
        digest_before = await _exec_value(sandbox, _digest_command())

        image = await sandbox.snapshot()
        assert image.provider == "docker"
        assert image.ref.startswith(_SNAP_IMAGE_PREFIX)

        await _exec_ok(
            sandbox,
            "echo corrupted > /app/t1/state.txt"
            " && rm -rf /app/t1/nested"
            " && echo extra > /app/t1/extra.txt"
            " && chmod 600 /app/t1/state.txt",
        )
        digest_mutated = await _exec_value(sandbox, _digest_command())
        assert digest_mutated != digest_before, (
            "mutation must move the digest — otherwise the digest is no oracle"
        )

        await sandbox.restore(image)

        digest_restored = await _exec_value(sandbox, _digest_command())
        assert digest_restored == digest_before, (
            "zero-delta restore must reproduce the exact pre-snapshot state"
        )

        # Negative control: a known-bad delta after restore is detectable.
        await _exec_ok(sandbox, "echo tampered >> /app/t1/state.txt")
        digest_tampered = await _exec_value(sandbox, _digest_command())
        assert digest_tampered != digest_before, (
            "post-restore tampering must be visible in the digest"
        )


async def test_composed_two_layer_round_trip(
    docker_prereqs: None, environment_dir: Path, tmp_path: Path
) -> None:
    """checkpoint_composed/restore_composed round-trip both layers (RFC §3.1).

    A real ManifestEnvironment (``[environment.state] kind=sqlite``) over the
    live sandbox: after restore, the file-plane digest AND a sqlite query
    both read their pre-checkpoint values, and the node's recorded
    StageSnapshot carries both layer refs.
    """
    db_path = "/app/t2/env.db"
    db_query = f"sqlite3 {db_path} \"SELECT v FROM kv WHERE k='phase';\""
    file_digest = _digest_command(exclude_glob="env.db*")

    async with _live_sandbox(environment_dir, tmp_path) as sandbox:
        await _exec_ok(
            sandbox, "mkdir -p /app/t2 && echo file-plane > /app/t2/file.txt"
        )
        await _exec_ok(
            sandbox,
            f"sqlite3 {db_path} "
            '"CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT); '
            "INSERT INTO kv VALUES('phase','before');\"",
        )

        # Real env-plane construction: manifest with declared sqlite state
        # over the injected live sandbox (the test_manifest_env shape, no
        # fakes). No [[services]]: the state plane is what T2 exercises.
        manifest = EnvironmentManifest(
            name=_ENV_NAME,
            image=f"bf__{_ENV_NAME}",
            state=StateSpec(kind="sqlite", paths=[db_path]),
        )
        environment = ManifestEnvironment(manifest, sandbox=sandbox)
        await environment.provision(None)
        probe = await environment.readiness()
        assert probe.ready, f"environment readiness failed: {probe.error}"

        digest_before = await _exec_value(sandbox, file_digest)
        assert await _exec_value(sandbox, db_query) == "before"

        tree = RolloutTree()
        node = tree.root
        snap = await checkpoint_composed(
            node, environment=environment, sandbox=sandbox, stage="pre-verify"
        )
        # The recorded StageSnapshot must carry both layer refs.
        assert isinstance(snap, StageSnapshot)
        assert node.state["snapshot"] is snap
        assert snap.stage == "pre-verify"
        assert isinstance(snap.environment_ref, StateSnapshot)
        assert snap.environment_ref.path.startswith("/tmp/benchflow-snapshots/")
        assert snap.sandbox_ref is not None
        assert snap.sandbox_ref.provider == "docker"
        assert snap.sandbox_ref.ref.startswith(_SNAP_IMAGE_PREFIX)

        # Destructive mutation on both planes.
        await _exec_ok(
            sandbox, f"sqlite3 {db_path} \"UPDATE kv SET v='after' WHERE k='phase';\""
        )
        await _exec_ok(
            sandbox,
            "echo scribbled > /app/t2/file.txt && echo junk > /app/t2/junk.txt",
        )
        assert await _exec_value(sandbox, db_query) == "after"
        digest_mutated = await _exec_value(sandbox, file_digest)
        assert digest_mutated != digest_before

        await restore_composed(node, environment=environment, sandbox=sandbox)

        assert await _exec_value(sandbox, file_digest) == digest_before, (
            "composed restore must reproduce the pre-checkpoint file plane"
        )
        assert await _exec_value(sandbox, db_query) == "before", (
            "composed restore must roll the declared sqlite state back"
        )


async def test_zero_delta_reward_proxy_invariant(
    docker_prereqs: None, environment_dir: Path, tmp_path: Path
) -> None:
    """RFC §5 (a) in miniature: zero-delta restore preserves the reward.

    A deterministic in-sandbox verifier — the reward function's stand-in —
    reads PASS on the checkpointed state, FAIL after a destructive mutation
    (the detectable known-bad delta), and PASS again after restore. Uses the
    composed ops in their sandbox-only shape (``require_layers={"sandbox"}``
    per RFC §3.1: a stateless env + snapshot-capable sandbox can branch).
    """
    verifier = (
        'if [ "$(cat /app/t3/answer.txt 2>/dev/null)" = "42" ]'
        " && [ -f /app/t3/nested/marker ];"
        " then echo PASS; else echo FAIL; fi"
    )

    async with _live_sandbox(environment_dir, tmp_path) as sandbox:
        await _exec_ok(
            sandbox,
            "mkdir -p /app/t3/nested"
            " && echo 42 > /app/t3/answer.txt"
            " && touch /app/t3/nested/marker",
        )
        assert await _exec_value(sandbox, verifier) == "PASS"

        tree = RolloutTree()
        node = tree.root
        snap = await checkpoint_composed(node, sandbox=sandbox, stage="pre-verify")
        assert snap.environment_ref is None
        assert snap.sandbox_ref is not None

        await _exec_ok(
            sandbox, "echo 43 > /app/t3/answer.txt && rm -f /app/t3/nested/marker"
        )
        assert await _exec_value(sandbox, verifier) == "FAIL", (
            "a known-bad delta must flip the verifier — otherwise PASS-after-"
            "restore would prove nothing"
        )

        await restore_composed(node, sandbox=sandbox)

        assert await _exec_value(sandbox, verifier) == "PASS", (
            "zero-delta restore must return the verifier to its pre-snapshot "
            "verdict — the reward-equality invariant"
        )


async def test_restore_keeps_the_rollout_bind_mounts_and_the_host_sees_writes(
    docker_prereqs: None, environment_dir: Path, tmp_path: Path
) -> None:
    """T2 proof for the live regression: a restored container is still mounted.

    ``DockerSandbox`` bind-mounts the rollout's ``verifier`` / ``agent`` /
    ``artifacts`` directories into ``main``. ``restore()`` re-creates that
    container outside compose, and a replacement created without those mounts
    is silently broken: the verifier's ``reward.txt`` is written inside the
    container, the host reads nothing, and the branch child is scored from a
    file that does not exist (the run that reported two arms at ``0.00``).

    Two assertions, in the order that matters: the mounts are *declared* on
    the restored container (``docker inspect``), and they are *effective* — a
    file written to the mounted path inside the container appears on the host,
    which is the property the verifier actually depends on. The pre-restore
    write proves the mount worked before, so a post-restore failure is
    attributable to the restore.
    """
    rollout_dir = tmp_path / "mounted-run"
    async with _live_sandbox(environment_dir, rollout_dir) as sandbox:
        host_verifier_dir = (rollout_dir / "verifier").resolve()
        container_verifier_dir = str(SandboxPaths.verifier_dir)

        await _exec_ok(sandbox, f"echo pre > {container_verifier_dir}/reward.txt")
        assert (host_verifier_dir / "reward.txt").read_text().strip() == "pre", (
            "the compose-created container must bind-mount the verifier dir — "
            "otherwise this test proves nothing about restore"
        )

        image = await sandbox.snapshot()
        await sandbox.restore(image)

        mounts = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{json .Mounts}}",
                await sandbox._main_container_id() or "",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert mounts.returncode == 0, mounts.stderr
        destinations = {
            mount["Destination"] for mount in json.loads(mounts.stdout or "[]")
        }
        assert container_verifier_dir in destinations, (
            "the restored container dropped the verifier bind mount: "
            f"{sorted(destinations)}"
        )
        assert str(SandboxPaths.agent_dir) in destinations
        assert str(SandboxPaths.artifacts_dir) in destinations

        # Effective, not just declared: the host must see a post-restore write.
        await _exec_ok(sandbox, f"echo 1.0 > {container_verifier_dir}/reward.txt")
        assert (host_verifier_dir / "reward.txt").read_text().strip() == "1.0", (
            "a file written to the mounted verifier path after restore must "
            "appear on the host — this is the read the verifier depends on"
        )

        # And the live mount check agrees with reality.
        assert await sandbox.has_host_mount(
            host_dir=host_verifier_dir, container_dir=container_verifier_dir
        )


async def test_a_childs_verifier_run_cannot_destroy_the_parents_evidence(
    docker_prereqs: None, environment_dir: Path, tmp_path: Path
) -> None:
    """T2 proof for the shared-mount clobbering, on a real bind mount.

    Because ``restore()`` replays the rollout's bind mounts (the test above),
    every branch child writes ``/logs/verifier`` straight into the *parent's*
    host directory — and a child whose own rollout paths differ sees
    ``has_host_mount() == False``, so it runs ``clear_verifier_output_dir``
    first, whose ``find /logs/verifier -mindepth 1 -exec rm -rf {} +`` empties
    the parent's directory before writing anything of its own. That is the
    exact command replayed here, in the container, against a live mount.

    The three assertions are the three moves of
    :class:`~benchflow.branch_artifacts.MountedArtifacts`: after ``hold`` the
    parent's files are out of the blast radius (and the mount really did carry
    the deletion through to the host, so this is not a vacuous test); after
    ``hand_off`` the child's own output is under its node directory; after
    ``release`` the parent's evidence is back at its canonical path with the
    transient hold directory gone.
    """
    from benchflow.branch_artifacts import (
        MountedArtifacts,
        child_mount_dir,
        parent_hold_dir,
    )

    rollout_dir = tmp_path / "branch-evidence-run"
    async with _live_sandbox(environment_dir, rollout_dir) as sandbox:
        host_verifier_dir = (rollout_dir / "verifier").resolve()
        container_verifier_dir = str(SandboxPaths.verifier_dir)

        # The parent's own verifier run, written through the mount.
        await _exec_ok(sandbox, f"echo 1.0 > {container_verifier_dir}/reward.txt")
        await _exec_ok(
            sandbox, f"echo 'parent tests passed' > {container_verifier_dir}/stdout.txt"
        )
        assert (host_verifier_dir / "reward.txt").read_text().strip() == "1.0"

        holder = MountedArtifacts.hold(run_dir=rollout_dir, parent_id="root")

        # A child's verifier: clear the shared output dir, then score itself.
        await _exec_ok(
            sandbox,
            f"find {container_verifier_dir} -mindepth 1 -exec rm -rf -- {{}} +",
        )
        assert not (host_verifier_dir / "reward.txt").exists(), (
            "the child's clear must reach the host through the mount — "
            "otherwise this test proves nothing about the clobbering"
        )
        assert (
            parent_hold_dir(rollout_dir, "root") / "verifier" / "reward.txt"
        ).read_text().strip() == "1.0"
        await _exec_ok(sandbox, f"echo 0.0 > {container_verifier_dir}/reward.txt")

        child_dir = child_mount_dir(rollout_dir, "root", "n1")
        holder.hand_off(child_dir)
        assert (child_dir / "verifier" / "reward.txt").read_text().strip() == "0.0"

        holder.release()
        assert (host_verifier_dir / "reward.txt").read_text().strip() == "1.0"
        assert (
            host_verifier_dir / "stdout.txt"
        ).read_text().strip() == "parent tests passed"
        assert not parent_hold_dir(rollout_dir, "root").exists()

        # The mount still works after all that moving — the mount root was
        # emptied, never removed.
        await _exec_ok(sandbox, f"echo post > {container_verifier_dir}/after.txt")
        assert (host_verifier_dir / "after.txt").read_text().strip() == "post"
