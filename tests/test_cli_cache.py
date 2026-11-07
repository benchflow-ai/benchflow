"""``bench cache list`` / ``clean`` / ``clean --remove`` (PY8 U13).

Docker is the in-memory stand-in of ``tests/test_env_layers_agent.py``,
extended with the commands the inventory reads; it logs every command, so
each test can say which ones removed something. Daytona is a fake sync
client with a snapshot list.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.sandbox import _docker_sweep, env_layers
from benchflow.sandbox.env_layers import (
    AGENT,
    LAYER_AGENT_LABEL,
    LAYER_AGENT_VERSION_LABEL,
    LAYER_BASE_LABEL,
    LAYER_KEY_LABEL,
    LAYER_LABEL,
    LAYER_LIBC_LABEL,
    PROXY,
    DockerLayers,
    alias_owner_tag,
    layer_tag,
    marker,
)
from tests.test_env_layers_agent import FakeDocker


def _optional(module: str) -> Any:
    """``module``, or None on a BenchFlow without it (the fails-before base),
    so these tests fail there on what BenchFlow does, not on an import."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


_layer_lock = _optional("benchflow.sandbox._layer_lock")
_pull_provenance = _optional("benchflow.sandbox._pull_provenance")

KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_P = "c" * 64
DEAD_PID = 4_000_001
# A process of this host that is gone: another boot, a pid that does not exist.
DEAD_OWNER = (
    f"{env_layers._host_hash(_docker_sweep.current_process().host)}-{DEAD_PID}-123-"
    "00000000-0000-0000-0000-000000000000"
)
REMOVING = ("image rm", "rmi", "builder prune", "image prune", "tag", "untag")


class InventoryDocker(FakeDocker):
    """FakeDocker plus the plain ``image ls`` and ``builder du`` the cache reads.

    ``store`` is the daemon's image store. "containerd": each pull and tag
    of a name gets a new LastTagTime. "classic": a pull leaves an image at
    the zero time, and ``docker tag`` sets one time for the whole image
    that untagging does not clear.
    """

    def __init__(self) -> None:
        super().__init__()
        self.created: dict[str, str] = {}
        self.record_sizes: dict[str, str] = {}
        self.store = "containerd"
        self.tag_ticks = 0

    def layer(
        self,
        kind: str,
        key: str,
        *,
        created: str = "2026-10-01T00:00:00.123456789Z",
        size: int = 1000,
        agent: str = "codex-acp",
    ) -> Any:
        labels = {
            LAYER_LABEL: kind,
            LAYER_KEY_LABEL: key,
            LAYER_BASE_LABEL: "sha256:" + "2" * 64,
            LAYER_LIBC_LABEL: "glibc",
        }
        if kind == AGENT:
            labels |= {LAYER_AGENT_LABEL: agent, LAYER_AGENT_VERSION_LABEL: "2.0.1"}
        image = self.add_image(layer_tag(kind, key), labels=labels, size=size)
        image.info["Created"] = created
        return image

    def add_image(self, tag: str, **kwargs: Any) -> Any:
        image = super().add_image(tag, **kwargs)
        # Like Docker's containerd image store: each pull or tag has a time.
        stamp = f"2026-10-16T00:00:00.{self._n:09d}Z"
        if self.store == "classic":
            stamp = "0001-01-01T00:00:00Z"
        image.info["Metadata"] = {"LastTagTime": stamp}
        return image

    def record(self, kind: str, key: str) -> None:
        self.records.append(
            {
                "id": f"rec{len(self.records)}",
                "description": f"mount / from exec /bin/sh -c {marker(kind, key)} true",
            }
        )

    async def run(self, *args, stdin=None, timeout=0):
        argv = list(args)
        if argv[:1] == ["info"] and "{{.ID}}" in argv:
            self.calls.append(argv)
            return self._ok("fake-daemon-1\n")
        if argv == ["info", "--format", "{{json .DriverStatus}}"]:
            self.calls.append(argv)
            if self.store == "containerd":
                return self._ok('[["driver-type","io.containerd.snapshotter.v1"]]\n')
            return self._ok(
                '[["Backing Filesystem","extfs"],["Supports d_type","true"]]\n'
            )
        if argv[:1] == ["tag"] and len(argv) == 3 and self.store == "classic":
            done = await super().run(*args, stdin=stdin, timeout=timeout)
            image = self.tags.get(argv[2])
            if done.code == 0 and image is not None:
                # The image's one time, which stays after the tag is removed.
                self.tag_ticks += 1
                stamp = f"2026-10-16T12:00:00.{self.tag_ticks:09d}Z"
                image.info["Metadata"] = {"LastTagTime": stamp}
            return done
        if argv == ["image", "ls", "--format", "{{json .}}"]:
            self.calls.append(argv)
            rows = []
            for tag, image in self.tags.items():
                repo, _, name = (
                    tag.rpartition(":") if ":" in tag else (tag, "", "latest")
                )
                rows.append(
                    json.dumps(
                        {
                            "Repository": repo,
                            "Tag": name,
                            "ID": image.id[7:19],
                            "CreatedAt": image.created,
                        }
                    )
                )
            return self._ok("".join(f"{r}\n" for r in rows))
        if argv[:3] == ["builder", "du", "--verbose"]:
            self.calls.append(argv)
            return self._ok(
                "".join(
                    f"ID:\t{r['id']}\nCreated at:\t2026-10-02 00:00:00.5 +0000 UTC\n"
                    f"Size:\t4.096kB\nDescription:\t{r['description']}\n\n"
                    for r in self.records
                )
            )
        return await super().run(*args, stdin=stdin, timeout=timeout)

    def removing_calls(self) -> list[list[str]]:
        return [
            c
            for c in self.calls
            if " ".join(c[:2]) in REMOVING or c[:1] in (["tag"], ["rmi"])
        ]


@pytest.fixture
def docker(monkeypatch, tmp_path_factory) -> InventoryDocker:
    monkeypatch.setenv(
        "BENCHFLOW_LAYER_LOCK_DIR", str(tmp_path_factory.mktemp("held") / "locks")
    )
    monkeypatch.setenv(
        "BENCHFLOW_PULL_RECORD_DIR", str(tmp_path_factory.mktemp("pulls"))
    )
    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    DockerLayers._claims.clear()
    getattr(DockerLayers, "_daemon_ids", {}).clear()
    if _layer_lock is not None:
        _layer_lock.clear_process_locks()
    real = _docker_sweep._pid_exists
    monkeypatch.setattr(
        _docker_sweep, "_pid_exists", lambda pid: pid != DEAD_PID and real(pid)
    )
    assert env_layers.alias_owner_state(DEAD_OWNER) == "gone"
    fake = InventoryDocker()

    async def run(self, *args, stdin=None, timeout=0):
        return await fake.run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(DockerLayers, "_docker", run)
    return fake


def _bench(*args: str):
    return CliRunner().invoke(app, ["cache", *args], prog_name="bench")


def _json(result) -> dict:
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _names(payload: dict) -> list[str]:
    return [item["name"] for item in payload["items"]]


def _stock(docker: InventoryDocker) -> None:
    docker.layer(AGENT, KEY_A)
    docker.layer(PROXY, KEY_P)
    docker.record(AGENT, KEY_A)
    docker.record(PROXY, KEY_P)
    docker.record(AGENT, KEY_B)  # an orphan: its image is gone
    # Never BenchFlow's layers: an unlabelled image under the layer
    # repository, a branch snapshot image, a task image.
    docker.add_image("benchflow-layer:mine")
    docker.add_image("bf-snap-trial-1:latest")
    docker.add_image("benchflow/hello:1.0")


# -- refusals and the base -------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["list"], ["clean"], ["clean", "--remove"], ["list", "--json"]],
    ids=["list", "clean", "remove", "json"],
)
def test_the_command_exists(args, docker):
    """On a BenchFlow without `bench cache` these exit 2 (unknown command)."""
    result = _bench(*args, "-s", "docker")
    assert result.exit_code == 0, result.output
    assert "No such command" not in result.output


@pytest.mark.parametrize("kind", ["sandboxes", "checkpoints"])
def test_kinds_python_keeps_elsewhere_exit_2_and_name_the_command(kind, docker):
    result = _bench("clean", "--only", kind)
    assert result.exit_code == 2
    assert "bench sandbox cleanup" in result.output
    assert docker.calls == []


@pytest.mark.parametrize("kind", ["sources", "registries", "logs"])
def test_kinds_python_does_not_keep_exit_2(kind, docker):
    result = _bench("list", "--only", kind)
    assert result.exit_code == 2
    assert "keeps no" in result.output


def test_unsupported_backends_exit_2_with_a_next_step(docker):
    result = _bench("list", "-s", "remote-docker")
    assert result.exit_code == 2
    assert "DOCKER_HOST" in result.output


# -- list ------------------------------------------------------------------


def test_list_shows_only_validated_layers_and_marker_records(docker):
    _stock(docker)
    payload = _json(_bench("list", "-s", "docker", "--json"))
    names = _names(payload)
    assert layer_tag(AGENT, KEY_A) in names
    assert layer_tag(PROXY, KEY_P) in names
    assert not any(
        n.startswith(("benchflow-layer:mine", "bf-snap-", "benchflow/hello"))
        for n in names
    )
    records = {i["key"]: i for i in payload["items"] if i["type"] == "records"}
    assert set(records) == {KEY_A, KEY_P, KEY_B}
    assert records[KEY_B]["image_present"] is False
    assert payload["age_basis"] == "created"
    agent = next(i for i in payload["items"] if i["name"] == layer_tag(AGENT, KEY_A))
    assert agent["bytes"] == 1000  # docker image inspect's Size
    assert agent["what"] == "codex-acp 2.0.1"
    assert docker.removing_calls() == []


def test_a_layer_image_whose_labels_do_not_match_is_not_listed(docker):
    image = docker.layer(AGENT, KEY_A)
    image.info["Config"]["Labels"][LAYER_KEY_LABEL] = KEY_B
    payload = _json(_bench("list", "-s", "docker", "--json"))
    assert layer_tag(AGENT, KEY_A) not in _names(payload)


def test_list_and_dry_run_remove_neither_aliases_nor_records(docker):
    _stock(docker)
    docker.tags[f"bf__layer__dead-trial:{DEAD_OWNER}"] = docker.tags[
        layer_tag(AGENT, KEY_A)
    ]
    before = dict(docker.tags)
    records = list(docker.records)
    for args in (["list"], ["clean"], ["clean", "--json"]):
        result = _bench(*args, "-s", "docker")
        assert result.exit_code == 0, result.output
    assert docker.tags == before and docker.records == records
    assert docker.removing_calls() == []


def test_json_keeps_stdout_machine_readable_and_notes_on_stderr(docker):
    docker.layer(AGENT, KEY_A)
    result = _bench("list", "-s", "docker,daytona", "--json")
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    assert "BENCHFLOW_DAYTONA_OWNER" in result.stderr


# -- clean -----------------------------------------------------------------


def test_clean_prints_and_removes_nothing(docker):
    _stock(docker)
    result = _bench("clean", "-s", "docker")
    assert result.exit_code == 0, result.output
    assert "would remove" in result.output
    assert "--remove" in result.output
    assert docker.removing_calls() == []


def test_clean_remove_runs_only_scoped_commands(docker):
    _stock(docker)
    result = _bench("clean", "-s", "docker", "--remove")
    assert result.exit_code == 0, result.output
    removing = docker.removing_calls()
    assert removing, "nothing was removed"
    allowed_rm = {layer_tag(AGENT, KEY_A), layer_tag(PROXY, KEY_P)}
    for call in removing:
        if call[:2] == ["image", "rm"]:
            # U10: never the untagged parents (--no-prune).
            assert call[2:] in (["--no-prune", t] for t in allowed_rm), call
        elif call[:2] == ["builder", "prune"]:
            assert call[2:4] == ["--all", "--force"]
            assert call[4] == "--filter" and call[5].startswith(
                'description~="benchflow[.]layer='
            ), call
        else:
            pytest.fail(f"unexpected removing command {call}")
    assert layer_tag(AGENT, KEY_A) not in docker.tags
    assert layer_tag(PROXY, KEY_P) not in docker.tags
    for kept in (
        "benchflow-layer:mine",
        "bf-snap-trial-1:latest",
        "benchflow/hello:1.0",
    ):
        assert kept in docker.tags
    assert docker.records == []


def test_only_images_keeps_records_and_only_build_cache_keeps_images(docker):
    _stock(docker)
    assert (
        _bench("clean", "-s", "docker", "--only", "images", "--remove").exit_code == 0
    )
    assert layer_tag(AGENT, KEY_A) not in docker.tags
    assert len(docker.records) == 3
    assert not [c for c in docker.calls if c[:2] == ["builder", "prune"]]

    docker2 = docker
    docker2.layer(AGENT, KEY_A)
    docker2.calls.clear()
    assert (
        _bench("clean", "-s", "docker", "--only", "build-cache", "--remove").exit_code
        == 0
    )
    assert layer_tag(AGENT, KEY_A) in docker2.tags
    assert docker2.records == []
    assert not [c for c in docker2.calls if c[:2] == ["image", "rm"]]


def test_held_layers_stay_with_the_reason(docker):
    docker.layer(AGENT, KEY_A)
    docker.layer(AGENT, KEY_B)
    # A live trial's alias (this process) and a container hold them.
    docker.tags[f"bf__layer__trial-1:{alias_owner_tag()}"] = docker.tags[
        layer_tag(AGENT, KEY_A)
    ]
    docker.used_ids.add(docker.tags[layer_tag(AGENT, KEY_B)].id)
    result = _bench("clean", "-s", "docker", "--remove")
    assert result.exit_code == 0, result.output
    assert layer_tag(AGENT, KEY_A) in docker.tags
    assert layer_tag(AGENT, KEY_B) in docker.tags
    assert "in use: alias bf__layer__trial-1" in result.output
    assert "container" in result.output


def test_a_dead_trials_alias_holds_nothing_on_remove(docker):
    docker.layer(AGENT, KEY_A)
    alias = f"bf__layer__dead-trial:{DEAD_OWNER}"
    docker.tags[alias] = docker.tags[layer_tag(AGENT, KEY_A)]
    result = _bench("clean", "-s", "docker", "--only", "images", "--remove")
    assert result.exit_code == 0, result.output
    assert alias not in docker.tags
    assert layer_tag(AGENT, KEY_A) not in docker.tags


def test_a_layer_whose_lock_is_held_stays(docker):
    docker.layer(AGENT, KEY_A)
    assert _layer_lock is not None, "no host lock in this BenchFlow"
    holder = _layer_lock.HostLock("fake-daemon-1", AGENT, KEY_A)
    assert holder.try_acquire()
    try:
        result = _bench("clean", "-s", "docker", "--only", "images", "--remove")
    finally:
        holder.release()
    assert result.exit_code == 0, result.output
    assert layer_tag(AGENT, KEY_A) in docker.tags
    assert "holds its lock" in result.output


def test_an_unusable_lock_folder_keeps_everything(docker, tmp_path, monkeypatch):
    docker.layer(AGENT, KEY_A)
    docker.record(AGENT, KEY_A)
    planted = tmp_path / "planted"
    planted.write_text("not a folder")
    monkeypatch.setenv("BENCHFLOW_LAYER_LOCK_DIR", str(planted))
    result = _bench("clean", "-s", "docker", "--remove")
    assert result.exit_code == 0, result.output
    assert layer_tag(AGENT, KEY_A) in docker.tags
    assert len(docker.records) == 1
    assert "lock cannot be used" in result.output


def test_older_than_counts_from_creation_and_keeps_unknown_ages(docker):
    now = datetime.now(UTC)
    old = (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    new = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    docker.layer(AGENT, KEY_A, created=old)
    docker.layer(AGENT, KEY_B, created=new)
    docker.layer(PROXY, KEY_P, created="")
    payload = _json(
        _bench(
            "clean", "-s", "docker", "--only", "images", "--older-than", "30d", "--json"
        )
    )
    actions = {i["name"]: i for i in payload["items"]}
    assert payload["age_basis"] == "created"
    assert actions[layer_tag(AGENT, KEY_A)]["action"] == "would remove"
    assert actions[layer_tag(AGENT, KEY_B)]["action"] == "kept"
    assert "less than 30d" in actions[layer_tag(AGENT, KEY_B)]["kept"]
    assert actions[layer_tag(PROXY, KEY_P)]["action"] == "kept"
    assert "unknown" in actions[layer_tag(PROXY, KEY_P)]["kept"]


def test_a_bad_age_exits_2(docker):
    assert _bench("list", "--older-than", "soon").exit_code == 2


# -- options given before the command (`bench cache --older-than 30d clean`) --


def test_older_than_before_clean_still_keeps_what_is_newer(docker):
    """`bench cache --help` offers the options on the group; given there they
    were dropped, so this removed layers built two days ago."""
    now = datetime.now(UTC)
    old = (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    new = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    docker.layer(AGENT, KEY_A, created=old)
    docker.layer(AGENT, KEY_B, created=new)
    result = _bench(
        "--older-than", "30d", "clean", "-s", "docker", "--only", "images", "--remove"
    )
    assert result.exit_code == 0, result.output
    assert layer_tag(AGENT, KEY_A) not in docker.tags
    assert layer_tag(AGENT, KEY_B) in docker.tags
    assert "less than 30d" in result.output


def test_only_before_clean_removes_only_that_kind(docker):
    _stock(docker)
    result = _bench("--only", "build-cache", "clean", "-s", "docker", "--remove")
    assert result.exit_code == 0, result.output
    assert not [c for c in docker.calls if c[:2] == ["image", "rm"]]
    assert layer_tag(AGENT, KEY_A) in docker.tags
    assert docker.records == []


def test_backend_and_json_before_list_are_used(docker):
    docker.layer(AGENT, KEY_A)
    payload = _json(_bench("-s", "docker", "--json", "list"))
    assert payload["backends"] == ["docker"]
    assert layer_tag(AGENT, KEY_A) in _names(payload)


def test_an_option_given_before_and_after_with_different_values_exits_2(docker):
    _stock(docker)
    result = _bench("--only", "images", "clean", "--only", "build-cache", "--remove")
    assert result.exit_code == 2
    assert "give it once" in result.output
    assert docker.removing_calls() == []


# -- base pins -------------------------------------------------------------


def test_only_new_pins_of_dead_local_processes_are_removed(docker):
    docker.add_image("bf__task")
    task_id = docker.tags["bf__task"].id
    dead = f"benchflow-layer-base/agent-{KEY_A}:{DEAD_OWNER}"
    live = f"benchflow-layer-base/agent-{KEY_B}:{alias_owner_tag()}"
    remote = f"benchflow-layer-base/agent-{KEY_P}:ffffffffffff-12-1-x"
    legacy = f"benchflow-layer-base:{KEY_A[:16]}-0123456789ab"
    for ref in (dead, live, remote, legacy):
        docker.tags[ref] = docker.tags["bf__task"]
    result = _bench("clean", "-s", "docker", "--only", "images", "--remove")
    assert result.exit_code == 0, result.output
    assert dead not in docker.tags
    for kept in (live, remote, legacy, "bf__task"):
        assert kept in docker.tags, kept
    assert docker.tags["bf__task"].id == task_id
    assert "older pin" in result.output


# -- pulled images ----------------------------------------------


def _pulls() -> list[Any]:
    if _pull_provenance is None:
        return []
    return _pull_provenance.read_pulls("fake-daemon-1")


def _pulled(docker: InventoryDocker, ref: str, owner: str = DEAD_OWNER) -> str:
    image = docker.add_image(ref)
    if _pull_provenance is not None:
        _pull_provenance.record_pull(
            "fake-daemon-1",
            ref,
            image.id,
            owner,
            image.info["Metadata"]["LastTagTime"],
        )
    return image.id


@pytest.mark.parametrize("replacement_at", ["container-query", "remove-dispatch"])
def test_pulled_cleanup_never_removes_a_mutable_replacement(
    docker, monkeypatch, replacement_at
):
    """Guard fbc4785f: a checked reference can change before removal."""
    ref = "example/task:latest"
    _pulled(docker, ref)
    record = _pulls()[0]
    run = docker.run
    replacement = []

    async def replacing(*args, **kwargs):
        if (replacement_at == "container-query" and args[:1] == ("ps",)) or (
            replacement_at == "remove-dispatch" and args[:2] == ("image", "rm")
        ):
            replacement.append(docker.add_image(ref).id)
        return await run(*args, **kwargs)

    monkeypatch.setattr(docker, "run", replacing)
    removed, kept = asyncio.run(
        env_layers.remove_pulled(
            DockerLayers(),
            [env_layers.PulledEntry(record=record)],
        )
    )
    assert ref in docker.tags, "removed the unowned replacement reference"
    assert not removed and ref in kept
    assert not [c for c in docker.calls if c[:2] == ["image", "rm"]]
    if replacement:
        assert docker.tags[ref].id == replacement[-1]
    assert record in _pulls()


def test_forgetting_an_old_pull_keeps_a_new_record_of_the_same_image(docker):
    """Guard 9bcc62da: retire the exact receipt, preserving a later pull."""
    ref = "example/task:latest"
    image_id = _pulled(docker, ref)
    old = _pulls()[0]
    _pull_provenance.record_pull(
        "fake-daemon-1",
        ref,
        image_id,
        DEAD_OWNER,
        "a-new-tag-time",
    )
    newer = _pulls()[0]
    _pull_provenance.forget_pulls("fake-daemon-1", [old])
    assert _pulls() == [newer]


def test_pulled_images_are_only_listed_when_named(docker):
    _pulled(docker, "ghcr.io/example/task:1")
    payload = _json(_bench("list", "-s", "docker", "--json"))
    assert "ghcr.io/example/task:1" not in _names(payload)
    payload = _json(_bench("list", "-s", "docker", "--only", "pulled-images", "--json"))
    assert "ghcr.io/example/task:1" in _names(payload)


def test_only_gone_pull_receipts_are_retired(docker):
    """Guard fbc4785f: existing refs stay even when their receipts match."""
    _pulled(docker, "ghcr.io/example/owned:1")
    # Preexisting: no record. Retagged: the ref names another image now.
    docker.add_image("ghcr.io/example/preexisting:1")
    _pulled(docker, "ghcr.io/example/retagged:1")
    docker.add_image("ghcr.io/example/retagged:1")
    # Also tagged by a user; pulled by a live run; owner unknown; in use.
    _pulled(docker, "ghcr.io/example/shared:1")
    docker.tags["mine:latest"] = docker.tags["ghcr.io/example/shared:1"]
    _pulled(docker, "ghcr.io/example/live:1", owner=alias_owner_tag())
    _pulled(docker, "ghcr.io/example/unknown:1", owner="not-an-owner")
    used = _pulled(docker, "ghcr.io/example/used:1")
    docker.used_ids.add(used)
    # Gone already: its record is dropped.
    if _pull_provenance is not None:
        _pull_provenance.record_pull(
            "fake-daemon-1", "ghcr.io/example/gone:1", "sha256:" + "f" * 64, DEAD_OWNER
        )
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert "ghcr.io/example/owned:1" in docker.tags
    assert "cannot atomically remove" in result.output
    for kept in (
        "ghcr.io/example/preexisting:1",
        "ghcr.io/example/retagged:1",
        "ghcr.io/example/shared:1",
        "ghcr.io/example/live:1",
        "ghcr.io/example/unknown:1",
        "ghcr.io/example/used:1",
    ):
        assert kept in docker.tags, kept
    rm = [c for c in docker.calls if c[:2] == ["image", "rm"]]
    assert rm == []
    left = {r.ref for r in _pulls()}
    assert "ghcr.io/example/owned:1" in left
    assert "ghcr.io/example/gone:1" not in left
    assert "ghcr.io/example/live:1" in left


def _pulled_again(docker: InventoryDocker, ref: str, image: Any) -> None:
    """`docker pull ref` of the same image id, later: a newer tag time."""
    metadata = {"LastTagTime": "2026-10-17T09:00:00.000000001Z"}
    docker.tags[ref] = dataclasses.replace(
        image, info=dict(image.info, Metadata=metadata)
    )


def test_a_ref_removed_and_pulled_again_by_the_user_is_kept(docker, monkeypatch):
    """BenchFlow pulled it; then `docker system prune -a` removed it and the
    user pulled the same tag (the same image id) for their own work. The old
    record made it look like BenchFlow's, so it was removed."""
    ref = "ghcr.io/example/task:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    assert {r.ref for r in _pulls()} == {ref}
    image = docker.tags.pop(ref)  # docker system prune -a
    _pulled_again(docker, ref, image)  # the user's own docker pull
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert not [c for c in docker.calls if c[:2] == ["image", "rm"]]
    assert "pulled or tagged again" in result.output


def test_even_a_matching_pull_receipt_needs_atomic_deletion(docker, monkeypatch):
    """Guard fbc4785f: valid provenance cannot make tag removal atomic."""
    ref = "ghcr.io/example/task:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert "cannot atomically remove" in result.output
    assert len(_pulls()) == 1


def test_a_pull_the_daemon_gave_no_tag_time_is_kept(docker, monkeypatch):
    """Docker's classic image store leaves a pulled image at the zero time."""
    ref = "ghcr.io/example/classic:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    image = docker.add_image(ref)
    image.info["Metadata"] = {"LastTagTime": "0001-01-01T00:00:00Z"}
    asyncio.run(DockerLayers().note_pulled(ref))
    assert [r.tag_time for r in _pulls()] == [""]
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert "did not say" in result.output


def test_classic_store_pull_tagged_by_a_layer_build_is_kept(docker, monkeypatch):
    """Docker's classic image store keeps one tag time per image: the layer
    build's pin (`docker tag`) sets it, untagging leaves it, and a later
    `docker pull` of the same ref by the user does not change it. The note
    after `compose up` recorded the pin's time, so the user's image looked
    like BenchFlow's and was removed (U13-R2-A-1)."""
    docker.store = "classic"
    ref = "ghcr.io/example/classic:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    layers = DockerLayers()
    base = asyncio.run(layers.inspect_base(ref, pull=True))

    async def pin_and_untag() -> None:
        pin = await layers._pin_base(base, KEY_A, AGENT)
        assert (await layers._docker("image", "rm", pin)).code == 0

    asyncio.run(pin_and_untag())
    assert env_layers.last_tag_time(docker.tags[ref].info), "the pin's time stays"
    asyncio.run(layers.note_pulled(ref))
    assert [r.tag_time for r in _pulls()] == [""]
    # The user's own `docker pull ref` now: "Image is up to date", no change.
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert not [c for c in docker.calls if c[:2] == ["image", "rm"] and c[2] == ref]
    assert "did not say" in result.output


def test_a_daemon_that_cannot_name_its_image_store_gets_no_tag_time(
    docker, monkeypatch
):
    ref = "ghcr.io/example/unknown-store:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    real_run = docker.run

    async def run(*args, stdin=None, timeout=0):
        if list(args[:1]) == ["info"] and "{{json .DriverStatus}}" in args:
            return env_layers._Run(stdout="", stderr="500 busy", code=1)
        return await real_run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(docker, "run", run)
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    assert [r.tag_time for r in _pulls()] == [""]


def test_a_late_note_keeps_the_record_taken_just_after_the_pull(docker, monkeypatch):
    """Guard 747b3f50/fbc4785f: preserve the pull receipt and existing ref."""
    """The containerd store's inspect shows the newest time of all the
    image's names. The note after `compose up` ran while a layer pin named
    the image and recorded the pin's time; once the pin was gone the image
    never matched again, so BenchFlow's own pull was never cleaned
    (U13-R2-A-2)."""
    from benchflow.sandbox.docker import DockerSandbox

    ref = "ghcr.io/example/task:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    # RD-CONF-1/0a27b49d centralizes client construction in the provider factory.
    sandbox = SimpleNamespace(
        _docker_client_env=lambda: None, _docker_layers=lambda: DockerLayers()
    )
    started = datetime.now(UTC)  # DockerSandbox.start, before the layer build
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    image = docker.tags[ref]
    own = image.info["Metadata"]["LastTagTime"]
    image.info["Metadata"] = {"LastTagTime": "2026-10-16T12:00:00.000000001Z"}
    note = DockerSandbox._note_task_image_pull
    asyncio.run(note(sandbox, ref, started))  # type: ignore[arg-type]
    image.info["Metadata"] = {"LastTagTime": own}  # the pin is gone
    assert [r.tag_time for r in _pulls()] == [own]
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert "cannot atomically remove" in result.output


def test_a_late_note_replaces_a_record_from_before_this_start(docker, monkeypatch):
    ref = "ghcr.io/example/task:1"
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    layers = DockerLayers()
    asyncio.run(layers.inspect_base(ref, pull=True))
    image = docker.tags.pop(ref)  # removed outside bench cache
    started = datetime.now(UTC) + timedelta(seconds=1)
    _pulled_again(docker, ref, image)  # this start's `compose up` pulls it
    asyncio.run(layers.note_pulled(ref, unless_recorded_since=started))
    assert [r.tag_time for r in _pulls()] == ["2026-10-17T09:00:00.000000001Z"]


def test_an_inspect_that_fails_at_clean_time_keeps_the_record(docker, monkeypatch):
    """Only Docker's "No such image" means gone: a busy daemon's 500 during
    clean dropped the record, and the image was never offered again
    (U13-R2-A-3, U13-R2-B-1)."""
    ref = "ghcr.io/example/task:1"
    _pulled(docker, ref)
    real_run = docker.run

    async def run(*args, stdin=None, timeout=0):
        if list(args) == ["image", "inspect", ref]:
            return env_layers._Run(
                stdout="", stderr="Error response from daemon: 500 busy", code=1
            )
        return await real_run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(docker, "run", run)
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert [r.ref for r in _pulls()] == [ref]
    assert "could not be asked" in result.output
    assert "no longer on the daemon" not in result.output


def test_a_later_pull_by_benchflow_replaces_the_record(docker, monkeypatch):
    """A dead run's record must not outlive the image and then speak for a
    live run's pull of the same image id (U13-R1-A-4)."""
    ref = "ghcr.io/example/task:1"
    live = alias_owner_tag()
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: DEAD_OWNER)
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    image = docker.tags.pop(ref)  # removed outside bench cache
    monkeypatch.setattr(env_layers, "alias_owner_tag", lambda: live)
    monkeypatch.setattr(
        docker, "add_image", lambda tag, **_: _pulled_again(docker, tag, image)
    )
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    assert [r.owner for r in _pulls()] == [live]
    result = _bench("clean", "-s", "docker", "--only", "pulled-images", "--remove")
    assert result.exit_code == 0, result.output
    assert ref in docker.tags
    assert "still going" in result.output


def test_an_inspect_that_failed_for_another_reason_records_no_pull(docker, monkeypatch):
    """A busy daemon's error on the first inspect is not "absent": the image
    may be the user's, and the pull after it a no-op (U13-R1-A-2)."""
    ref = "ghcr.io/example/busy:1"
    docker.add_image(ref)  # the user's, already there
    real_run = docker.run
    failed: list[str] = []

    async def run(*args, stdin=None, timeout=0):
        if list(args) == ["image", "inspect", ref] and not failed:
            failed.append(ref)
            return env_layers._Run(
                stdout="", stderr="Error response from daemon: 500 busy", code=1
            )
        return await real_run(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(docker, "run", run)
    asyncio.run(DockerLayers().inspect_base(ref, pull=True))
    assert failed == [ref]
    assert _pulls() == []


def test_the_suite_keeps_pull_records_out_of_the_home_folder(tmp_path):
    """tests/conftest.py points every test's pull records at its tmp_path
    (U13-R1-B-5): a test that pulls must not leave a record a developer's
    later `clean --only pulled-images --remove` acts on."""
    assert _pull_provenance is not None, "no pull records in this BenchFlow"
    assert _pull_provenance.record_dir().is_relative_to(tmp_path)


async def test_a_pull_by_benchflow_is_recorded_and_a_present_image_is_not(docker):
    layers = DockerLayers()
    docker.add_image("ghcr.io/example/there:1")
    await layers.inspect_base("ghcr.io/example/there:1", pull=True)
    await layers.inspect_base("ghcr.io/example/absent:1", pull=True)
    refs = {r.ref for r in _pulls()}
    assert refs == {"ghcr.io/example/absent:1"}


# -- Daytona content snapshots -----------------------------------------------


class FakeSnapshots:
    def __init__(self, names: list[str]) -> None:
        self.items = {
            n: SimpleNamespace(
                id=f"id-{n}",
                name=n,
                state=SimpleNamespace(value="active"),
                size=1.5,
                created_at="2026-10-01T00:00:00Z",
            )
            for n in names
        }
        self.deleted: list[str] = []

    def list(self, page=1, limit=100):
        return SimpleNamespace(items=list(self.items.values()), total_pages=1)

    def get(self, name):
        return self.items[name]

    def delete(self, snap):
        self.deleted.append(snap.name)
        self.items.pop(snap.name, None)


def _owner_prefix(owner: str) -> str:
    import hashlib

    return f"bf-img2-{hashlib.sha256(owner.encode()).hexdigest()[:32]}-"


@pytest.fixture
def daytona(monkeypatch):
    cache_cli = _optional("benchflow.cli.cache")
    mine = _owner_prefix("py8-lock")
    names = [
        mine + "0123456789abcdef",
        mine + "fedcba9876543210",
        _owner_prefix("py8-lockx") + "0123456789abcdef",  # another owner
        mine + "0123456789abcdef0",  # malformed: too long
        mine[:-1] + "0123456789abcdef",  # near-prefix
        "bf-snap2-" + mine.split("-")[-2] + "-trial-0123456789ab",
        "bf-img-py8-lock-cada9a-0123456789abcdef",  # ambiguous legacy identity
        "bf-snap-py8-lock-cada9a-trial-0123456789ab",  # legacy branch
    ]
    snapshots = FakeSnapshots(names)
    client = SimpleNamespace(snapshot=snapshots)
    if cache_cli is not None:
        monkeypatch.setattr(cache_cli, "_daytona_client", lambda: client)
    return snapshots, names


def test_daytona_without_an_owner_lists_none_and_says_why(docker, daytona):
    result = _bench("list", "-s", "daytona")
    assert result.exit_code == 0, result.output
    assert "BENCHFLOW_DAYTONA_OWNER" in result.output
    assert "bf-img" not in result.output


def test_daytona_lists_and_removes_only_this_owners_content_snapshots(
    docker, daytona, monkeypatch
):
    """Guards FF-sw-daytona-C2: keep foreign and legacy snapshots untouched."""
    snapshots, names = daytona
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "py8-lock")
    payload = _json(_bench("list", "-s", "daytona", "--json"))
    assert sorted(_names(payload)) == sorted(names[:2])
    result = _bench("clean", "-s", "daytona", "--only", "images", "--remove")
    assert result.exit_code == 0, result.output
    assert sorted(snapshots.deleted) == sorted(names[:2])
    assert not any(n.startswith("bf-snap") for n in snapshots.deleted)


def test_sandbox_reaper_never_selects_a_content_snapshot(monkeypatch):
    from benchflow.sandbox.daytona_reaper import reap_stale_snapshots

    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "py8-lock")
    name = _owner_prefix("py8-lock") + "0123456789abcdef"
    snapshots = FakeSnapshots([name])
    snapshots.items[name].created_at = "2020-01-01T00:00:00Z"
    counts = reap_stale_snapshots(
        SimpleNamespace(snapshot=snapshots), max_age_minutes=0
    )
    assert counts["deleted"] == 0 and snapshots.deleted == []


async def test_a_prebuilt_task_image_compose_pulls_is_recorded_once_absent(docker):
    """DockerSandbox notes the prebuilt image as BenchFlow's pull only when it
    was absent before start; a present one is never recorded."""
    from benchflow.sandbox.docker import DockerSandbox

    absent_probe = getattr(DockerSandbox, "_absent_task_image", None)
    assert absent_probe is not None, "no pull-provenance hook in this BenchFlow"

    def sandbox(ref: str) -> SimpleNamespace:
        return SimpleNamespace(
            task_env_config=SimpleNamespace(docker_image=ref),
            _use_prebuilt=True,
            _recovery_baseline=None,
            _docker_client_env=lambda: None,
            _docker_layers=lambda: DockerLayers(),
        )

    docker.add_image("ghcr.io/example/present:1")
    present = sandbox("ghcr.io/example/present:1")
    assert await DockerSandbox._absent_task_image(present) is None  # type: ignore[arg-type]
    fresh = sandbox("ghcr.io/example/compose-pulled:1")
    ref = await DockerSandbox._absent_task_image(fresh)  # type: ignore[arg-type]
    assert ref == "ghcr.io/example/compose-pulled:1"
    docker.add_image(ref)  # what `compose up` pulled
    await DockerSandbox._note_task_image_pull(fresh, ref)  # type: ignore[arg-type]
    await DockerSandbox._note_task_image_pull(fresh, None)  # type: ignore[arg-type]
    assert {r.ref for r in _pulls()} == {"ghcr.io/example/compose-pulled:1"}
