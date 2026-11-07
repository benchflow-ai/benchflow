"""The pinned SDK's upload cache must preserve the whole captured manifest."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")

from daytona._async.object_storage import AsyncObjectStorage
from daytona._async.snapshot import AsyncSnapshotService

from benchflow.sandbox import daytona_content_snapshots as content
from benchflow.sandbox.daytona_layers import InstallLayerPlan
from tests.test_daytona_content_snapshots import (
    PINNED_BASE,
    Client,
    Snapshots,
    _record,
    _start,
    _task,
)
from tests.test_daytona_content_snapshots import server as snapshot_server


@pytest.fixture
def server(monkeypatch):
    return snapshot_server.__wrapped__(monkeypatch)


@pytest.mark.parametrize("change", ["file-mode", "directory-mode", "boundaries"])
async def test_sdk_upload_cache_keeps_every_manifest_distinct(
    monkeypatch, server, tmp_path, change
):
    """Guard abb528a3/U13-CONFIRM-1: cached SDK tars match snapshot inputs."""
    directory = change != "file-mode"
    copied = "seed" if directory else "run.sh"
    source = "seed/a" if directory else "run.sh"
    dockerfile = _task(
        tmp_path / "task",
        f"FROM {PINNED_BASE}\nCOPY {copied} /task/seed\n",
        {source: "bc"},
    )
    original = dockerfile.parent / copied
    original.chmod(0o755 if directory else 0o644)
    # Exercise the real SDK's process_image_context -> upload -> cache key.
    # Only object storage I/O is replaced: this never contacts a service.
    storage = object.__new__(AsyncObjectStorage)
    cached = {}
    uploaded = []

    async def exists(key):
        return key in cached

    async def upload(key, source_path, archive_path):
        entries, reason = content.context_entries([(source_path, archive_path)])
        assert reason is None
        cached[key] = entries

    monkeypatch.setattr(storage, "_file_exists_in_s3", exists)
    monkeypatch.setattr(storage, "_upload_as_tar", upload)
    # SDK 0.207 passes region= and reads the push credentials' region.
    monkeypatch.setattr(
        "daytona._async.snapshot.AsyncObjectStorage", lambda *a, **k: storage
    )
    api = SimpleNamespace(
        get_push_access=AsyncMock(
            return_value=SimpleNamespace(
                storage_url="dummy",
                access_key="dummy",
                secret="dummy",
                session_token="dummy",
                bucket="dummy",
                organization_id="dummy",
                region="dummy",
            )
        )
    )
    create = Snapshots.create

    async def cached_create(self, params, timeout=None):
        copy_line = next(
            line
            for line in params.image.dockerfile().splitlines()
            if line.startswith("COPY ")
        )
        if copy_line.startswith("COPY ["):
            operands = json.loads(copy_line.removeprefix("COPY "))
            assert operands == [
                params.image._context_list[0].archive_path,
                "/task/seed",
            ]
        expected, reason = content.context_entries(
            [(c.source_path, c.archive_path) for c in params.image._context_list]
        )
        assert reason is None
        keys = await AsyncSnapshotService.process_image_context(api, params.image)
        actual = [entry for key in keys for entry in cached[f"dummy/{key}/context.tar"]]
        uploaded.append((keys, actual, expected))
        return await create(self, params, timeout)

    monkeypatch.setattr(Snapshots, "create", cached_create)
    first = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial-1",
        image=None,
        dockerfile=dockerfile,
    )
    if change == "boundaries":
        (original / "a").unlink()
        (original / "ab").write_text("c")
    else:
        original.chmod(0o700 if directory else 0o755)
    second = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial-2",
        image=None,
        dockerfile=dockerfile,
    )
    assert _record(first)["snapshot_name"] != _record(second)["snapshot_name"]
    assert len(uploaded) == 2
    assert uploaded[0][1] == uploaded[0][2]
    assert uploaded[1][1] == uploaded[1][2], "SDK reused a tar with different inputs"
    assert uploaded[0][0] != uploaded[1][0]
    assert content.IDENTITY_FORMAT >= 3


@pytest.mark.parametrize("syntax", ["shell", "json"])
async def test_capture_keeps_copy_flags_operands_and_stable_upload_keys(
    tmp_path, syntax
):
    """Guard abb528a3/U13-CONFIRM-1: namespacing preserves COPY semantics."""
    from daytona import Image
    from daytona.common.image import Context

    image = Image.base(PINNED_BASE)
    image._dockerfile = f"FROM {PINNED_BASE} AS base\nFROM {PINNED_BASE}\n"
    names = ["one", "two"] if syntax == "shell" else ["seed one", 'seed"two']
    for index, name in enumerate(names):
        path = tmp_path / str(index)
        path.write_text(f"dummy-{index}")
        path.chmod(0o640)
        image._context_list.append(Context(source_path=str(path), archive_path=name))
    operands = (
        " ".join([*names, "/target/"])
        if syntax == "shell"
        else json.dumps([*names, "/target/"])
    )
    image._dockerfile += (
        f"COPY --chown=123:456 --chmod=0755 {operands}\nCOPY --from=base /from /to\n"
    )
    original = image.dockerfile()
    keys = []
    storage = object.__new__(AsyncObjectStorage)
    for _ in range(2):
        with content.captured_image(image) as captured:
            lines = captured.dockerfile().splitlines()
            assert lines[-1] == "COPY --from=base /from /to"
            prefix = "COPY --chown=123:456 --chmod=0755 "
            copied = json.loads(lines[-2].removeprefix(prefix))
            assert copied == [
                *(c.archive_path for c in captured._context_list),
                "/target/",
            ]
            assert [
                Path(c.source_path).read_text() for c in captured._context_list
            ] == ["dummy-0", "dummy-1"]
            keys.append(
                [
                    await storage._compute_hash_for_path_md5(
                        c.source_path, c.archive_path
                    )
                    for c in captured._context_list
                ]
            )
        assert image.dockerfile() == original
        assert [c.archive_path for c in image._context_list] == names
    assert keys[0] == keys[1]


@pytest.mark.parametrize(
    "copy", ["COPY --parents seed /target", "COPY $SOURCE /target"]
)
def test_namespace_declines_copy_semantics_it_cannot_preserve(copy):
    """Guard abb528a3/U13-CONFIRM-1: never change source-dependent semantics."""
    assert content.check_recipe(f"FROM {PINNED_BASE}\n{copy}\n")[1] is not None


@pytest.mark.parametrize(
    "destination",
    [r"/out/\$DEST", r"/out/\${DEST}", r'"/out/\$DEST"', "'/out/$DEST'"],
)
async def test_escaped_copy_destination_uses_unchanged_recipe_fallback(
    monkeypatch, server, tmp_path, destination
):
    """Guard 5fd5da91/U13-CONFIRM-4: COPY quoting must retain its meaning."""
    recipe = f"FROM {PINNED_BASE}\nENV DEST=expanded\nCOPY seed {destination}\n"
    dockerfile = _task(tmp_path / "task", recipe, {"seed": "dummy-marker"})
    client = Client(server)
    sandbox = await _start(
        monkeypatch,
        client,
        tmp_path / "trial",
        image=None,
        dockerfile=dockerfile,
        configure=False,
        _install_layers=InstallLayerPlan(fresh=False),
    )
    assert _record(sandbox)["snapshot"] == "off"
    assert "COPY" in _record(sandbox)["snapshot_reason"]
    assert not server.create_calls
    assert client.requests[0].image.dockerfile() == recipe
    context = client.requests[0].image._context_list[0]
    assert context.archive_path == "seed"
    assert Path(context.source_path).read_text() == "dummy-marker"


async def test_plain_copy_destination_still_supports_content_snapshots(
    monkeypatch, server, tmp_path
):
    """Guard 5fd5da91/U13-CONFIRM-4: supported operands still snapshot."""
    dockerfile = _task(
        tmp_path / "task",
        f"FROM {PINNED_BASE}\nENV DEST=expanded\nCOPY seed /out/$DEST\n",
        {"seed": "dummy-marker"},
    )
    sandbox = await _start(
        monkeypatch,
        Client(server),
        tmp_path / "trial",
        image=None,
        dockerfile=dockerfile,
        configure=False,
        _install_layers=InstallLayerPlan(fresh=False),
    )
    assert _record(sandbox)["snapshot"] == "created"
    copy = server.create_calls[0].image.dockerfile().splitlines()[-1]
    assert json.loads(copy.removeprefix("COPY "))[-1] == "/out/$DEST"


@pytest.mark.parametrize("destination", [r"/out/\$DEST", "/out/'quoted'", r"/out/a\b"])
def test_json_copy_destination_keeps_its_escaped_string(tmp_path, destination):
    """Guard 5fd5da91/U13-CONFIRM-4: JSON operands survive capture intact."""
    from daytona import Image
    from daytona.common.image import Context

    source = tmp_path / "seed"
    source.write_text("dummy-marker")
    image = Image.base(PINNED_BASE)
    image._dockerfile += f"COPY {json.dumps(['seed', destination])}\n"
    image._context_list.append(Context(source_path=str(source), archive_path="seed"))
    original = image.dockerfile()
    with content.captured_image(image) as captured:
        assert content.check_recipe(captured.dockerfile())[1] is None
        copy = captured.dockerfile().splitlines()[-1]
        assert json.loads(copy.removeprefix("COPY ")) == [
            captured._context_list[0].archive_path,
            destination,
        ]
    assert image.dockerfile() == original
