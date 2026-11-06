"""Guards C7, an incomplete sibling of the C6 fix in 67f9c75d."""

import copy

import pytest

from tests.test_compose_image_consumers import BACKENDS, start_plan


@pytest.fixture(autouse=True)
def _without_bfx(monkeypatch):
    # These check the task's own mounts and Compose files; bfx's read-only
    # overlay (present when the pinned binaries are built) is checked in
    # tests/test_bfx_*.py.
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")


SPELLINGS = [
    ("task", "task:latest"),
    ("task:latest", "task"),
    ("task:v1", "docker.io/library/task:v1"),
    ("docker.io/task:v1", "library/task:v1"),
    ("index.docker.io/library/task:v1", "task:v1"),
    ("team/task", "docker.io/team/task:latest"),
    ("localhost:5000/task:v1", "localhost:5000/task:v1"),
    ("registry.invalid/team/task", "registry.invalid/team/task:latest"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("producer", ["main", "worker"])
@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("additional", [False, True])
@pytest.mark.parametrize("source,consumer", SPELLINGS)
async def test_image_mounts_follow_private_outputs(
    tmp_path, monkeypatch, backend, producer, prebuilt, additional, source, consumer
):
    if prebuilt:
        producer = "worker"
    mounts = [
        {
            "type": "image",
            "source": consumer,
            "target": "/baseline",
            "read_only": True,
            "image": {"subpath": "data"},
        },
        {"type": "volume", "source": "answers", "target": "/answers"},
        {"type": "bind", "source": "/tmp/fixture", "target": "/fixture"},
        {"type": "image", "source": consumer, "target": "/second"},
        {"type": "image", "source": "external:v1", "target": "/external"},
    ]
    document = {
        "services": {
            "main": {"image": "trusted:v1", "volumes": copy.deepcopy(mounts)},
            "grader": {"image": "grader:v1", "volumes": copy.deepcopy(mounts)},
        }
    }
    document["services"].setdefault(producer, {})
    document["services"][producer].update(
        image="different:primary" if additional else source,
        build={"context": ".", "tags": [source] if additional else []},
    )
    result = await start_plan(tmp_path, monkeypatch, backend, document, prebuilt)
    for service in ("main", "grader"):
        expected = copy.deepcopy(mounts)
        for index in (0, 3):
            expected[index]["source"] = result[producer]["image"]
        assert result[service]["volumes"] == expected
    assert document["services"]["grader"]["volumes"] == mounts
    if prebuilt:
        assert result["main"]["image"] == "trusted:v1"
        assert not result["main"].get("build")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("consumer_service", ["main", "worker", "grader"])
@pytest.mark.parametrize("additional", [False, True])
async def test_ambiguous_image_mount_refuses_before_build(
    tmp_path, monkeypatch, backend, consumer_service, additional
):
    document = {
        "services": {
            "main": {"image": "task:v1", "build": "."},
            "worker": {
                "image": "other:v1" if additional else "docker.io/library/task:v1",
                "build": {"context": ".", "tags": ["task:v1"] if additional else []},
            },
            "grader": {"image": "grader:v1"},
        }
    }
    document["services"][consumer_service]["volumes"] = [
        {
            "type": "image",
            "source": "index.docker.io/library/task:v1",
            "target": "/baseline",
        }
    ]
    with pytest.raises(RuntimeError, match="Could not isolate Compose build"):
        await start_plan(tmp_path, monkeypatch, backend, document)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_other_mount_types_and_image_identities_stay_distinct(
    tmp_path, monkeypatch, backend
):
    mounts = [
        {"type": "image", "source": source, "target": f"/baseline/{index}"}
        for index, source in enumerate(
            [
                "task:v2",
                "team/task:v1",
                "registry.invalid/task:v1",
                "task@sha256:" + "a" * 64,
            ]
        )
    ] + [
        {"type": kind, "source": "task:v1", "target": f"/{kind}"}
        for kind in ("volume", "bind")
    ]
    document = {
        "services": {
            "main": {"image": "task:v1", "build": "."},
            "grader": {"image": "grader:v1", "volumes": mounts},
        }
    }
    result = await start_plan(tmp_path, monkeypatch, backend, document)
    assert result["grader"]["volumes"] == mounts
