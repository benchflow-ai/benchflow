"""The pure parts of compose teardown (sandbox/_compose_teardown.py)."""

from __future__ import annotations

import json

import pytest

from benchflow.sandbox._compose_teardown import (
    COMPOSE_KILL_COMMAND,
    RECOVERY_LEASE_REPOSITORY,
    built_image_refs,
    compose_down_command,
    image_repository,
)

LEASE = "benchflow-recovery-lease:" + "0" * 32


def test_kill_is_compose_kill():
    assert list(COMPOSE_KILL_COMMAND) == ["kill"]


def test_down_command_never_removes_images_and_never_waits():
    command = compose_down_command()
    assert command[0] == "down"
    assert "--rmi" not in command
    assert command[command.index("-t") + 1] == "0"
    assert "--volumes" not in command


@pytest.mark.parametrize(
    "ref,repository",
    [
        ("bf__hello:latest", "bf__hello"),
        ("bf__hello", "bf__hello"),
        ("registry.test:5000/team/web:1", "registry.test:5000/team/web"),
        ("web@sha256:abc", "web"),
    ],
)
def test_image_repository(ref, repository):
    assert image_repository(ref) == repository


def test_built_image_refs_picks_our_tags_only():
    """Guards U10-CONFIRM-2 after 22672877: inspect both tags and digests."""
    listing = json.dumps(
        [
            {
                "Id": "sha256:a",
                "RepoTags": [LEASE, "bf__hello:latest"],
                "RepoDigests": [],
            },
            {
                "Id": "sha256:b",
                "RepoTags": ["p__1-api:latest", "operator/kept:1"],
                "RepoDigests": [],
            },
            {"Id": "sha256:c", "RepoTags": ["example.test/web:1"], "RepoDigests": []},
            {"Id": "sha256:d", "RepoTags": None, "RepoDigests": None},
            {
                "Id": "sha256:e",
                "RepoTags": None,
                "RepoDigests": ["archive/hello@sha256:abc"],
            },
            {
                "Id": "sha256:f",
                "RepoTags": ["example.test/web:1"],
                "RepoDigests": ["archive/hello@sha256:def"],
            },
            {"Id": "sha256:unknown"},
            "garbage line here",
        ]
    )
    assert built_image_refs(
        listing,
        build_tags={"bf__hello:latest", "p__1-api:latest", "example.test/web:1"},
    ) == [
        "sha256:c",
        "sha256:d",
    ]


def test_a_lease_stays_even_as_a_builds_only_tag():
    listing = json.dumps(
        [
            {"Id": "sha256:a", "RepoTags": [LEASE], "RepoDigests": []},
            {
                "Id": "sha256:b",
                "RepoTags": ["bf__hello:latest", f"{LEASE[:-1]}1"],
                "RepoDigests": [],
            },
        ]
    )
    assert built_image_refs(listing, build_tags={"bf__hello:latest"}) == []


def test_the_lease_repository_is_the_one_recovery_tags_with():
    from benchflow.sandbox._recovery_baseline import DockerRecoveryBaseline

    pattern = DockerRecoveryBaseline.model_fields["lease_tag"].metadata[0].pattern
    assert pattern.startswith(f"^{RECOVERY_LEASE_REPOSITORY}:")
