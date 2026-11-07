"""Two jobs that build the same task at once retry the losing build.

Every main image is tagged ``bf__<task>``, and ``pull_policy: build`` makes
``compose up`` build as well, so two jobs of one task on one daemon race on
BuildKit's export. Seen on 2026-10-13 with robohub's metaworld-button-press:
the second job failed with ``#11 ERROR: image
"docker.io/library/bf__metaworld-button-press:latest": already exists`` and its
trial was scored as an infrastructure error.
"""

from benchflow.sandbox import docker as d

RACE = (
    "Docker compose command failed for environment metaworld-button-press. ... "
    '#11 ERROR: image "docker.io/library/bf__metaworld-button-press:latest": already exists'
)


def test_the_export_race_is_a_retryable_build_error():
    assert d._is_retryable_docker_build_error(RACE)


def test_the_export_race_is_retried_by_compose_up():
    assert d._IMAGE_EXPORT_RACE.search(RACE)


def test_an_ordinary_build_failure_is_not_retried():
    assert not d._is_retryable_docker_build_error(
        '#7 ERROR: process "/bin/sh -c pip install nope" did not complete successfully: exit code: 1'
    )
