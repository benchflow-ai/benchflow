"""Guards FF-sw-daytona content-input eligibility at 363d815b."""

import pytest

from benchflow.sandbox.daytona_content_snapshots import check_recipe

BASE = "FROM example/base@sha256:" + "1" * 64 + "\n"


@pytest.mark.parametrize(
    "mount",
    [
        "--mount=from=alpine:latest,target=/src",
        "--mount=type=cache,from=alpine:latest,target=/src",
        "--mount=type=cache,from=stage,target=/src",
        "--mount=type=cache,from=example/base@sha256:" + "2" * 64 + ",target=/src",
        "--mount=target=/src",
        "--mount=type=bind,target=/src",
        "--mount=type=secret,target=/src",
        "--mount=type=ssh,target=/src",
        "--mount=type=unknown,target=/src",
        "--mount=type=bind,type=cache,target=/src",
        "--mount=type=cache,type=bind,target=/src",
        "--mount='type=cache,from=alpine:latest,target=/src'",
    ],
)
@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("", ""),
        ("--mount=type=cache,target=/cache ", ""),
        ("", " --mount=type=tmpfs,target=/tmp"),
    ],
)
def test_each_mount_requires_modeled_inputs(mount, prefix, suffix):
    """Guard 363d815b: another mount cannot bless unmodeled input."""
    _, reason = check_recipe(BASE + f"RUN {prefix}{mount}{suffix} true\n")
    assert reason is not None


@pytest.mark.parametrize(
    "mount",
    [
        "--mount=type=cache,target=/cache",
        "--mount=type=tmpfs,target=/tmp",
        "--mount=type=cache,target=/cache --mount=type=tmpfs,target=/tmp",
    ],
)
def test_self_contained_mounts_remain_eligible(mount):
    """Guard 363d815b: scratch cache and tmpfs mounts remain supported."""
    assert check_recipe(BASE + f"RUN {mount} true\n")[1] is None
