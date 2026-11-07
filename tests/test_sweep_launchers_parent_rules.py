"""Guards the Docker parent-rule regression at 04d3d5f9."""

import pytest

from tests.test_sweep_launchers_ignore import (
    test_docker_context_membership as check_membership,
)


@pytest.mark.parametrize("backend", ["digest", "local", "codebuild"])
@pytest.mark.parametrize(
    "pattern,path,ignored",
    [
        ("config/private\n!config", "config/private/token", True),
        ("config/private\n!config/", "config/private/token", True),
        ("/config/private/\n!/config/", "config/private/token", True),
        ("**/private\n!config", "config/private/token", True),
        ("config/pr?vate\n!config", "config/private/token", True),
        ("config/[ps]*\n!config", "config/private/token", True),
        ("config/private\n!con*", "config/private/token", True),
        ("config/private\n!**/config", "config/private/token", True),
        ("a/b/c\n!a/b\n!a", "a/b/c/token", True),
        ("config/private\n!config", "config/public/token", False),
        ("config/private\n!config/private", "config/private/token", False),
        ("config\n!config\nconfig/private", "config/private/token", True),
        ("config\nconfig/private\n!config", "config/private/token", False),
        (
            "config/private\n!config\n!config/private/token",
            "config/private/token",
            False,
        ),
        ("config\n!unrelated/*", "config/private/token", True),
    ],
)
def test_parent_state_matches_docker(tmp_path, backend, pattern, path, ignored):
    check_membership(tmp_path, pattern, path, ignored, backend)
