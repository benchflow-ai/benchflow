"""Guards FF-sw-launchers Docker matching against the Git semantics at 363d815b."""

import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox.agentcore_builder import (
    BuildRequest,
    CodeBuildBuilder,
    materialized,
)


@pytest.mark.parametrize(
    "pattern,path,ignored",
    [
        ("data", "sub/data/input.txt", False),
        ("data/", "sub/data/input.txt", False),
        ("/data/", "sub/data/input.txt", False),
        ("*.txt", "sub/input.txt", False),
        ("data", "data/input.txt", True),
        ("**/data", "sub/data/input.txt", True),
        ("**/data", "data/input.txt", True),
        ("a/**/b", "a/b/input.txt", True),
        ("a/**/b", "a/x/y/b/input.txt", True),
        ("ab**cd", "ab/x/cd", True),
        ("a?c", "abc/file", True),
        ("a?c", "a/c/file", False),
        ("[^a].txt", "b.txt", True),
        ("[^a].txt", "a.txt", False),
        ("[!a].txt", "b.txt", False),
        ("[!a].txt", "!.txt", True),
        (r"\!keep", "!keep/file", True),
        ("\ufeffdata", "data/file", True),
        (" #data", "#data/file", True),
        ("data\n!data/keep", "data/keep/file", False),
        ("**\n!sub/data", "sub/data/input.txt", False),
        ("**\n!data", "sub/data/input.txt", True),
    ],
)
@pytest.mark.parametrize("backend", ["digest", "local", "codebuild"])
def test_docker_context_membership(tmp_path, pattern, path, ignored, backend):
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("marker")
    (tmp_path / ".dockerignore").write_text(pattern + "\n")
    request = BuildRequest(
        context_dir=tmp_path,
        dockerfile_text="FROM scratch",
        shim_text="pass",
        image_uri="repo:test",
        registry="repo",
        region="us-east-1",
        force_build=False,
        timeout_sec=60,
    )
    if backend == "digest":
        names = {name for _, name in provisioning.iter_context_files(tmp_path)}
    elif backend == "local":
        with materialized(request) as dockerfile:
            names = {
                p.relative_to(dockerfile.parent).as_posix()
                for p in dockerfile.parent.rglob("*")
            }
    else:
        blob = CodeBuildBuilder(MagicMock(), "1", "us-east-1")._package(request)
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            names = set(archive.namelist())
    assert (path not in names) == ignored
