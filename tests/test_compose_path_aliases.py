"""Guards S1-CONF-2 and S1-CONF-3 at 41014317: one folder, many spellings.

A writable ancestor must be found whatever spelling a launch source uses
(case, Unicode normalization or a host bind mount name one folder twice),
and a valid operator symlink root must be mapped whatever separators
Compose leaves in it.
"""

from __future__ import annotations

import json
import os
import shlex
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.task import compose_env
from benchflow.task.compose_env import (
    _operator_project_paths,
    _ProjectCheck,
    check_compose_in_vm,
)
from tests.test_compose_launch import _from_task, _launched
from tests.test_compose_mutable_sources import _project
from tests.test_compose_opaque_values import PAYLOADS
from tests.test_compose_oracle import (
    _compose_available,
    _compose_task,
    _gate,
    _refused,
)

CONSUMERS = [
    "bind",
    "secrets",
    "configs",
    "env_file",
    "label_file",
    "device",
    "context",
    "dockerfile",
    "additional_contexts",
    "oci_context",
    "cache_from",
    "cache_to",
    "ssh",
]

#: Separator spellings Compose keeps in an absolute value (compose-go
#: paths/unix.go returns absolute paths as written).
SEPARATORS = ["", "//", "/./", "///./", "/.//"]


@pytest.fixture(autouse=True)
def operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)


def _identity(path: str):
    """The host's file identity, as the check reads it."""
    try:
        found = os.stat(path)
    except OSError:
        return None
    return (found.st_dev, found.st_ino)


def _folded(root: Path, fold):
    """A file identity as a filesystem that folds names with *fold* sees it:
    every spelling of an existing entry below *root* is that entry."""

    def identity(path: str):
        if path != str(root) and not path.startswith(str(root) + "/"):
            return _identity(path)
        found = root
        for part in path[len(str(root)) :].split("/"):
            if part in ("", "."):
                continue
            names = {fold(p.name): p for p in found.iterdir()} if found.is_dir() else {}
            if fold(part) not in names:
                return None
            found = names[fold(part)]
        return _identity(str(found))

    return identity


SPELLINGS = {
    # ext4 casefold, APFS and NTFS fold case.
    "case": ("work", "WORK", str.casefold),
    "case-mixed": ("Work", "wORK", str.casefold),
    # APFS and ext4 casefold compare Unicode-normalized names.
    "unicode": (
        unicodedata.normalize("NFC", "café"),
        unicodedata.normalize("NFD", "café"),
        lambda name: unicodedata.normalize("NFC", name),
    ),
}


@pytest.mark.parametrize("writer", ["bind", "sidecar", "volume", "inherited"])
@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
@pytest.mark.parametrize("separator", ["", "//", "/./"])
def test_alias_spellings_of_a_writable_root_are_refused(
    monkeypatch, tmp_path, writer, consumer, spelling, separator
):
    """Guards S1-CONF-2 at 41014317 across every consumer and writer."""
    written, alias, fold = SPELLINGS[spelling]
    shared = tmp_path / "shared"
    (shared / written / "data").mkdir(parents=True)
    source = f"{shared}/{alias}{separator or '/'}data"
    # A string: Path() would clean the separators under test.
    project = _project(shared / written, source, writer, consumer)
    monkeypatch.setattr(
        compose_env, "_local_identity", _folded(shared, fold), raising=False
    )
    check = _ProjectCheck(str(tmp_path), (str(tmp_path),))
    assert any("writable" in r.describe() for r in check.check(project))


def test_host_bind_mount_alias_of_a_writable_root_is_refused(monkeypatch, tmp_path):
    """Guards S1-CONF-2's bind-mount sibling: /srv/b bound from /srv/a."""
    (tmp_path / "work/data").mkdir(parents=True)
    (tmp_path / "bound/data").mkdir(parents=True)
    work = _identity(str(tmp_path / "work"))

    def identity(path):
        if path == str(tmp_path / "bound"):
            return work
        return _identity(path)

    monkeypatch.setattr(compose_env, "_local_identity", identity, raising=False)
    project = _project(tmp_path / "work", tmp_path / "bound/data", "bind", "bind")
    check = _ProjectCheck(str(tmp_path), (str(tmp_path),))
    assert any("writable" in r.describe() for r in check.check(project))


@pytest.mark.parametrize("relation", ["case-distinct", "sibling", "same"])
def test_distinct_folders_stay_valid_on_a_case_sensitive_host(tmp_path, relation):
    """Guards against false refusals: identities differ, so nothing is shared."""
    (tmp_path / "work").mkdir()
    (tmp_path / "WORK/data").mkdir(parents=True)
    (tmp_path / "other/data").mkdir(parents=True)
    source = {
        "case-distinct": tmp_path / "WORK/data",
        "sibling": tmp_path / "other/data",
        "same": tmp_path / "work",
    }[relation]
    project = _project(tmp_path / "work", source, "bind", "bind")
    assert not _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)


@pytest.mark.skipif(not _compose_available(), reason="needs Compose config")
@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
@pytest.mark.parametrize("separator", ["", "//"])
def test_named_case_alias_is_refused_through_the_gate(
    monkeypatch, tmp_path, spelling, separator
):
    """Guards S1-CONF-2's input: WORK and INPUT named by the operator."""
    written, alias, fold = SPELLINGS[spelling]
    shared = tmp_path / "shared"
    (shared / written / "data").mkdir(parents=True)
    monkeypatch.setattr(
        compose_env, "_local_identity", _folded(shared, fold), raising=False
    )
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "WORK,INPUT")
    monkeypatch.setenv("WORK", f"{shared}/{written}")
    monkeypatch.setenv("INPUT", f"{shared}/{alias}{separator or '/'}data")
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n"
        '    volumes: ["${WORK}:/workspace", "${INPUT}:/data:ro"]\n',
    )
    assert "writable over the launch source" in _refused(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["alias", "short", "failed"])
async def test_vm_check_compares_identities_too(tmp_path, monkeypatch, answer):
    """Guards S1-CONF-2 on the DinD VM: the same rule, with the VM's stat."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    project = _project(
        Path("/tmp/task/work"), Path("/tmp/task/WORK/data"), "bind", "bind"
    )

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"] and "stat" in argv[2]:
            paths = argv[4:]
            if answer == "failed":
                return SimpleNamespace(return_code=1, stdout="")
            if answer == "short":
                paths = paths[1:]
            return SimpleNamespace(
                return_code=0,
                stdout="\n".join(
                    "1:7" if p.casefold() == "/tmp/task/work" else f"1:{i + 100}"
                    for i, p in enumerate(paths)
                ),
            )
        if argv[:2] == ["sh", "-c"]:
            return SimpleNamespace(return_code=0, stdout="\n".join(argv[4:]))
        return SimpleNamespace(return_code=0, stdout="x$$y")

    refused = await check_compose_in_vm(
        run, ["docker", "compose"], "/tmp/task", ["/tmp/task"]
    )
    assert refused, answer


@pytest.mark.asyncio
async def test_vm_check_without_writers_needs_no_identities(monkeypatch):
    """Guards DinD availability: identities matter only beside a writer."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    project = {
        "services": {
            "main": {
                "volumes": [
                    {
                        "type": "bind",
                        "source": "/tmp/task/data",
                        "target": "/data",
                        "read_only": True,
                    }
                ]
            }
        }
    }

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"] and "stat" in argv[2]:
            return SimpleNamespace(return_code=1, stdout="")
        if argv[:2] == ["sh", "-c"]:
            return SimpleNamespace(return_code=0, stdout="\n".join(argv[4:]))
        return SimpleNamespace(return_code=0, stdout="x$$y")

    assert not await check_compose_in_vm(
        run, ["docker", "compose"], "/tmp/task", ["/tmp/task"]
    )


def test_vm_identity_script_reads_every_path(tmp_path):
    """Guards the VM batch itself: one line per path, `-` when missing."""
    import subprocess

    (tmp_path / "a").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "a", target_is_directory=True)
    paths = [str(tmp_path / "a"), str(tmp_path / "link"), str(tmp_path / "gone")]
    out = subprocess.run(
        ["sh", "-c", compose_env._VM_IDENTITY, "identity", *paths],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    found = os.stat(tmp_path / "a")
    assert out == [f"{found.st_dev}:{found.st_ino}"] * 2 + ["-"]


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("/srv//logical", "/srv/logical"),
        ("//srv/logical", "/srv/logical"),
        ("///srv/./logical/", "/srv/logical"),
        ("/srv/logical/.", "/srv/logical"),
        ("/srv/a/../logical", "/srv/logical"),
        ("/", "/"),
        ("//", "/"),
    ],
)
def test_clean_path_matches_docker_bind_cleaning(raw, clean):
    """Guards S1-CONF-3's one rule: Go's path.Clean, as Docker cleans binds."""
    assert compose_env._clean_path(raw) == clean


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("value_sep", SEPARATORS)
@pytest.mark.parametrize("consumer_sep", SEPARATORS)
@pytest.mark.parametrize("trailing", ["", "/", "/."])
def test_operator_roots_match_whatever_separators_compose_kept(
    tmp_path, consumer, value_sep, consumer_sep, trailing
):
    """Guards S1-CONF-3 at 41014317: raw consumers with interior separators."""
    physical = tmp_path / "physical"
    (physical / "data").mkdir(parents=True)
    (tmp_path / "logical").symlink_to(physical, target_is_directory=True)
    value = f"{tmp_path}{value_sep or '/'}logical{trailing}"
    # Compose prints an absolute interpolated value as written.
    source = f"{tmp_path}{consumer_sep or '/'}logical{consumer_sep or '/'}data"
    project = _project(tmp_path / "work", Path("/placeholder"), "bind", consumer)
    if consumer == "oci_context":
        # Compose prints additional_contexts as a map, including list input.
        project["services"]["main"]["build"]["additional_contexts"] = {
            "data": "oci-layout:///placeholder"
        }
    text = json.dumps(project).replace("/placeholder", source)
    project = json.loads(text)
    mapped = _operator_project_paths(project, {"S1_PATH"}, {"S1_PATH": value})
    paths = _ProjectCheck(str(tmp_path), (str(tmp_path),)).paths(mapped)
    assert str(physical / "data") in paths
    assert not any("logical" in path for path in paths)


@pytest.mark.parametrize("escape", ["/../outside", "/data/../../outside"])
def test_descendants_that_leave_an_operator_root_are_not_mapped(tmp_path, escape):
    """Guards S1-CONF-3's fix: only the authorized root is replaced."""
    physical = tmp_path / "physical"
    physical.mkdir()
    (tmp_path / "logical").symlink_to(physical, target_is_directory=True)
    value = f"{tmp_path}//logical"
    project = _project(tmp_path / "work", value + escape, "bind", "bind")
    mapped = _operator_project_paths(project, {"S1_PATH"}, {"S1_PATH": value})
    source = mapped["services"]["main"]["volumes"][1]["source"]
    assert source == value + escape


@pytest.mark.skipif(not _compose_available(), reason="needs Compose config")
@pytest.mark.parametrize("syntax", ["short", "long", "device"])
@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize("suffix", ["", "//data", "/./data"])
def test_named_symlink_roots_with_separators_reach_frozen_project(
    monkeypatch, tmp_path, syntax, separator, suffix
):
    """Guards S1-CONF-3's input: S1_PATH=/srv//logical, a writable log mount."""
    physical = tmp_path / "physical"
    (physical / "data").mkdir(parents=True)
    (tmp_path / "logical").symlink_to(physical, target_is_directory=True)
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "S1_PATH")
    monkeypatch.setenv("S1_PATH", f"{tmp_path}{separator or '/'}logical")
    source = "${S1_PATH}" + suffix
    mounts = {
        "short": f'    volumes: ["{source}:/data:ro"]\n',
        "long": f'    volumes: [{{type: bind, source: "{source}", target: /data, read_only: true}}]\n',
        "device": '    volumes: ["data:/data:ro"]\n'
        f'volumes:\n  data:\n    driver_opts: {{type: none, o: bind, device: "{source}"}}\n',
    }
    task = _compose_task(tmp_path / "task", "services:\n  main:\n" + mounts[syntax])
    _gate(task)
    project = _launched(_from_task(task).inner)
    expected = str(physical / "data") if suffix else str(physical)
    if syntax == "device":
        assert project["volumes"]["data"]["driver_opts"]["device"] == expected
    else:
        assert (
            next(
                m["source"]
                for m in project["services"]["main"]["volumes"]
                if m["target"] == "/data"
            )
            == expected
        )


@pytest.mark.parametrize("payload", PAYLOADS)
def test_cleaning_never_touches_content(tmp_path, payload):
    """Guards S1-CONF-1 beside the S1-CONF-3 fix: only path fields change."""
    project = {
        "services": {
            "main": {
                "environment": {"TEXT": payload},
                "command": [payload],
                "labels": {"text": payload},
            }
        },
        "configs": {"text": {"content": payload}},
        "secrets": {"text": {"environment": "S1_DATA"}},
    }
    mapped = _operator_project_paths(project, {"S1_DATA"}, {"S1_DATA": payload})
    assert mapped == project
