"""Guards CONF3-rollout-1 at 2e65151b: collection must never repair a link."""

import errno
import os
import shlex
import tarfile

import pytest

from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout._artifacts import MANIFEST_NAME, collect_artifacts
from benchflow.rollout._separate_verifier import _install_script, build_transfer_payload
from benchflow.sandbox.protocol import ExecResult
from tests.reachable_fixture import directory as reachable_directory
from tests.test_separate_verifier import LocalTransport


class TarTransport(LocalTransport):
    async def exec(self, command, **kwargs):
        if shlex.split(command)[0] == "python3":
            return ExecResult(127, "", "python3: not found")
        return await super().exec(command, **kwargs)


@pytest.mark.parametrize("route", ["mounted", "python", "tar"])
@pytest.mark.parametrize("prefix", ["", "nested/"])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("suffix", ["/", "/.", "//", "/./", "//."])
async def test_invalid_directory_requirement_never_becomes_a_file(
    tmp_path, route, prefix, absolute, suffix
):
    root = tmp_path / "sandbox-artifacts"
    (root / "nested").mkdir(parents=True)
    (root / prefix / "value").write_text("42\n")
    target = (str(root / prefix / "value") if absolute else "value") + suffix
    (root / prefix / "answer").symlink_to(target)
    with pytest.raises(OSError) as error:
        (root / prefix / "answer").read_bytes()
    assert error.value.errno == errno.ENOTDIR
    await _check_transfer(tmp_path, root, route, prefix, target, admitted=False)


@pytest.mark.parametrize("route", ["mounted", "python", "tar"])
@pytest.mark.parametrize("prefix", ["", "nested/"])
@pytest.mark.parametrize(
    "target,admitted",
    [
        ("value", True),
        ("directory", True),
        ("directory/value", True),
        ("missing", True),
        ("file with spaces", True),
        ("unicodé", True),
        ("back\\slash", True),
        ("value/child", True),
        ("directory/", False),
        ("directory/.", False),
        ("./value", False),
        ("directory/./value", False),
        ("directory//value", False),
        ("directory/../value", False),
        ("../value", False),
        (".", False),
        ("..", False),
        ("/", False),
        ("//", False),
        ("/tmp/outside-marker", False),
        ("middle", False),
        ("dirlink/value", False),
        ("cycle", False),
        ("self", False),
        ("absolute", False),
        ("double-absolute", False),
    ],
)
async def test_only_plain_unchanged_relative_links_reach_verifier(
    tmp_path, route, prefix, target, admitted
):
    root = tmp_path / "sandbox-artifacts"
    base = root / prefix
    (base / "directory").mkdir(parents=True)
    for name in [
        "value",
        "directory/value",
        "file with spaces",
        "unicodé",
    ]:
        (base / name).write_text("42\n")
    # R7-1 at a2052fa6: only the explicit backslash case carries a name
    # forbidden by portable archive extraction. Other link cases stay valid.
    if target == "back\\slash":
        (base / target).write_text("42\n")
    (base / "middle").symlink_to("value")
    (base / "dirlink").symlink_to("directory")
    (base / "cycle").symlink_to("self")
    (base / "self").symlink_to("cycle")
    if target in {"absolute", "double-absolute"}:
        target = ("/" if target == "double-absolute" else "") + str(base / "value")
    (base / "answer").symlink_to(target)
    await _check_transfer(tmp_path, root, route, prefix, target, admitted=admitted)


async def _check_transfer(tmp_path, root, route, prefix, target, *, admitted):
    trial = tmp_path / "trial"
    trial.mkdir()
    mounted = route == "mounted"
    if mounted:
        root.rename(trial / "artifacts")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
    manifest = await collect_artifacts(
        TarTransport() if route == "tar" else LocalTransport(),
        artifacts=[],
        workspace=str(workspace),
        artifacts_dir=reachable_directory(trial / "artifacts"),
        manifest_path=trial / MANIFEST_NAME,
        mounted=mounted,
        logs_source=str(root),
    )
    collection = manifest["collections"][0]
    if not mounted and "\\" in target:
        assert collection["status"] == "error", collection
        assert "Unsafe evidence path" in collection["reason"]
        assert not (trial / "artifacts" / prefix / "answer").exists()
        return
    assert collection["status"] == "collected", collection
    if not admitted:
        assert any(
            item.get("path") == prefix + "answer"
            and item.get("link_target") == target
            and item.get("reason") == "symlink_refused"
            for item in collection.get("exclusions", [])
        ), collection
    archive = tmp_path / "payload.tar"
    transfer = build_transfer_payload(trial, archive)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        name = "logs/artifacts/" + prefix + "answer"
        assert (name in names) == admitted
        if admitted:
            assert os.fsencode(tar.getmember(name).linkname) == os.fsencode(target)
    restored = tmp_path / "restored"
    restored.mkdir()
    installed = await LocalTransport().exec(
        shlex.join(
            [
                "sh",
                "-c",
                _install_script(
                    str(archive), str(workspace), transfer["sha256"], str(restored)
                ),
            ]
        )
    )
    assert installed.return_code == 0, installed.stderr
    answer = restored / "logs/artifacts" / prefix / "answer"
    if admitted:
        assert os.fsencode(os.readlink(answer)) == os.fsencode(target)
        original = (trial / "artifacts" if mounted else root) / prefix / "answer"
        assert _read_result(answer) == _read_result(original)
    else:
        assert not os.path.lexists(answer)


def _read_result(path):
    try:
        return path.read_bytes()
    except OSError as error:
        return error.errno
