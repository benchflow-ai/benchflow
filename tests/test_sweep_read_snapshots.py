"""Guards CONFIRM-taskmd-rewards-R6-1 at f854531f and integ a662659d."""

import os

import pytest

from benchflow.rollout import _verifier_recovery as recovery
from tests.test_verifier_recovery import child, trial  # noqa: F401


def _plant(directory, marker, kind, spelling):
    directory.mkdir(parents=True, exist_ok=True)
    target = marker.parent if kind == "directory" else marker
    if spelling == "absolute":
        text = str(target)
    elif spelling == "relative":
        text = os.path.relpath(target, directory)
    else:
        # A legal workspace-relative link becomes controller-absolute after
        # relocation: excess .. components stop at / on POSIX.
        text = "../" * 64 + str(target).lstrip("/")
    (directory / "leak").symlink_to(text, target_is_directory=kind == "directory")
    (directory / "reward.json").write_text('{"reward": 1}')
    return text


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("spelling", ["absolute", "relative", "deep"])
def test_recovery_publication_preserves_links_as_entries(tmp_path, kind, spelling):
    """Guards f854531f (R6-1): a private restored copy must not dereference relocated links."""
    marker = tmp_path / "controller-only" / "marker.json"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    root = tmp_path / "trial"
    attempt = root / "verifier-recovery" / "attempt"
    text = _plant(attempt / "verifier", marker, kind, spelling)
    recovery._publish_verifier_dir(root, attempt)
    leaf = root / "verifier" / "leak"
    assert leaf.is_symlink(), "controller marker was materialized during publication"
    assert os.readlink(leaf) == text
    assert (root / "verifier" / "reward.json").read_text() == '{"reward": 1}'
    assert marker.read_text() == "DUMMY_CONTROLLER_MARKER"


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.asyncio
async def test_recovered_outputs_stay_links_through_both_copies(
    trial,  # noqa: F811
    child,  # noqa: F811
    monkeypatch,
    tmp_path,
    kind,
):
    """Guards R6-1 through actual recovery and a trusted verifier's archive output."""
    marker = tmp_path / "controller-only" / "marker.json"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")

    async def verify(env, task, paths, *args, **kwargs):
        # Model cp -a of restored output: the verifier preserves the agent's
        # entry. Its own code and reward are trusted and unchanged.
        _plant(paths.rollout_dir / "verifier", marker, kind, "deep")
        return {"reward": 1}, None, None

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    assert await recovery.recover_verifier(trial) == ({"reward": 1}, None)
    root = trial._require_rollout_dir()
    assert (root / "verifier" / "leak").is_symlink()
    copies = list(root.glob("verifier-recovery/*/verifier/leak"))
    assert len(copies) == 1 and copies[0].is_symlink()
    child.run.assert_not_awaited()


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("spelling", ["absolute", "relative", "deep"])
@pytest.mark.parametrize("publication", ["separate", "recovery", "download"])
def test_every_publication_uses_the_no_follow_copy(
    tmp_path, kind, spelling, publication
):
    """Guards f854531f (R6-1) normal, recovery and downloaded backend-neutral output paths."""
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.publish._snapshot import staged_folder
    from benchflow.rollout._separate_verifier import _publish_verifier_outputs

    source = tmp_path / "source"
    marker = tmp_path / "controller" / "marker.json"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    text = _plant(source, marker, kind, spelling)
    root = tmp_path / "trial"
    root.mkdir()
    if publication == "separate":
        _publish_verifier_outputs(ReachablePath.directory(source), root, {})
    else:
        attempt = root / "attempt"
        attempt.mkdir()
        ReachablePath.directory(source).copy_to(attempt / "verifier")
        if publication == "recovery":
            recovery._publish_verifier_dir(root, attempt)
        else:
            ReachablePath.rollout(root, "verifier").publish_entries(
                attempt / "verifier"
            )
    assert (root / "verifier/leak").is_symlink()
    assert os.readlink(root / "verifier/leak") == text
    # Both upload clients use this regular-file-only boundary. A preserved
    # directory link cannot be enumerated, nor a leaf materialized later.
    with staged_folder(root) as published:
        assert list(published.rglob("leak")) == []
        assert (published / "verifier/reward.json").read_text() == '{"reward": 1}'
        assert all(
            b"DUMMY_CONTROLLER_MARKER" not in p.read_bytes()
            for p in published.rglob("*")
            if p.is_file()
        )


@pytest.mark.parametrize("root", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize("entry", ["leaf", "ancestor", "root", "fifo"])
@pytest.mark.parametrize("reader", ["copy", "text", "file"])
def test_snapshot_refuses_links_and_special_inputs(tmp_path, root, entry, reader):
    """Guards f854531f (R6-1) across every mounted root and relative lookup component."""
    from benchflow._utils.reachable_write import ReachablePath, snapshot_file

    real = tmp_path / root
    (real / "sub").mkdir(parents=True)
    marker = tmp_path / "controller" / "answer.txt"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    held = ReachablePath.directory(real)
    if entry in {"leaf", "fifo"}:
        path = real / "sub/answer.txt"
        if entry == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(marker)
    else:
        path = real if entry == "root" else real / "sub"
        path.rename(path.with_name(path.name + "-retired"))
        path.symlink_to(marker.parent, target_is_directory=True)
    source = held / "sub/answer.txt"
    if reader == "copy" and entry == "leaf":
        source.copy_to(tmp_path / "copy")
        assert (tmp_path / "copy").is_symlink()
    else:
        with pytest.raises(OSError):
            if reader == "copy":
                source.copy_to(tmp_path / "copy")
            elif reader == "text":
                source.read_text()
            else:
                with snapshot_file(source):
                    pytest.fail("unsafe input acquired a snapshot")


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
@pytest.mark.parametrize(
    "judge", ["agent", "ors", "legacy", "taskmd-read", "taskmd-citation"]
)
def test_every_judge_route_refuses_restored_links(tmp_path, kind, judge):
    """Guards f854531f and 616001d88: refuse relocated links with typed evidence errors."""
    from types import SimpleNamespace

    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.rewards.file_readers import find_deliverables
    from benchflow.task.verifier_errors import (
        AgentJudgeInputError,
        ORSEpisodeInputError,
        VerifierOutputParseError,
    )
    from benchflow.task.verifier_judge_inputs import _read_agent_judge_input
    from benchflow.task.verifier_ors_episode import _load_ors_episode_records
    from benchflow.taskmd import judging

    root = tmp_path / "restored"
    root.mkdir()
    marker = tmp_path / "controller" / "answer.txt"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    (root / "safe.txt").write_text("answer")
    if kind == "leaf":
        (root / "answer.txt").symlink_to(marker)
        relative = "answer.txt"
    else:
        (root / "sub").symlink_to(marker.parent, target_is_directory=True)
        relative = "sub/answer.txt"
    held = ReachablePath.directory(root) / relative
    if judge == "agent":
        with pytest.raises(AgentJudgeInputError):
            _read_agent_judge_input(held)
    elif judge == "ors":
        with pytest.raises((OSError, ORSEpisodeInputError)):
            _load_ors_episode_records(
                held, strategy=SimpleNamespace(), declared_path=relative
            )
    elif judge == "legacy":
        assert find_deliverables(root) == {"safe.txt": "answer"}
    else:
        files = judging.JudgeFiles(root=root, kept=["/"], views={})
        try:
            if judge == "taskmd-read":
                answer = judging.read_result(
                    files, {}, 1, "FENCE", {"path": "/" + relative}
                )
                assert "DUMMY_CONTROLLER_MARKER" not in repr(answer)
            else:
                context = judging.CitationContext(
                    files=files, record={}, tests={}, instruction=b"", workdir=None
                )
                answer = judging.check_citation(
                    {
                        "source": "file",
                        "path": "/" + relative,
                        "quote": "DUMMY_CONTROLLER_MARKER",
                    },
                    context,
                    {},
                )
                assert answer["verified"] is not True
        except VerifierOutputParseError as exc:
            assert isinstance(exc.__cause__, OSError)
        except OSError:
            pass


@pytest.mark.parametrize("root", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize("import_style", ["qualified", "module-alias", "name-alias"])
@pytest.mark.parametrize(
    "sink",
    [
        "copytree",
        "copy",
        "copy2",
        "copyfile",
        "copyfileobj",
        "walk",
        "open",
        "path-open",
    ],
)
def test_source_scan_covers_read_and_copy_spellings(
    tmp_path, monkeypatch, root, import_style, sink
):
    """Guards f854531f (R6-1)'s source scan: constructors and import aliases cannot hide I/O."""
    from tests import test_reachable_write_scan as scan

    if sink == "path-open":
        imports, operation = "", 'self.source.open("rb")'
    else:
        module = "os" if sink == "walk" else "builtins" if sink == "open" else "shutil"
        if import_style == "qualified":
            imports, name = f"import {module}", f"{module}.{sink}"
        elif import_style == "module-alias":
            imports, name = f"import {module} as io_alias", f"io_alias.{sink}"
        else:
            imports, name = f"from {module} import {sink} as io_alias", "io_alias"
        args = (
            'self.source, "private"'
            if module == "shutil"
            else "self.source, followlinks=True"
            if sink == "walk"
            else 'self.source, "rb"'
        )
        operation = f"{name}({args})"
    (tmp_path / "consumer.py").write_text(f"""{imports}
class Reader:
    def __init__(self, source):
        self.source = source
    def consume(self):
        {operation}
""")
    (tmp_path / "caller.py").write_text(f'''from benchflow.consumer import Reader

def run(rollout_dir):
    Reader(source=rollout_dir / "{root}" / "entry")
''')
    monkeypatch.setattr(scan, "SRC", tmp_path)
    monkeypatch.setattr(scan, "HELPER", tmp_path / "_utils/reachable_write.py")
    sites, _ = scan._scan()
    assert sites, f"scan missed {operation}"
    assert all(site[1] == "Reader.consume" for site in sites)


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("publication", ["recovery", "separate"])
@pytest.mark.asyncio
async def test_admitted_workspace_link_survives_restore_without_dereference(
    tmp_path, kind, publication
):
    """Guards f854531f (R6-1)'s complete payload: safe in /app, hostile only after relocation."""
    import shutil

    from benchflow.review.evidence import capture_workspace, validate_workspace
    from benchflow.rollout._separate_verifier import _publish_verifier_outputs
    from tests.test_separate_verifier import LocalTransport

    marker = tmp_path / "controller-only" / "marker.json"
    marker.parent.mkdir()
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    workspace = tmp_path / "workspace"
    output = workspace.joinpath(*(["d"] * 64))
    output.mkdir(parents=True)
    benign = workspace / str(marker).lstrip("/")
    benign.parent.mkdir(parents=True)
    benign.write_text("benign sandbox file")
    target = marker.parent if kind == "directory" else marker
    text = "../" * 64 + str(target).lstrip("/")
    (output / "leak").symlink_to(text, target_is_directory=kind == "directory")
    link_file = (
        output / "leak" / marker.name if kind == "directory" else output / "leak"
    )
    assert link_file.read_text() == "benign sandbox file"
    bundle = tmp_path / "bundle"
    manifest = await capture_workspace(LocalTransport(), str(workspace), bundle)
    assert not manifest.exclusions
    validate_workspace(bundle / "workspace", manifest)
    restored_output = (bundle / "workspace").joinpath(*(["d"] * 64))
    assert (restored_output / "leak").is_symlink()
    root = tmp_path / "trial"
    attempt = root / "verifier-recovery" / "attempt"
    # Trusted cp -a-like verifier behavior; no hostile verifier or image.
    shutil.copytree(restored_output, attempt / "verifier", symlinks=True)
    if publication == "recovery":
        recovery._publish_verifier_dir(root, attempt)
    else:
        from benchflow._utils.reachable_write import ReachablePath

        _publish_verifier_outputs(
            ReachablePath.directory(attempt / "verifier"),
            root,
            {},
        )
    assert (root / "verifier/leak").is_symlink()
    assert os.readlink(root / "verifier/leak") == text


@pytest.mark.parametrize("destination_exists", [False, True])
def test_regular_snapshot_keeps_opened_inode_and_refuses_reused_destination(
    tmp_path, monkeypatch, destination_exists
):
    """Guards f854531f (R6-1): swapping a regular name cannot change an opened source."""
    from benchflow._utils import reachable_write as rw

    source = tmp_path / "source"
    source.mkdir()
    leaf = source / "answer.txt"
    leaf.write_text("accepted")
    marker = tmp_path / "dummy-marker"
    marker.write_text("DUMMY_CONTROLLER_MARKER")
    target = tmp_path / "copy"
    if destination_exists:
        target.symlink_to(marker)
    original = rw._snapshot_stream

    def swap(stream, destination, **kwargs):
        leaf.unlink()
        leaf.symlink_to(marker)
        original(stream, destination, **kwargs)

    monkeypatch.setattr(rw, "_snapshot_stream", swap)
    held = rw.ReachablePath.directory(source) / leaf.name
    if destination_exists:
        with pytest.raises(FileExistsError):
            held.copy_to(target)
    else:
        held.copy_to(target)
        assert target.read_text() == "accepted"
    assert marker.read_text() == "DUMMY_CONTROLLER_MARKER"


@pytest.mark.parametrize("root", ["agent", "verifier", "artifacts"])
@pytest.mark.parametrize(
    "operation",
    [
        "open(file=source)",
        "io.open(file=source)",
        "os.open(path=source, flags=os.O_RDONLY)",
        "os.walk(source, True, None, True)",
        'shutil.copytree(src=source, dst="private", symlinks=True)',
        'shutil.copyfileobj(fsrc=source, fdst="private")',
    ],
)
def test_source_scan_covers_keyword_reads_and_positional_walk(
    tmp_path, monkeypatch, root, operation
):
    """Guards f854531f (R6-1): argument spelling cannot hide a raw reachable input."""
    from tests import test_reachable_write_scan as scan

    (tmp_path / "caller.py").write_text(f'''import io, os, shutil

def run(rollout_dir):
    source = rollout_dir / "{root}" / "entry"
    {operation}
''')
    monkeypatch.setattr(scan, "SRC", tmp_path)
    monkeypatch.setattr(scan, "HELPER", tmp_path / "_utils/reachable_write.py")
    sites, _ = scan._scan()
    assert len(sites) == 1
