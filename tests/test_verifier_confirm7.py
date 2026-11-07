"""Guards V3-CONFIRM7-3..6 at f656f4a0 in labeled, mount-free containers."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox.lockdown import _build_verifier_env
from benchflow.task import RolloutPaths, Verifier
from tests import test_verifier_environment_refusals as fixtures


@pytest.fixture(scope="module")
def image():
    yield from fixtures.image.__wrapped__()


async def verify(
    box,
    tmp_path,
    script,
    *,
    required=(),
    optional=(),
    before=None,
    helper=False,
    user="root",
):
    """Use the real upload, backend env wrapper and reward reader on both tips."""
    task = fixtures._lifecycle_task(tmp_path, script)
    if helper:
        package = task.paths.tests_dir / "bin"
        package.mkdir()
        program = package / "check-answer"
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755)
    # Keep the regression executable on f656f4a0, before this schema field exists.
    object.__setattr__(
        task.config.verifier,
        "required_tools",
        None if required is None else list(required),
    )
    task.config.verifier.optional_tools = list(optional)
    task.config.verifier.user = user
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()

    async def harden(env, task, sandbox_user, **kwargs):
        task.config.verifier.env = await _build_verifier_env(
            env, task, sandbox_user, "/app"
        )
        if before:
            before(task.config.verifier.env)

    planes = SimpleNamespace(harden_before_verify=harden, verifier=Verifier)
    result = await _verify_rollout(box.env, task, paths, {}, planes, "agent", "/app")
    record = paths.verifier_dir / "trusted-path.json"
    return (*result[:2], json.loads(record.read_text()) if record.exists() else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["bin", ".", "", "../tools", " bin "])
@pytest.mark.parametrize("probe", ["command -v", "which", "type"])
async def test_unrelated_names_never_become_lookup_results(
    image, tmp_path, entry, probe
):
    """Guards CONFIRM7-3: global stubs turn an absent lookup into a false 1."""
    with fixtures._box(image, f"/usr/bin:/bin:{entry}") as box:
        fixtures._sh(
            box, "mkdir -p /opt/fixtures; cp /bin/true /opt/fixtures/check-answer"
        )
        reward, error, _ = await verify(
            box,
            tmp_path,
            f"#!/bin/sh\nif {probe} check-answer >/dev/null 2>&1; then echo 1; "
            "else echo 0; fi > /logs/verifier/reward.txt\n",
            # An optional discovery of an absent tool must remain absent.
            required=[],
        )
    assert (reward, error) == ({"reward": 0.0}, None)


@pytest.mark.asyncio
async def test_undeclared_lookup_dependency_is_unscored(image, tmp_path):
    """Guards CONFIRM7-3's original task with no new author contract."""
    with fixtures._box(image, "/usr/bin:/bin:bin") as box:
        fixtures._sh(
            box, "mkdir -p /opt/fixtures; cp /bin/true /opt/fixtures/check-answer"
        )
        reward, error, _ = await verify(
            box,
            tmp_path,
            "#!/bin/sh\nif command -v check-answer >/dev/null; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n",
            required=None,
        )
    assert reward is None
    assert error and "required_tools" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("score", ["0", "1"])
@pytest.mark.parametrize(
    "name", ["a" * 255, "." + "b" * 254, "-" + "c" * 254, "é" * 127 + "d"]
)
async def test_maximum_length_names_leave_refusal_evidence(
    image, tmp_path, name, score, caplog
):
    """Guards CONFIRM7-4: a legal basename must fit without a marker prefix."""
    with fixtures._box(image) as box:
        program = "/opt/probe/bin/" + name
        fixtures._write(box, program, "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, f"chmod 755 {shlex.quote(program)}")
        reward, error, record = await verify(
            box,
            tmp_path,
            f"#!/bin/sh\n{shlex.quote(name)} >/dev/null 2>&1 || true\n"
            f"echo {score} > /logs/verifier/reward.txt\n",
        )
    assert reward == {"reward": 0.0}
    assert error is None
    assert name in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": name, "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["upload", "during-verification"])
@pytest.mark.parametrize("entry", ["bin", "/tests/bin", "/tests/bin/../bin"])
@pytest.mark.parametrize(
    "required", [None, ["check-answer"]], ids=["undeclared", "declared"]
)
async def test_late_relative_helpers_cannot_become_scored_failures(
    image, tmp_path, when, required, entry
):
    """Guards CONFIRM7-5: neither a late upload nor generated helper is inventoried."""
    with fixtures._box(image, f"/usr/bin:/bin:{entry}") as box:
        script = "#!/bin/sh\ncd /tests\n"
        if when == "during-verification":
            script += "mkdir -p bin; printf '#!/bin/sh\\nexit 0\\n' > bin/check-answer; chmod +x bin/check-answer\n"
        script += "if check-answer >/dev/null 2>&1; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n"
        reward, error, _ = await verify(
            box, tmp_path, script, required=required, helper=when == "upload"
        )
    assert reward is None
    assert error and "required_tools" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("inventory", ["many", "newline", "non-utf8"])
async def test_unrelated_files_never_veto_an_independent_score(
    image, tmp_path, inventory
):
    """Guards CONFIRM7-6: unrelated names cannot exhaust or invalidate PATH evidence."""
    with fixtures._box(image, "/usr/bin:/bin:bin") as box:
        setup = "mkdir -p /opt/fixtures; "
        if inventory == "many":
            setup += "i=0; while [ $i -lt 4097 ]; do ln -s /bin/true /opt/fixtures/fixture-$i; i=$((i+1)); done"
        else:
            suffix = (
                "'line\nbreak'" if inventory == "newline" else "\"$(printf '\\377')\""
            )
            setup += f"cp /bin/true /opt/fixtures/{suffix}"
        fixtures._sh(box, setup)
        reward, error, _ = await verify(
            box,
            tmp_path,
            "#!/bin/sh\n/usr/bin/cmp /bin/true /bin/true && echo 1 > /logs/verifier/reward.txt\n",
            required=[],
        )
    assert (reward, error) == ({"reward": 1.0}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "unwritable"])
async def test_marker_storage_failure_invalidates_optional_scores(
    image, tmp_path, damage, caplog
):
    """Guards CONFIRM7-4: unavailable evidence is never interpreted as no use."""
    with fixtures._box(image) as box:
        fixtures._write(box, "/opt/probe/bin/check-answer", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, "chmod 755 /opt/probe/bin/check-answer")

        def damage_storage(env):
            directory = env["PATH"].rsplit(":", 1)[-1].removesuffix("/bin")
            # Dummy fixture state only, in this disposable container.
            fixtures._sh(box, f"rm -r {shlex.quote(directory + '/hits')}")
            if damage == "unwritable":
                fixtures._sh(box, f"mkdir -m 555 {shlex.quote(directory + '/hits')}")

        reward, error, _ = await verify(
            box,
            tmp_path,
            "#!/bin/sh\ncheck-answer >/dev/null 2>&1 || true\necho 1 > /logs/verifier/reward.txt\n",
            optional=["check-answer"],
            before=damage_storage,
        )
    assert reward == {"reward": 0.0}
    assert error is None
    assert "unreadable" in caplog.text
    assert "Verifier scored 0" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("oracle", [True, False], ids=["oracle", "nop"])
async def test_controls(image, tmp_path, oracle):
    """Controls for CONFIRM7's interleaved replay, using absolute cmp."""
    with fixtures._box(image, "/usr/bin:/bin:bin") as box:
        if oracle:
            fixtures._write(box, "/app/answer", "expected\n")
        reward, error, _ = await verify(
            box,
            tmp_path,
            "#!/bin/sh\nprintf 'expected\\n' > /tests/expected\nif /usr/bin/cmp /app/answer /tests/expected 2>/dev/null; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n",
            required=[],
        )
    assert (reward, error) == ({"reward": float(oracle)}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [False, True], ids=["hit", "write-denied"])
async def test_nonroot_marker_writes_fail_closed(image, tmp_path, damage, caplog):
    """Guards CONFIRM7-4 when the task's verifier user is non-root."""
    with fixtures._box(image) as box:
        fixtures._write(box, "/opt/probe/bin/check-answer", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, "chmod 755 /opt/probe/bin/check-answer")

        def deny_write(env):
            directory = env["PATH"].rsplit(":", 1)[-1].removesuffix("/bin")
            # Deny the append as nobody; invalidating the pre-created state
            # file must not require allocating disk space.
            fixtures._sh(
                box,
                f"if [ -d {shlex.quote(directory)}/hits/1 ]; then "
                f"chmod 444 {shlex.quote(directory)}/hits/*/state; "
                f"else chmod 555 {shlex.quote(directory)}/hits; fi",
            )

        reward, error, record = await verify(
            box,
            tmp_path,
            "#!/bin/sh\ncheck-answer >/dev/null 2>&1 || true\necho 1 > /logs/verifier/reward.txt\n",
            user="nobody",
            before=deny_write if damage else None,
        )
    assert reward == {"reward": 0.0}
    assert error is None
    assert ("unreadable" if damage else "check-answer") in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"]
