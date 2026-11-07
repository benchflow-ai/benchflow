"""Guards V3-RECONFIRM-3..5 at 76a0723c, in disposable labeled containers."""

from __future__ import annotations

import json
import shlex

import pytest

from benchflow._utils.trusted_path import (
    TrustedPathReport,
    refused_programs,
)
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox.lockdown import _path_entries_to_judge, _relative_path_entries
from benchflow.task.config import TaskConfig
from tests import test_verifier_environment_refusals as fixtures
from tests.test_verifier_trusted_path import _rollout_paths


def test_optional_tool_contract_round_trips():
    """Guards RECONFIRM-3 at 76a0723c in the real task schema."""
    config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\noptional_tools = ["cargo", "cargo"]\n'
    )
    assert config.verifier.optional_tools == ["cargo"]
    assert TaskConfig.model_validate(config.model_dump()).verifier.optional_tools == [
        "cargo"
    ]


@pytest.mark.parametrize("name", ["", "../cargo", "cargo\x00"])
def test_optional_tools_are_program_basenames(name):
    """Guards the RECONFIRM-3 contract against ambiguous PATH spellings."""
    with pytest.raises(ValueError, match="program basenames"):
        TaskConfig.model_validate({"verifier": {"optional_tools": [name]}})


def test_record_framing_preserves_cr_and_path_spaces():
    """Guards RECONFIRM-4 at 76a0723c without a sandbox transport."""
    assert _path_entries_to_judge("/opt/open/bin :/opt/open/bin\r") == [
        "/opt/open/bin ",
        "/opt/open/bin\r",
    ]
    assert _relative_path_entries(" bin :") == [" bin ", "."]
    report = TrustedPathReport(
        path="/usr/bin:/bin",
        dropped=(("/opt/open/bin ", "writable"),),
        refusals="/tmp/dummy-refusals",
        stubs=(("\r", "/opt/open/bin "), ("%0D", "/opt/open/bin ")),
    )
    assert refused_programs(report, "1 yes\x002 no\x00") == [("\r", "/opt/open/bin ")]
    assert refused_programs(report, "1 no\x002 yes\x00") == [("%0D", "/opt/open/bin ")]


@pytest.fixture(scope="module")
def image():
    yield from fixtures.image.__wrapped__()


async def _verify(box, tmp_path, script, *, optional=(), required=()):
    fixtures._write(box, "/tests/test.sh", script)
    task = fixtures._script_task(tmp_path, script)
    task.config.verifier.optional_tools = list(optional)
    task.config.verifier.required_tools = None if required is None else list(required)
    paths = _rollout_paths(tmp_path)
    rewards, error, _ = await _verify_rollout(
        box.env,
        task,
        paths,
        {},
        fixtures._refusing_planes(fixtures._ScriptVerifier(box, task, paths)),
        "agent",
        "/app",
    )
    record = paths.verifier_dir / "trusted-path.json"
    return rewards, error, json.loads(record.read_text()) if record.exists() else None


@pytest.mark.asyncio
@pytest.mark.parametrize("score", ["1", "0.5", "0"])
@pytest.mark.parametrize("optional", [False, True])
async def test_refusal_uses_the_contract_instead_of_the_score(
    image, tmp_path, score, optional, caplog
):
    """Guards RECONFIRM-3 at 76a0723c: a missing negative check awards 1."""
    with fixtures._box(image) as box:
        fixtures._write(box, "/opt/probe/bin/detect-invalid", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, "chmod 755 /opt/probe/bin/detect-invalid")
        rewards, error, record = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\nif detect-invalid /app/answer >/dev/null 2>&1; "
            f"then echo 0; else echo {score}; fi > /logs/verifier/reward.txt\n",
            optional=("detect-invalid",) if optional else (),
        )
    if optional:
        assert (rewards, error) == ({"reward": float(score)}, None)
    else:
        assert rewards == {"reward": 0.0}
        assert error is None
        assert "detect-invalid" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": "detect-invalid", "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "folder", ["/opt/open/bin ", "/opt/open/ bin", "/opt/open/bin\r"]
)
async def test_path_whitespace_is_part_of_the_folder(image, tmp_path, folder, caplog):
    """Guards RECONFIRM-4 at 76a0723c: strip changed the directory judged."""
    with fixtures._box(image, f"/usr/bin:/bin:{folder}") as box:
        fixtures._sh(
            box, f"mkdir -p {shlex.quote(folder)}; chmod 777 {shlex.quote(folder)}"
        )
        fixtures._write(box, folder + "/check-answer", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, f"chmod 755 {shlex.quote(folder + '/check-answer')}")
        rewards, error, record = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\nif check-answer >/dev/null 2>&1; then echo 1; else echo 0; fi"
            " > /logs/verifier/reward.txt\n",
        )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "check-answer" in caplog.text
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": "check-answer", "folder": folder}
    ]


@pytest.mark.asyncio
async def test_carriage_return_program_name_survives_both_records(
    image, tmp_path, caplog
):
    """Guards RECONFIRM-4 at 76a0723c: CR is filename data, not framing."""
    with fixtures._box(image) as box:
        fixtures._write(box, "/opt/probe/bin/\r", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, "chmod 755 '/opt/probe/bin/\r'")
        rewards, error, record = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\nif '\r' >/dev/null 2>&1; then echo 1; else echo 0; fi"
            " > /logs/verifier/reward.txt\n",
        )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "Verifier scored 0" in caplog.text
    assert record["refused_programs_run"] == [
        {"program": "\r", "folder": "/opt/probe/bin"}
    ]


@pytest.mark.asyncio
async def test_non_utf8_program_name_is_refused_explicitly(image, tmp_path, caplog):
    """Guards the RECONFIRM-4 byte boundary; invalid UTF-8 cannot disappear."""
    with fixtures._box(image) as box:
        fixtures._sh(
            box,
            "n=$(printf '\\377'); printf '#!/bin/sh\\nexit 0\\n' > /opt/probe/bin/\"$n\"; "
            'chmod 755 /opt/probe/bin/"$n"',
        )
        rewards, error, _ = await _verify(
            box, tmp_path, "#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n"
        )
    assert rewards == {"reward": 0.0}
    assert error is None
    assert "non-UTF-8" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["bin\nnext", "/opt/open/bin\nnext"])
async def test_newline_path_entry_is_left_out_on_the_record(
    image, tmp_path, entry, caplog
):
    """Guards RECONFIRM-4 at 76a0723c: newline entries disappeared silently.

    7c6d5461f passes PATH entries to the judge as shell literals, so a
    newline is entry bytes like any other, no longer a refused spelling
    (docs/sandboxes/threat-model.md: each entry left out is logged with its
    reason and listed in verifier/trusted-path.json). The entry is still not
    lost: it is left out by its exact bytes, logged and recorded, and the
    verifier's own score stands."""
    with fixtures._box(image, f"/usr/bin:/bin:{entry}") as box:
        rewards, error, record = await _verify(
            box, tmp_path, "#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n"
        )
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert [row["folder"] for row in record["dropped"]] == [entry]
    assert "leaves out 1 folder(s)" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", [".", "", "bin"])
async def test_unused_relative_entry_preserves_independent_scores(
    image, tmp_path, entry
):
    """Guards RECONFIRM-5 at 76a0723c with CONFIRM7's explicit empty contract."""
    with fixtures._box(image, f"/usr/bin:/bin:{entry}") as box:
        rewards, error, _ = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\nprintf answer > /app/answer\nprintf answer > /tests/expected\n"
            "if /usr/bin/cmp /app/answer /tests/expected; then echo 1; else echo 0; fi"
            " > /logs/verifier/reward.txt\n",
            required=[],
        )
    assert (rewards, error) == ({"reward": 1.0}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/app/sub", "/opt/elsewhere"])
async def test_relative_dependency_is_recorded_after_cd(image, tmp_path, workdir):
    """Guards RECONFIRM-5 at 76a0723c: CONFIRM7 refuses the dependency before cd."""
    with fixtures._box(image, "/usr/bin:/bin:bin") as box:
        fixtures._sh(box, f"mkdir -p {workdir}/bin")
        fixtures._write(box, f"{workdir}/bin/check-answer", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, f"chmod 755 {workdir}/bin/check-answer")
        rewards, error, record = await _verify(
            box,
            tmp_path,
            f"#!/bin/sh\ncd {workdir}\nif check-answer >/dev/null 2>&1; "
            "then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n",
            required=["check-answer"],
        )
    assert rewards is None
    assert error and "check-answer" in error
    assert "required_tools" in error
    assert record["dropped"][0]["folder"] == "bin"
    assert record["refused_programs_run"] == []


@pytest.mark.asyncio
async def test_an_inventory_name_elsewhere_is_not_a_dependency(image, tmp_path):
    """Guards RECONFIRM-5: inventory alone must not invalidate a score."""
    with fixtures._box(image, "/usr/bin:/bin:bin") as box:
        fixtures._sh(box, "mkdir -p /opt/elsewhere/bin")
        fixtures._write(box, "/opt/elsewhere/bin/optional-probe", "#!/bin/sh\nexit 0\n")
        fixtures._sh(box, "chmod 755 /opt/elsewhere/bin/optional-probe")
        rewards, error, record = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\noptional-probe >/dev/null 2>&1 || true\n"
            "echo 0.5 > /logs/verifier/reward.txt\n",
            required=[],
        )
    assert (rewards, error) == ({"reward": 0.5}, None)
    assert record["refused_programs_run"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("oracle", [True, False], ids=["oracle", "nop"])
async def test_reconfirm_controls(image, tmp_path, oracle):
    """Oracle 1 and nop 0 controls for the 76a0723c RECONFIRM replays."""
    with fixtures._box(image, "/usr/bin:/bin") as box:
        if oracle:
            fixtures._write(box, "/app/answer", "expected\n")
        rewards, error, _ = await _verify(
            box,
            tmp_path,
            "#!/bin/sh\nprintf 'expected\\n' > /tests/expected\n"
            "if /usr/bin/cmp /app/answer /tests/expected 2>/dev/null; "
            "then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n",
        )
    assert (rewards, error) == ({"reward": float(oracle)}, None)
