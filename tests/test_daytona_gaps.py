"""DAYTONA-GAPS (DX-1 round 2 + DX-3): the rules that need no provider fake.

Each test fails at 29b6ea03c (DX-1's tip) and passes once its rule holds:

- a staged payload runs whole or not at all (DX-1 review A L4, review B L1);
- an image's start-up variables are carried aside as data on Daytona and the
  task route gets exactly the image's values back (rule 2);
- a manifest snapshot chain keeps the time its 8 KiB batches had, and its
  error carries a bounded command (review A L5);
- every send the reviews listed goes through ``_transport.sent`` (review A
  L1, review B L2).

Round 3 (review A M1): the start-up names are one list
(``_base.startup_name``) for every backend, and an image or sandbox env
that sets bash's (an exported function, SHELLOPTS, BASHOPTS) is refused at
start; review A L2 (the stage proof follows no link it made root's) and L4
(a raised upload or unpack leaves no stage folder).

The contract cases on every backend's fake (a link at the staging folder,
a 20 KB compose env, a 2,000-file directory, the start-up names) are in
``tests/test_exec_transport_contract.py``.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import _transport

SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"


def _script(size: int, tail: str) -> str:
    line = ": 'it''s' \"quoted\" $HOME # padding\n"
    return line * (size // len(line)) + tail + "\n"


SHELLS = [s for s in ("/bin/sh", "/bin/bash", shutil.which("dash")) if s]


@pytest.mark.parametrize("shell", sorted(set(SHELLS)))
def test_a_cut_payload_runs_nothing_and_a_whole_one_runs(tmp_path, shell):
    """Review B L1's reproducer: a payload ``rm -rf <d>/keep/sub`` cut
    before ``/sub`` deleted ``<d>/keep`` at 29b6ea03c."""
    sub = tmp_path / "keep" / "sub"
    sub.mkdir(parents=True)
    script = _script(100_000, f"rm -rf {tmp_path}/keep/sub")
    fitted = _transport.fit(lambda c: [shell, "-c", c], script, cat="/bin/cat")
    assert fitted.payload is not None
    cut = fitted.payload[: fitted.payload.index(b"/keep/sub") + len(b"/keep")]
    run = subprocess.run(fitted.sent, input=cut, capture_output=True, timeout=60)
    assert run.returncode == _transport.READ_FAILED, run.stderr
    assert sub.exists()
    assert b"incomplete" in run.stderr
    # Every cut, the last byte included, runs nothing.
    run = subprocess.run(
        fitted.sent, input=fitted.payload[:-1], capture_output=True, timeout=60
    )
    assert run.returncode == _transport.READ_FAILED and sub.exists()
    whole = subprocess.run(
        fitted.sent, input=fitted.payload, capture_output=True, timeout=60
    )
    assert whole.returncode == 0, whole.stderr
    assert not sub.exists() and (tmp_path / "keep").exists()


def test_a_staged_payload_keeps_its_exit_status_and_trailing_newlines(tmp_path):
    script = _script(100_000, "printf 'a\\n\\n'; exit 7") + "\n\n"
    fitted = _transport.fit(lambda c: ["/bin/sh", "-c", c], script, cat="/bin/cat")
    run = subprocess.run(fitted.sent, input=fitted.payload, capture_output=True)
    assert run.returncode == 7 and run.stdout == b"a\n\n"


def test_the_docker_and_apple_stdin_is_framed_too():
    """The pipe backends build their own stdin (env lines, then the
    command): it ends with the reader's end line as well."""
    fitted = _transport.fit(lambda c: ["sh", "-c", c], "x" * 70_000, cat="/bin/cat")
    framed = fitted.frame(b"export A=1\n" + b"x" * 70_000)
    assert framed.endswith(fitted.end.encode()) and fitted.end in fitted.line
    with pytest.raises(ValueError):
        fitted.frame(b"echo " + fitted.end.encode())


# --- rule 2: start-up variables carried aside --------------------------------


@pytest.mark.asyncio
async def test_an_image_env_and_bash_env_reach_only_the_task_route(
    tmp_path, monkeypatch
):
    """The transport pack's noisy-shell-env on the Daytona harness: an image
    whose ENV and BASH_ENV name files that print text. Refused at create at
    29b6ea03c ("image loader settings require protected task routing")."""
    from benchflow.sandbox._task_route import as_task_command
    from benchflow.task import RolloutPaths
    from tests.test_verifier_script_shell import LocalImage, _backend

    pytest.importorskip("daytona")
    sh_env = tmp_path / "sh-env.sh"
    sh_env.write_text('printf "TRANSPORT_ENV_STARTUP\\n"\n')
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text('printf "TRANSPORT_BASH_STARTUP\\n"\n')
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path,
        paths,
        with_bash=True,
        image_env={"ENV": str(sh_env), "BASH_ENV": str(bash_env)},
    )
    exec_ = _backend("daytona", image, monkeypatch)
    plumbing = await exec_('echo "ran"', None)
    assert plumbing.return_code == 0, plumbing.stderr
    assert plumbing.stdout == "ran\n"
    created = exec_.toolbox["creates"][-1].env_vars
    assert created["ENV"] == created["BASH_ENV"] == ""
    assert created["_BF_IMAGE_ENV"] == str(sh_env)
    assert created["_BF_IMAGE_BASH_ENV"] == str(bash_env)
    with as_task_command():
        task = await exec_('printf "%s|%s\\n" "$ENV" "$BASH_ENV"', None)
    assert task.return_code == 0, task.stderr
    assert task.stdout == f"{sh_env}|{bash_env}\n"
    assert "STARTUP" not in plumbing.stderr + task.stderr


# --- review A L5: the snapshot chain's time ----------------------------------


def test_a_snapshot_chain_keeps_the_time_its_batches_had():
    from benchflow.environment import manifest_env

    small = "cp a b"
    assert manifest_env._state_timeout(small) == 120
    chain = manifest_env._state_command(["cp /x/a.db /y/a.db"] * 2000)
    units = -(-len(chain.encode()) // (8 * 1024))
    assert units > 1
    assert manifest_env._state_timeout(chain) == 120 * units
    told = manifest_env._state_error_command(chain)
    assert len(told) < 4200 and told.endswith(f"({len(chain)} characters)")


# --- review A L1 / review B L2: every send is seen ---------------------------

# (file under src/benchflow, function qualname): each hands its provider a
# command or argv, and calls _transport.sent first.
SENDS = [
    ("sandbox/docker.py", "DockerSandbox.bfx_call"),
    ("sandbox/docker.py", "DockerSandbox.bfx_session"),
    ("sandbox/docker.py", "DockerSandbox.bfx_tar_out"),
    ("sandbox/docker.py", "_DockerCredentialOps.run_as"),
    ("sandbox/_exec_channel.py", "ExecChannel.start"),
    ("sandbox/_exec_channel.py", "WarmExecPool._spawn"),
    ("sandbox/process/docker.py", "DockerProcess.start"),
    ("sandbox/process/apple.py", "AppleContainerProcess.start"),
    ("sandbox/process/apple.py", "AppleContainerProcess._write_env_to_container"),
    ("sandbox/process/apple.py", "AppleContainerProcess._remove_env_from_container"),
    ("sandbox/process/daytona.py", "_cleanup_daytona_remote_env_file"),
    ("sandbox/process/daytona.py", "DaytonaProcess.start"),
    ("sandbox/process/daytona.py", "DaytonaPtyProcess.start"),
    ("sandbox/modal_impl.py", "ModalSandbox._check_bfx"),
]


def _functions(tree: ast.AST) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[f"{prefix}{child.name}"] = child

    visit(tree, "")
    return found


@pytest.mark.parametrize(("path", "name"), SENDS)
def test_every_listed_send_is_seen_by_the_bound(path, name):
    functions = _functions(ast.parse((SRC / path).read_text()))
    assert name in functions, f"{path}: no {name}"
    calls = [
        node
        for node in ast.walk(functions[name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sent"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_transport"
    ]
    assert calls, f"{path}:{name} sends without _transport.sent"


# --- round 3: one list of start-up names (review A M1) ----------------------

_SAMPLE_NAMES = [
    "LD_PRELOAD",
    "BASH_ENV",
    "ENV",
    "MALLOC_TRACE",
    "SHELLOPTS",
    "BASHOPTS",
    "BASH_FUNC_sed%%",
    "BASH_FUNC_a-b%%",
    "BASH_FUNC_g()",
    "BASH_FUNC_",
    "LD_DEBUG",
    "PATH",
    "ENVX",
    "XSHELLOPTS",
    "BASH_FUNCX",
    "_BF_IMAGE_BASH_FUNC_x%%",
]


def test_the_start_up_names_are_one_list():
    """The shell pattern the probes use and the route's own names match
    exactly ``startup_name``, prefix included."""
    from benchflow.sandbox._base import (
        route_owned_name,
        startup_name,
        startup_name_pattern,
    )

    script = f"case $1 in {startup_name_pattern()}) echo y;; *) echo n;; esac"
    for name in _SAMPLE_NAMES:
        out = subprocess.run(
            ["/bin/sh", "-c", script, "sh", name], capture_output=True, text=True
        ).stdout.strip()
        assert (out == "y") == startup_name(name), name
        assert not startup_name(name) or route_owned_name(name), name


# Files that may spell start-up names themselves, and why: none of them is
# where a sandbox shell starts.
_OWN_NAME_USES = {
    # The names su must never assign (bash keeps them read-only).
    "sandbox/_su.py": "su's read-only names",
    # The operator's environment for docker compose on this machine.
    "task/compose_env.py": "the host's Compose environment",
    # The verifier's pinned values (PYTHONPATH, LD_PRELOAD, ... empty) and
    # the task's own BASH_ENV, inside the task's command.
    "sandbox/lockdown.py": "the verifier command's own pinned env",
}


def _literals(tree: ast.AST) -> list[str]:
    docs = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
    ]


def test_no_module_keeps_its_own_list_of_start_up_names():
    """The probes, the emptying, the carry-aside, the
    receipt and the refusals take their names from ``_base``; no other
    module spells bash's start-up names or copies the loader list."""
    from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS

    own = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "sandbox/_base.py":
            continue
        literals = _literals(ast.parse(path.read_text()))
        if rel in _OWN_NAME_USES:
            continue
        if any(
            "BASH_FUNC_" in text or text in ("SHELLOPTS", "BASHOPTS")
            for text in literals
        ):
            own.append(rel)
        # A copy of the loader list: three or more of its names.
        if len(set(literals) & set(SYSTEM_SHELL_EMPTY_VARS)) >= 3:
            own.append(rel)
    assert not own, own


# Each start path, and the call in it that admits the image's environment
# before any shell runs (the contract file runs each call against an image).
_ADMISSIONS = [
    (
        "sandbox/daytona.py",
        "DaytonaSandbox._create_sandbox_without_toolbox_bash_env",
        "_toolbox_startup_env",
    ),
    (
        "sandbox/daytona_strategies.py",
        "_DaytonaDirect.start",
        "_create_sandbox_without_toolbox_bash_env",
    ),
    # DinD's task image runs in the inner containers: the overlay admits it.
    (
        "sandbox/daytona_dind.py",
        "_DaytonaDinD._prepare_inert_bootstrap",
        "_inert_bootstrap.prepare",
    ),
    ("sandbox/_inert_bootstrap.py", "overlay", "refused_startup_names"),
    (
        "sandbox/docker.py",
        "DockerSandbox._snapshot_image_bind_baseline",
        "refused_startup_names",
    ),
    ("sandbox/modal_impl.py", "ModalSandbox.start", "_refuse_startup_names"),
    ("sandbox/setup.py", "BenchFlowModalSandbox.start", "_refuse_startup_names"),
    (
        "sandbox/apple_container.py",
        "AppleContainerSandbox.start",
        "_refuse_startup_names",
    ),
    (
        "sandbox/agentcore_image.py",
        "AgentCoreImagePublisher.generated_dockerfile_text",
        "_STARTUP_NAMES_CHECK",
    ),
]


def _qualified(tree: ast.AST) -> dict[str, str]:
    out: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    out[name] = ast.unparse(child)
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


@pytest.mark.parametrize(("path", "function", "call"), _ADMISSIONS)
def test_every_start_path_admits_the_image_environment(path, function, call):
    # A class made inside a factory (BenchFlowModalSandbox) has the
    # factory's name in front.
    found = {
        name: source
        for name, source in _qualified(ast.parse((SRC / path).read_text())).items()
        if name == function or name.endswith(f".{function}")
    }
    assert found, f"{path}: no {function}"
    for name, source in found.items():
        assert call in source, f"{path}:{name} does not call {call}"


def test_daytona_creates_only_through_the_admission():
    """A sandbox of the task's image is created only inside the admission,
    which probes it before any other exec. The DinD VM is BenchFlow's own
    DinD image; the task's image runs in its inner containers."""
    for path in sorted((SRC / "sandbox").glob("daytona*.py")):
        for name, source in _qualified(ast.parse(path.read_text())).items():
            if "._create_sandbox(" in source:
                assert name.endswith("_create_sandbox_without_toolbox_bash_env") or (
                    name == "_DaytonaDinD.start" and "dind_image" in source
                ), f"{path.name}:{name} creates a sandbox without the admission"


_IMPORTED = '() { echo BF-IMAGE-FUNCTION-RAN >&2; command sed "$@"; }'


@pytest.mark.parametrize(
    ("shell", "env", "values", "refused"),
    [
        ("bash", {}, {}, []),
        (
            "bash",
            {"BASH_ENV": "/a\nE5:x=y", "LD_LIBRARY_PATH": "/opt/lib"},
            {"BASH_ENV": "/a\nE5:x=y", "LD_LIBRARY_PATH": "/opt/lib"},
            [],
        ),
        ("bash", {"BASH_FUNC_sed%%": _IMPORTED}, {}, ["BASH_FUNC_sed%%"]),
        ("bash", {"BASH_FUNC_sed%%": ""}, {}, []),
        ("bash", {"SHELLOPTS": "xtrace", "BASHOPTS": ""}, {}, ["SHELLOPTS"]),
        ("dash", {"SHELLOPTS": "xtrace"}, {}, ["SHELLOPTS"]),
        # dash passes no BASH_FUNC_ entry on: nothing it starts imports one.
        ("dash", {"BASH_FUNC_sed%%": _IMPORTED, "ENV": "/e"}, {"ENV": "/e"}, []),
    ],
)
def test_the_daytona_probe_reads_the_toolbox_shells_start_environment(
    shell, env, values, refused
):
    """The create probe, in a real POSIX-mode bash and in dash: every
    start-up entry of the shell's own start environment, builtins only."""
    from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
    from benchflow.sandbox.daytona import (
        _TOOLBOX_STARTUP_PROBE,
        _parse_startup_probe,
    )

    program = shutil.which(shell)
    if program is None:
        pytest.skip(f"no {shell}")
    run = subprocess.run(
        [program, "-c", _TOOLBOX_STARTUP_PROBE],
        env={"PATH": "/usr/bin:/bin", "POSIXLY_CORRECT": "1", **env},
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    parsed = _parse_startup_probe((run.stdout + run.stderr).decode())
    assert parsed is not None, run.stdout
    expected = dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, "") | values
    assert parsed == (expected, refused)


@pytest.mark.parametrize(
    ("image", "name"),
    [
        # bash's POSIX mode stops before the probe at a special builtin's
        # name, and reports a name that is no identifier.
        ({"BASH_FUNC_unset%%": '() { builtin unset "$@"; }'}, "BASH_FUNC_unset%%"),
        ({"BASH_FUNC_a-b%%": "() { :; }"}, "BASH_FUNC_a-b%%"),
    ],
)
@pytest.mark.asyncio
async def test_a_function_bash_will_not_import_in_posix_mode_is_named(image, name):
    from types import SimpleNamespace

    from benchflow.sandbox._base import StartupNamesRefused
    from benchflow.sandbox.daytona import DaytonaSandbox

    program = shutil.which("bash")
    if program is None:
        pytest.skip("no bash")

    async def exec_(command, env=None, timeout=None):
        run = subprocess.run(
            [program, "-c", command],
            env={"PATH": "/usr/bin:/bin", **image_env, **(env or {})},
            capture_output=True,
            text=True,
        )
        return SimpleNamespace(exit_code=run.returncode, result=run.stdout + run.stderr)

    image_env = image
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(process=SimpleNamespace(exec=exec_))
    with pytest.raises(StartupNamesRefused, match=r"^benchflow: ") as refused:
        await sandbox._toolbox_startup_env()
    assert name in str(refused.value)


@pytest.mark.asyncio
async def test_an_image_function_is_refused_at_daytona_create(tmp_path, monkeypatch):
    """Review A M1 on the Daytona harness (the whole create path): an image
    ENV ``BASH_FUNC_sed%%`` was admitted at 9d04974bf and root's plumbing
    exec ran the function (``sed`` in its first line); now the start is
    refused with a ``benchflow:`` line and nothing runs."""
    from types import SimpleNamespace

    from benchflow.sandbox.protocol import SandboxStartupError
    from benchflow.task import RolloutPaths
    from tests.test_verifier_script_shell import LocalImage, _backend

    pytest.importorskip("daytona")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(
        tmp_path, paths, with_bash=True, image_env={"BASH_FUNC_sed%%": _IMPORTED}
    )
    exec_ = _backend("daytona", image, monkeypatch)
    # What the strategy's start sets before it creates (its startup error).
    exec_.sandbox._create_attempts = 1
    exec_.sandbox.task_env_config = SimpleNamespace(build_timeout_sec=60)
    try:
        result = await exec_('echo "ran"', None)
    except SandboxStartupError as exc:
        assert "benchflow: " in str(exc) and "BASH_FUNC_sed%%" in str(exc)
        assert exec_.sandbox._sandbox is None
        return
    assert "BF-IMAGE-FUNCTION-RAN" not in f"{result.stdout}{result.stderr}"
    pytest.fail("an image that defines a bash function was admitted")


# --- round 3: review A L2 and L4 --------------------------------------------


def test_the_stage_proof_makes_no_link_root_s():
    """Review A L2: every folder the proof makes root's is changed with
    ``chown -h`` (never through a link) and tested again before ``chmod``."""
    import re

    proof = _transport.stage_proof(
        ["/opt/benchflow/run/stage-x/cmd"], shell="/bin/sh", path="/usr/bin:/bin"
    )
    owners = re.findall(r"chown (\S+) 0:0 (\S+) && (\[ ! -L \S+ \]) && chmod", proof)
    assert len(owners) == 3 and all(flag == "-h" for flag, _, _ in owners), proof
    assert not re.search(r"chown 0:0", proof)


@pytest.mark.asyncio
async def test_a_raised_unpack_leaves_no_stage_folder(tmp_path):
    """Review A L4: the archive's stage folder is removed when the upload or
    the unpack exec raises (a timeout, a lost transport)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from benchflow.sandbox._base import ExecRoute
    from benchflow.sandbox.daytona import DaytonaSandbox

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a").write_text("a")
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = __import__("logging").getLogger("test")
    sandbox._sandbox = SimpleNamespace(fs=SimpleNamespace(upload_file=AsyncMock()))
    sent: list[str] = []

    async def sandbox_exec(command, **kwargs):
        sent.append(command)
        if not command.startswith("rm -rf -- "):
            raise TimeoutError("lost")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox._sandbox_exec = sandbox_exec
    sandbox._stage_proof = AsyncMock(return_value="true")
    route = ExecRoute(pin="/usr/bin:/bin", task_root=False, system_shell=True)
    with pytest.raises(Exception, match="lost"):
        await sandbox._sdk_upload_dir(tmp_path / "src", "/target", route=route)
    # Every attempt (the SDK retry's too) removes its own folder.
    folders = [
        call.args[1].rsplit("/", 1)[0]
        for call in sandbox._sandbox.fs.upload_file.await_args_list
    ]
    assert folders and [c for c in sent if c.startswith("rm ")] == [
        f"rm -rf -- {folder}" for folder in folders
    ]
