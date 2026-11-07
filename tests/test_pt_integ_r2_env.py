"""PT-integ round 2: env values never in argv.

PT-INTEG-B-1 (review B, High) at 403505e5: Modal wrote the merged env's
start-up values, PATH and the image PATH into its command text as
``export K=<value>``, and on the protected kit ``with_system_path`` expanded
every kept value as an ``N=value`` argument of ``env -i``, while Modal still
declares ``exec_env_is_private``. Integ c911a719 sent them only as a Modal
Secret. The same ``env -i`` words carried PT-R16's Python search data and
the route's proxy values (D925 2).

These tests run the real text with the kit's paths mapped onto a local
folder whose ``env`` records its argv.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._runtime_tree import KIT_BIN, KIT_SH
from benchflow.sandbox import _base

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/bin/python3"),
    reason="the kit stand-in's env logs its argv with /usr/bin/python3",
)

SECRETS = {
    "JUDGE_API_KEY": "sk-judge-secret-0001",
    "LD_LIBRARY_PATH": "/opt/secret-ld-0002/lib",
    "PYTHONPATH": "/opt/secret-pythonpath-0003",
    "PATH": "/opt/secret-path-0004/bin:/usr/bin:/bin",
    "_BENCHFLOW_IMAGE_PATH": "/opt/secret-image-path-0005/bin:/usr/bin",
    "https_proxy": "http://user:secret-proxy-0006@proxy.invalid:3128",
    "ODD_VALUE": 'it\'s "odd"\n$(touch /tmp/never-0007) `false` \\ end\n',
}


def _secret_words() -> list[str]:
    return [
        w for v in SECRETS.values() for w in v.split() if "secret" in w or "0007" in w
    ]


@pytest.fixture
def kit(tmp_path: Path) -> SimpleNamespace:
    """A stand-in for the protected kit's bin: the system's sh, cat, tr and
    bash, and an ``env`` that writes each argv it gets to a log."""
    bin_dir = tmp_path / "kit-bin"
    bin_dir.mkdir()
    for name, target in (
        ("sh", "/bin/sh"),
        ("cat", "/bin/cat"),
        ("tr", "/usr/bin/tr"),
        ("bash", "/bin/bash"),
    ):
        (bin_dir / name).symlink_to(target)
    log = tmp_path / "env-argv.jsonl"
    (bin_dir / "env").write_text(
        "#!/bin/sh\n"
        f'/usr/bin/python3 -I -c \'import json,sys; open(sys.argv[1],"a").write(json.dumps(sys.argv[2:])+"\\n")\' {shlex.quote(str(log))} "$@"\n'
        'exec /usr/bin/env "$@"\n'
    )
    (bin_dir / "env").chmod(0o755)
    return SimpleNamespace(bin=str(bin_dir), log=log)


def _local(text: str, kit: SimpleNamespace) -> str:
    return text.replace(KIT_BIN, kit.bin)


def _env_argvs(kit: SimpleNamespace) -> list[list[str]]:
    if not kit.log.exists():
        return []
    return [json.loads(line) for line in kit.log.read_text().splitlines()]


DUMP = "/usr/bin/python3 -I -c " + shlex.quote(
    "import json,os; print(json.dumps(dict(os.environ)))"
)


def test_with_system_path_carries_kept_values_on_a_pipe(kit, tmp_path):
    """Fails on 403505e5: ``env -i ${N+"N=$N"}`` put every kept value, the
    image PATH and the Python search data in env's argv."""
    pin = f"{KIT_BIN}:/usr/bin:/bin"
    keep = ["JUDGE_API_KEY", "ODD_VALUE", "LD_LIBRARY_PATH", "https_proxy"]
    text = _local(_base.with_system_path(DUMP, pin, keep=keep), kit)
    outer = {k: v for k, v in SECRETS.items() if k != "PATH"}
    outer["PATH"] = "/usr/bin:/bin"
    done = subprocess.run(
        ["/bin/sh", "-c", text], env=outer, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    argvs = _env_argvs(kit)
    assert argvs, "the kit's env was not used"
    for argv in argvs:
        for word in _secret_words():
            assert all(word not in arg for arg in argv), (word, argv)
    seen = json.loads(done.stdout)
    for name in ("JUDGE_API_KEY", "ODD_VALUE", "https_proxy", "_BENCHFLOW_IMAGE_PATH"):
        assert seen[name] == SECRETS[name], name
    # PT-R16's search data: kept as data, the original unset by the route.
    assert seen["_BENCHFLOW_IMAGE_PYTHONPATH"] == SECRETS["PYTHONPATH"]
    assert "PYTHONPATH" not in seen
    # Route-owned names are never carried; PATH and HOME are the route's.
    assert "LD_LIBRARY_PATH" not in seen
    assert seen["PATH"] == _local(pin, kit)
    assert seen["HOME"] == _base.ROOT_HOME
    assert not Path("/tmp/never-0007").exists()


def _modal(monkeypatch) -> tuple[object, list]:
    import modal

    from benchflow.sandbox.modal_impl import ModalSandbox

    monkeypatch.setattr(modal.Secret, "from_dict", lambda values: dict(values))
    box = ModalSandbox.__new__(ModalSandbox)
    calls: list = []

    async def send(*argv, **kwargs):
        calls.append((list(argv), kwargs))
        process = MagicMock()
        process.stdout.read.aio = AsyncMock(return_value="")
        process.stderr.read.aio = AsyncMock(return_value="")
        process.wait.aio = AsyncMock(return_value=0)
        return process

    box._sandbox = MagicMock()
    box._sandbox.exec.aio = send
    return box, calls


ROUTES = {
    "kit-plumbing": _base.ExecRoute(
        pin=f"{KIT_BIN}:{_base.SYSTEM_PATH}",
        task_root=False,
        system_shell=True,
        explicit_root=True,
        shell=KIT_SH,
        bash=f"{KIT_BIN}/bash",
        tools=KIT_BIN,
        kit=True,
    ),
    "kit-task-root": _base.ExecRoute(
        pin=None,
        task_root=True,
        explicit_root=True,
        shell=KIT_SH,
        bash=f"{KIT_BIN}/bash",
        tools=KIT_BIN,
        kit=True,
    ),
    "plain-plumbing": _base.SYSTEM_SHELL_ROUTE,
    "plain-task": _base.ExecRoute(pin=None, task_root=False, root=False),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_modal_never_writes_an_env_value_into_argv(route, monkeypatch):
    """PT-INTEG-B-1, fails on 403505e5 (``export K=<value>`` and ``env -i
    N=value``): every value reaches Modal only in the Secret, and the
    command text names them."""
    box, calls = _modal(monkeypatch)
    await box._dispatch_exec(
        "true", route=ROUTES[route], agent_bootstrap=False, env=dict(SECRETS)
    )
    ((argv, kwargs),) = calls
    for word in _secret_words():
        assert all(word not in arg for arg in argv), (word, route)
    secret = {k: v for s in kwargs["secrets"] for k, v in s.items()}
    for value in SECRETS.values():
        assert value in secret.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["kit-plumbing", "plain-task"])
async def test_modal_command_still_gets_its_values(route, kit, monkeypatch):
    """The values come back by name: the command sees the caller's PATH, the
    credential and the loader value (task route) as before."""
    box, calls = _modal(monkeypatch)
    await box._dispatch_exec(
        DUMP, route=ROUTES[route], agent_bootstrap=False, env=dict(SECRETS)
    )
    ((argv, kwargs),) = calls
    env = {"PATH": "/usr/bin:/bin"}
    for values in kwargs["secrets"]:
        env.update(values)
    done = subprocess.run(
        [_local(arg, kit) for arg in argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout.splitlines()[-1])
    assert seen["JUDGE_API_KEY"] == SECRETS["JUDGE_API_KEY"]
    assert seen["ODD_VALUE"] == SECRETS["ODD_VALUE"]
    assert seen["PATH"] == SECRETS["PATH"]
    if route == "plain-task":
        assert seen["LD_LIBRARY_PATH"] == SECRETS["LD_LIBRARY_PATH"]
    assert not any(k.startswith("_BF_MODAL_ENV_") for k in seen)
    for argv_ in _env_argvs(kit):
        for word in _secret_words():
            assert all(word not in arg for arg in argv_), (word, argv_)
