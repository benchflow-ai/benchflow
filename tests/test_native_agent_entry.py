"""Guards K1-R16-1/2 at c2e6b630: the native session barrier comes first.

Replay actual provider dispatch in a disposable Linux container; only the
remote transport is mocked. Markers never require sudo, surviving processes,
credentials or an untrusted image.
"""

from __future__ import annotations

import json
import os
import pwd
import resource
import shlex
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import _base
from benchflow.sandbox.agent_exec import agent_exec_call


@pytest.fixture
def tmp_path(tmp_path):
    if os.getuid() != 0:
        yield tmp_path
        return
    # Real root-to-nobody checks need a directly traversable test folder.
    # Never change the permissions of pytest's ancestors or the host temp root.
    with tempfile.TemporaryDirectory(prefix="k1-native-entry-", dir="/tmp") as folder:
        path = Path(folder)
        path.chmod(0o777)
        yield path


async def dispatch(
    kind,
    command,
    environment,
    tmp_path,
    monkeypatch,
    *,
    user=None,
    capture=False,
    persistent=None,
):
    """Replay actual provider dispatch; mock only remote process/file transport."""
    user = user or ("nobody" if os.getuid() == 0 else pwd.getpwuid(os.getuid()).pw_name)
    image_env = {"PATH": "/usr/bin:/bin", "K1_IMAGE_VALUE": "image"}
    calls = []
    uploads = []

    def local_path(path):
        if os.getuid() == 0:
            return path
        return path.replace("/.runtime-agent-env-", str(tmp_path / "env-"))

    def run(argv, env=None, stdin=None):
        argv = [local_path(arg) for arg in argv]
        if os.getuid() != 0:
            for account in (user, "root"):
                # PT's route also starts root's su with bash (C8).
                for shell in ("/bin/sh", "/bin/bash"):
                    argv[-1] = argv[-1].replace(
                        f"/bin/su {account} -s {shell} -c", f"{shell} -c"
                    )
        calls.append((argv, env or image_env))
        result = subprocess.run(
            argv, env=env or image_env, input=stdin, text=True, capture_output=True
        )
        return _base.ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    async def upload(source, target, **kwargs):
        Path(local_path(target)).write_bytes(Path(source).read_bytes())

    if kind == "modal":
        import modal

        from benchflow.sandbox.modal_impl import ModalSandbox

        def secret(values):
            # Modal checks these limits remotely; model them at that boundary.
            assert all(len(key) <= 16384 and key.isidentifier() for key in values)
            assert all(len(value) <= 32768 for value in values.values())
            return values

        monkeypatch.setattr(modal.Secret, "from_dict", secret)
        backend = ModalSandbox.__new__(ModalSandbox)

        async def send(*argv, **kwargs):
            env = dict(image_env)
            for values in kwargs["secrets"]:
                env.update(values)
            result = run(list(argv), env)
            process = MagicMock()
            process.stdout.read.aio = AsyncMock(return_value=result.stdout)
            process.stderr.read.aio = AsyncMock(return_value=result.stderr)
            process.wait.aio = AsyncMock(return_value=result.return_code)
            return process

        backend._sandbox = MagicMock()
        backend._sandbox.exec.aio = send

        async def copy_from_local(source, path):
            # Sandbox.filesystem: Modal refuses the legacy Sandbox.open.
            # The upload's folder is the fresh root-only one.
            folder = Path(local_path(path)).parent.stat()
            assert folder.st_mode & 0o777 == 0o700
            assert folder.st_uid == os.getuid()
            with open(local_path(path), "wb") as stream:
                stream.write(Path(source).read_bytes())
            uploads.append((path, stream))

        backend._sandbox.filesystem.copy_from_local.aio = AsyncMock(
            side_effect=copy_from_local
        )
    elif kind == "agentcore":
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        backend = AgentCoreSandbox.__new__(AgentCoreSandbox)
        backend.runtime_arn = backend.runtime_session_id = "dummy"
        backend._invoke_command = lambda payload, timeout: run(shlex.split(payload))

        async def stage(environment, **kwargs):
            path = tmp_path / "staged.sh"
            path.write_text(_base.env_file_body(environment))
            path.chmod(0o644)
            return str(path)

        backend._sealed = SimpleNamespace(stage_env=stage)
    elif kind == "apple":
        from benchflow.sandbox import apple_container

        backend = apple_container.AppleContainerSandbox.__new__(
            apple_container.AppleContainerSandbox
        )
        backend._container_name = "dummy"

        async def send(*args, **kwargs):
            argv = list(args[args.index("dummy") + 1 :])
            if os.getuid() == 0 and user != "root":
                argv = [
                    "/usr/bin/setpriv",
                    f"--reuid={user}",
                    f"--regid={pwd.getpwnam(str(user)).pw_gid}",
                    "--init-groups",
                    "--",
                    *argv,
                ]
            data = kwargs.get("stdin_data")
            return run(argv, stdin=data.decode() if data is not None else None)

        monkeypatch.setattr(apple_container, "_run_cli", send)
    else:
        from benchflow.sandbox import daytona

        daytona._load_daytona_sdk()
        backend = daytona.DaytonaSandbox.__new__(daytona.DaytonaSandbox)
        # The direct strategy, as a native Daytona sandbox has (exec_route).
        backend._strategy = daytona._DaytonaDirect(backend)
        backend._sandbox = MagicMock()
        backend._sandbox.process.create_session = AsyncMock()
        backend._sandbox.process.delete_session = AsyncMock()
        responses = {}

        async def send(session, request, **kwargs):
            responses[session] = run(["/bin/sh", "-c", request.command])
            return SimpleNamespace(cmd_id="dummy")

        backend._sandbox.process.execute_session_command = send

        async def poll(session, *args, **kwargs):
            return responses[session]

        backend._poll_response = poll

        async def sync(session, payload, timeout):
            return run(["/bin/sh", "-c", payload])

        backend._sync_exec = sync
        monkeypatch.setattr(
            daytona, "_sync_exec_eligible", lambda _: kind == "daytona-sync"
        )
        backend._exec_counts = lambda: SimpleNamespace(
            exec_calls=0,
            transient_exec_calls=0,
            exec_seconds=0.0,
            transient_exec_seconds=0.0,
        )
    if kind != "modal":
        backend.upload_file = upload
    backend._persistent_env = persistent or {}
    backend.default_user = None
    shell = "/bin/sh" if kind == "apple" else "/bin/bash"
    wrapped, selected = agent_exec_call(
        command, user, sandbox=SimpleNamespace(agent_exec_shell=shell)
    )
    if kind.startswith("daytona"):
        # Daytona's exec takes its route first (C8).
        result = await backend._sandbox_exec(
            wrapped,
            env=environment,
            user=selected,
            timeout_sec=5,
            route=await backend.exec_route(selected),
        )
    else:
        result = await backend.exec(wrapped, env=environment, user=selected)
    if capture:
        return calls
    return SimpleNamespace(
        returncode=result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
        dispatches=calls,
        uploads=uploads,
    )


def probe(marker):
    return (
        "if /bin/grep -q 'NoNewPrivs:[[:space:]]*0' /proc/self/status; then "
        f"printf before-barrier > {shlex.quote(str(marker))}; fi\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daytona", "daytona-sync", "agentcore", "apple"])
async def test_backend_environment_file_is_sourced_inside_barrier(
    kind, tmp_path, monkeypatch
):
    """R16-1: model a watcher appending code to the agent-owned env file."""
    marker = tmp_path / "before-barrier"
    original = _base.env_file_body
    monkeypatch.setattr(
        _base, "env_file_body", lambda env: original(env) + probe(marker)
    )
    result = await dispatch(kind, "/bin/echo ok", {"A": "1"}, tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n", result.stderr
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["daytona", "daytona-sync", "modal", "agentcore", "apple"]
)
@pytest.mark.parametrize("field", ["PATH", "BASH_ENV", "ENV", "LD_PRELOAD", "LD_AUDIT"])
async def test_caller_environment_cannot_run_before_native_barrier(
    kind, field, tmp_path, monkeypatch
):
    """R16-1/2: exercise each backend with executable caller environment."""
    marker = tmp_path / "before-barrier"
    if field == "PATH":
        tools = tmp_path / "bin"
        tools.mkdir()
        for name in ("id", "env", "su", "getent", "timeout", "base64", "rm"):
            tool = tools / name
            tool.write_text(f'#!/bin/sh\n{probe(marker)}exec /usr/bin/{name} "$@"\n')
            tool.chmod(0o755)
        value = f"{tools}:/usr/bin:/bin"
    elif field in ("BASH_ENV", "ENV"):
        startup = tmp_path / "startup.sh"
        startup.write_text(probe(marker))
        value = str(startup)
    else:
        library = tmp_path / "hook.so"
        source = tmp_path / "hook.c"
        source.write_text(
            "#include <stdio.h>\n#include <string.h>\n"
            "unsigned int la_version(unsigned int v) { return 1; }\n"
            "__attribute__((constructor)) static void init(void) {\n"
            'FILE *s=fopen("/proc/self/status","r"); char line[256];\n'
            "if (!s) return; while(fgets(line,sizeof line,s)) {\n"
            'int n; if (sscanf(line,"NoNewPrivs:%d", &n)==1 && !n) {\n'
            f'FILE *m=fopen({json.dumps(str(marker))},"w"); '
            'if(m) { fputs("before-barrier",m); fclose(m); }\n'
            "} } fclose(s); }\n"
        )
        subprocess.run(
            ["/usr/bin/cc", "-shared", "-fPIC", str(source), "-o", str(library)],
            check=True,
        )
        value = str(library)
    result = await dispatch(kind, "/bin/echo ok", {field: value}, tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n", result.stderr
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["daytona", "daytona-sync", "modal", "agentcore", "apple"]
)
async def test_native_entry_preserves_account_group_shell_and_environment(
    kind, tmp_path, monkeypatch
):
    """R16-1: the earlier nobody:nogroup and backend-shell contract stays intact."""
    selected = "nobody" if os.getuid() == 0 else pwd.getpwuid(os.getuid()).pw_name
    account = pwd.getpwnam(selected)
    body = (
        'printf "%s\\n" "$K1_VALUE" "$K1_IMAGE_VALUE"; '
        "/usr/bin/id -u; /usr/bin/id -g; "
        "/bin/grep '^NoNewPrivs:' /proc/self/status; "
        + ("printf sh" if kind == "apple" else "a=(bash); printf '%s' ${a[0]}")
    )
    result = await dispatch(
        kind, body, {"K1_VALUE": "x' $()\nvalue"}, tmp_path, monkeypatch, user=selected
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout
        == f"x' $()\nvalue\nimage\n{account.pw_uid}\n{account.pw_gid}\nNoNewPrivs:\t1\n"
        + ("sh" if kind == "apple" else "bash")
    )


@pytest.mark.asyncio
async def test_modal_transport_keeps_large_values_and_transport_like_names(
    tmp_path, monkeypatch
):
    """R16-2: inert transport neither truncates data nor treats caller keys as state."""
    environment = {f"K1_LARGE_{i}": "v" * 40000 for i in range(4)}
    environment["_BF_AGENT_ENV_0"] = "caller-owned"
    body = '/usr/bin/python3 -I -c \'import os; print([len(os.environ["K1_LARGE_"+str(i)]) for i in range(4)]); print(os.environ["_BF_AGENT_ENV_0"])\''
    result = await dispatch("modal", body, environment, tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "[40000, 40000, 40000, 40000]\ncaller-owned\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["-K1_CALLER", "--ignore-environment", "K1.WITH.DOTS"])
async def test_modal_transport_never_interprets_keys_as_env_options(
    key, tmp_path, monkeypatch
):
    """R16-2 transport: arbitrary provider env keys remain literal assignments."""
    result = await dispatch(
        "modal", "/bin/echo ok", {key: "caller-value"}, tmp_path, monkeypatch
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        *_base.SYSTEM_SHELL_UNSET_VARS,
        "PATH",
        "BASHOPTS",
        "SHELLOPTS",
        "BASH_FUNC_probe%%",
        "_BF_AGENT_ENV_0",
    ],
)
async def test_modal_first_process_receives_only_inert_caller_keys(
    field, tmp_path, monkeypatch
):
    """R16-2: all keys follow one transport, including future loader spellings."""
    value = "dummy ' $()\nvalue"
    calls = await dispatch(
        "modal",
        ":",
        {field: value},
        tmp_path,
        monkeypatch,
        capture=True,
        persistent={field: value},
    )
    # PT's route first looks the account's uid up: root's own plumbing exec,
    # outside the agent barrier, with no loader value (C8).
    probes = [c for c in calls if c[0][0] != "/usr/bin/setpriv"]
    assert len(probes) <= 1 and all("id -u" in argv[-1] for argv, _ in probes)
    assert all(
        not env.get(name) for _, env in probes for name in _base.SYSTEM_SHELL_UNSET_VARS
    )
    calls = [c for c in calls if c[0][0] == "/usr/bin/setpriv"]
    assert len(calls) == 3  # folder creation, dispatch, final cleanup
    for argv, env in calls:
        # PT's route passes the loader names empty (C8): inert, no value.
        assert all(
            not env.pop(name)
            for name in list(env)
            if name in _base.SYSTEM_SHELL_UNSET_VARS
        )
        assert env == {"PATH": "/usr/bin:/bin", "K1_IMAGE_VALUE": "image"}
        assert argv[:5] == ["/usr/bin/setpriv", "--no-new-privs", "--", "/bin/sh", "-c"]
        assert value not in argv[-1], "caller values must stay in the file API"


@pytest.mark.asyncio
async def test_modal_loader_never_executes_as_root(tmp_path, monkeypatch):
    """R16-2 at c2e6b630 and 7a2387ac: real root-to-nobody container replay."""
    marker = tmp_path / "root-loader"
    source = tmp_path / "root-hook.c"
    library = tmp_path / "root-hook.so"
    # Let the selected account traverse the folder, while a root constructor
    # could create the marker even before su or the shell's first instruction.
    source.write_text(
        "#include <stdio.h>\n#include <unistd.h>\n"
        "__attribute__((constructor)) static void init(void) {\n"
        "if (geteuid()==0) { "
        f'FILE *m=fopen({json.dumps(str(marker))},"w"); '
        'if(m) { fputs("root-loader",m); fclose(m); } } }\n'
    )
    subprocess.run(
        ["/usr/bin/cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True
    )
    user = "nobody" if os.getuid() == 0 else pwd.getpwuid(os.getuid()).pw_name
    result = await dispatch(
        "modal",
        "/usr/bin/id -u",
        {"LD_PRELOAD": str(library)},
        tmp_path,
        monkeypatch,
        user=user,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{pwd.getpwnam(user).pw_uid}\n"
    assert not marker.exists(), "caller loader ran as root before account selection"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["modal", "daytona", "daytona-sync", "agentcore", "apple"]
)
@pytest.mark.parametrize("count,width", [(500, 400), (3000, 1)])
async def test_r17_large_environment_starts_and_restores_exactly(
    kind, count, width, tmp_path, monkeypatch
):
    """Guards K1-R17-1 introduced at 255a1ec0 and present at 02648360."""
    import hashlib

    environment = {f"CFG_{i:04d}": "x" * width for i in range(count)}
    marker = tmp_path / "must-not-evaluate"
    environment.update(
        {
            "K1_ODD": f"'\"\n\t\r\x01 $({{ printf BAD > {marker}; }}) `false` \\ é🙂\n\n",
            "K1_ODD_DOLLAR": f"$(printf BAD > {marker})",
            "K1_ODD_BACKTICK": f"`printf BAD > {marker}`",
            "_BF_AGENT_ENV_0": "caller-owned",
            "_BF_AGENT_DATA_0": "caller-owned-too",
        }
    )
    expected = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
    body = "/usr/bin/python3 -I -c " + shlex.quote(
        "import hashlib,json,os; "
        "e={k:v for k,v in os.environ.items() if k.startswith(('CFG_', 'K1_ODD', '_BF_AGENT_'))}; "
        "print(hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest())"
    )
    result = await dispatch(kind, body, environment, tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected + "\n"
    assert not marker.exists()
    # A fixed bound, independent of the env (whose values never reach argv,
    # below). PT's route adds a constant prefix to an agent command on
    # Daytona (restore_image_env, about 4.4 KB once quoted; C8), so the bound
    # is 32 KiB, still a quarter of Linux's 128 KiB per-argument limit
    # (flagged for K1's reviewer).
    assert (
        max(len(arg.encode()) for argv, _ in result.dispatches for arg in argv) < 32768
    )
    assert all("x" * 100 not in arg for argv, _ in result.dispatches for arg in argv)
    for path, stream in result.uploads:
        assert stream.closed
        assert not Path(
            path
            if os.getuid() == 0
            else path.replace("/.runtime-agent-env-", str(tmp_path / "env-"))
        ).parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["modal", "daytona", "daytona-sync", "agentcore", "apple"]
)
async def test_r17_odd_names_reach_shell_as_literal_environment(
    kind, tmp_path, monkeypatch
):
    """02648360 loses non-shell identifiers; inspect the shell's initial environ.

    Shells may discard non-identifiers for children. The transport must deliver
    them intact to the shell, including names that resemble options or syntax.
    """
    import base64

    environment = {
        "-option": "one",
        "dot.name": "two",
        "quote'\"": "three",
        "line\nname": "four",
        "name\udcff": "byte-name",
        "é": "five",
        "$(false)": "six",
        "K1_BYTES": "\udcff\t\n\x01end\n",
    }
    # Keep the shell alive after the child: /proc/PPID/environ then observes
    # the exact environment received by the backend shell, before its parsing.
    body = (
        "/usr/bin/python3 -I -c "
        + shlex.quote(
            "import base64,os; print(base64.b64encode(open('/proc/%d/environ'%os.getppid(),'rb').read()).decode())"
        )
        + "; :"
    )
    result = await dispatch(kind, body, environment, tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    actual = dict(
        item.split(b"=", 1)
        for item in base64.b64decode(result.stdout).split(b"\0")
        if item
    )
    for key, value in environment.items():
        assert actual[os.fsencode(key)] == os.fsencode(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["modal", "daytona", "daytona-sync", "agentcore", "apple"]
)
@pytest.mark.parametrize(
    "count,value,stack_mb",
    [(500, "x" * 400, 8), (48, "é" * 20000, 8), (500, "é" * 3000, 32)],
    ids=["200KB", "review-1920KB", "3MB"],
)
async def test_r18_aggregate_environment_uses_no_exec_transport_space(
    kind, count, value, stack_mb, tmp_path, monkeypatch
):
    """Guards K1-R18-1 at 5f1525ba: encoding must not consume ARG_MAX.

    The 3 MB case uses a 32 MiB stack (6 MiB ARG_MAX), so the original
    environment is valid while its escaped/base64 transport would not be.
    The review's exact case fits the usual 8 MiB stack (2 MiB ARG_MAX).
    """
    import hashlib

    environment = {f"CFG_{i:04d}": value for i in range(count)}
    expected = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
    body = "/usr/bin/python3 -I -c " + shlex.quote(
        "import hashlib,json,os; "
        "e={k:v for k,v in os.environ.items() if k.startswith('CFG_')}; "
        "print(hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest())"
    )
    old_stack = resource.getrlimit(resource.RLIMIT_STACK)
    try:
        resource.setrlimit(
            resource.RLIMIT_STACK, (stack_mb * 1024 * 1024, old_stack[1])
        )
        result = await dispatch(kind, body, environment, tmp_path, monkeypatch)
    finally:
        resource.setrlimit(resource.RLIMIT_STACK, old_stack)
    if result.returncode and stack_mb == 32:
        # Some trusted PAM configurations reset su's stack to 8 MiB. The
        # original 3 MB environment then exceeds the final exec limit too;
        # refuse it with sizes and alternatives, before the command runs.
        assert result.returncode == 97, result.stderr
        assert "environment is 300" in result.stderr
        assert "exec limit is 2097152 bytes" in result.stderr
        assert "smaller agent environment" in result.stderr
        assert "uploaded files" in result.stderr
        assert result.stdout == ""
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected + "\n"
    # Include all setup/cleanup invocations, not just the agent command.
    assert all(
        sum(len(os.fsencode(arg)) + 1 for arg in argv)
        + sum(len(os.fsencode(k + "=" + v)) + 1 for k, v in env.items())
        < 16384
        for argv, env in result.dispatches
    )


@pytest.mark.asyncio
async def test_r18_modal_upload_failure_does_not_run_command(tmp_path, monkeypatch):
    """Guards K1-R18-1 at 5f1525ba: failed file API handoff cleans up."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    paths = []

    async def fail_upload(self, source, target, **kwargs):
        path = Path(
            target
            if os.getuid() == 0
            else target.replace("/.runtime-agent-env-", str(tmp_path / "env-"))
        )
        paths.append(path)
        path.write_bytes(b"incomplete dummy payload")
        raise RuntimeError("dummy file API failure")

    monkeypatch.setattr(ModalSandbox, "upload_file", fail_upload)
    marker = tmp_path / "must-not-run"
    with pytest.raises(RuntimeError, match="dummy file API failure"):
        await dispatch(
            "modal",
            f"echo BAD > {shlex.quote(str(marker))}",
            {"CFG": "dummy"},
            tmp_path,
            monkeypatch,
        )
    assert len(paths) == 1
    assert not paths[0].parent.exists()
    assert not marker.exists()
