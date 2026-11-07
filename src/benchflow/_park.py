"""A parked call: the local runtime's sandbox prepared for the next identical run.

One-trial-per-call harnesses (run-08's) pay a sandbox start on every CLI
call. With ``bench daemon start --prepare N``, when an attached run of one
fresh single-task Docker job ends well, the runtime runs the same command
again (same argv, environment, folder, umask, streams and installation) in
a new worker, with its output held in a private spool. That worker goes
through the normal CLI and job path; the job's own warm pool
(``benchflow._warm_pool``) prepares its one trial by the same
``setup()``, ``start()`` and ``install_agent()`` calls a cold start makes,
and the worker stops there, before the trial is taken, holding the
prepared sandbox (``hold``).

When an identical call arrives, the runtime asks the parked worker to check
that nothing it read changed (every file it opened for reading and every
folder it listed, outside its own job, temporary folders and the installed
code, which the runtime's identity covers), then hands it the caller's
standard streams and connection. The spooled output is written out, and the
job continues exactly as a cold one: the pool's ``take`` re-reads the task's
digest and image, and the trial runs. Otherwise, or after the idle time
limit, the worker removes the sandbox, the trial folder and the job folder it
made, and ends without a trace in the jobs folder.

What a prepared call keeps from its preparation: the trial's start time, as
any warm-pool trial's (its sandbox was made then), and the job's printed
lines, which keep their order. The job's elapsed time leaves out the wait.

Only a call that can be prepared exactly is parked: one task, Docker,
``--concurrency 1``, a new job folder (a resumed job is never prepared), no
budget, the stock trial lifecycle, the warm pool not switched off, and a
task whose sandbox shares nothing with the host (D928/D939's list).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, NoReturn, cast

# This process's place in the local runtime: None (a plain CLI), "run" (an
# attached call) or "parked" (a prepared call waiting for its caller).
_role: str | None = None
_parkable = False
_channel: socket.socket | None = None
_caller: socket.socket | None = None
_spool: tuple[Path, Path] | None = None
_job_dir: Path | None = None
_jobs_dir_made: Path | None = None
_waited = 0.0
_started_ns = 0
_recording = False
_reads: dict[str, None] = {}
_listed: dict[str, None] = {}
# Paths only probed (os.stat, os.lstat, os.access: exists(), is_file() and
# the like), which no audit event reports.
_probed: dict[str, None] = {}

# The images the task's sandbox comes from (references), and their IDs when
# this call began; checked again at the gate and before a hand-over.
_image_refs_used: list[str] = []
_images_at_start: dict[str, str | None] = {}
# Why the last snapshot() could not be taken (for the runtime's log).
_refusal = ""

# In a prepared job's folder while it waits: "resume the latest job" skips
# it (it is no job yet); removed when a caller takes it.
PREPARED_MARKER = ".benchflow-prepared"
_written: set[str] = set()
_own_roots: set[str] = set()

# Above these, the call is not parked: checking it again would cost more
# than a cold start saves.
LIMIT_BYTES = 256 * 1024 * 1024
LIMIT_FILES = 20_000


def role() -> str | None:
    return _role


def parkable() -> bool:
    """Whether this attached call's job could have been prepared ahead."""
    return _parkable


def admission_wait() -> float:
    """Seconds this job waited, prepared, for its caller (0 when cold)."""
    return _waited


def caller() -> socket.socket | None:
    """The admitted caller's connection, once a parked call is taken."""
    return _caller


def begin_run() -> None:
    global _role
    _role = "run"


def begin_parked(channel: socket.socket, spool: tuple[Path, Path]) -> None:
    """Make this process a parked call; record what it reads from now on."""
    global _role, _channel, _spool, _started_ns, _recording
    _role, _channel, _spool = "parked", channel, spool
    _started_ns = time.time_ns()
    _recording = True
    sys.addaudithook(_audit)
    _record_probes()


def _record_probes() -> None:
    """Record the paths Python code probes, as reads are: a file that did not
    exist while the call was prepared (a new .env) must stop the hand-over."""
    for name in ("stat", "lstat", "access"):
        original = getattr(os, name)

        @functools.wraps(original)
        def probe(
            path: Any, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            if _recording and kwargs.get("dir_fd") is None:
                found = _name(path)
                if found is not None:
                    _probed.setdefault(found, None)
            return _original(path, *args, **kwargs)

        # Code asks os which functions take dir_fd or follow_symlinks
        # (the safe writes do): the probe answers as the original.
        for supported in (
            os.supports_dir_fd,
            os.supports_fd,
            os.supports_follow_symlinks,
            os.supports_effective_ids,
        ):
            if original in supported:
                supported.add(probe)
        setattr(os, name, probe)


def _name(path: Any) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    try:
        return os.path.abspath(os.fsdecode(path))
    except (TypeError, ValueError):
        return None


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _recording:
        return
    if event == "open":
        path, mode, flags = args
        name = _name(path)
        if name is None:
            return
        if mode is not None:
            write = any(letter in mode for letter in "wax+")
        else:
            write = bool(
                flags
                & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        if write:
            _written.add(name)
        else:
            _reads.setdefault(name, None)
    elif event in ("os.listdir", "os.scandir"):
        name = _name(args[0] if args and args[0] is not None else ".")
        if name is not None:
            _listed.setdefault(name, None)
    elif event in ("tempfile.mkdtemp", "tempfile.mkstemp"):
        name = _name(args[0])
        if name is not None:
            _own_roots.add(name)


def _excluded_roots() -> tuple[str, ...]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/proc", "/sys", "/dev"}
    # The installed code: the runtime's identity covers it.
    roots.add(str(Path(__file__).resolve().parent))
    roots.update(entry for entry in sys.path if "site-packages" in entry)
    if _spool is not None:
        roots.add(str(_spool[0].parent))
    if _job_dir is not None:
        roots.add(str(_job_dir))
    roots.update(_own_roots)
    return tuple(os.path.join(root, "") for root in roots if root)


def _excluded(name: str, roots: tuple[str, ...]) -> bool:
    if name in _written:
        return True
    # The jobs folder's listing picks the job's name: a prepared call's job is
    # new by construction, and its own folder appears there.
    if _job_dir is not None and name == str(_job_dir.parent):
        return True
    return os.path.join(name, "").startswith(roots)


def _docker_config() -> str:
    folder = os.environ.get("DOCKER_CONFIG") or os.path.join(
        os.path.expanduser("~"), ".docker"
    )
    return os.path.abspath(os.path.join(folder, "config.json"))


def _dotenv() -> str:
    """The .env BenchFlow reads (benchflow._dotenv), resolved in this folder."""
    return os.path.abspath(os.environ.get("BENCHFLOW_DOTENV_PATH", ".env"))


def _kind(name: str) -> Any:
    """What a probe of *name* can see: absent, a folder, a file and its size."""
    try:
        info = os.stat(name)
    except OSError:
        return None
    if stat.S_ISDIR(info.st_mode):
        return "dir"
    if stat.S_ISREG(info.st_mode):
        return ["file", info.st_size]
    return stat.S_IFMT(info.st_mode)


def _digest(name: str) -> str:
    digest = hashlib.sha256()
    with open(name, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(*, since_ns: int | None = None) -> dict[str, Any] | None:
    """Every input this call read, as it is now; None when one cannot be
    checked again (too large, or changed by someone else after it began)."""
    roots = _excluded_roots()
    # Docker's own client settings (its current context) are read by the
    # docker CLI, which no audit hook sees.
    # The .env is read only after an exists() probe: a new one changes the
    # agent's credentials and route (benchflow._dotenv).
    names = [*_reads, _docker_config(), _dotenv()]
    files: dict[str, Any] = {}
    total = 0
    for name in names:
        if name in files or _excluded(name, roots):
            continue
        try:
            info = os.stat(name)
        except OSError:
            files[name] = None
            continue
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            files[name] = stat.S_IFMT(info.st_mode)
            continue
        if since_ns is not None and info.st_mtime_ns > since_ns:
            return _refuse(f"{name} changed while the call was prepared")
        total += info.st_size
        if total > LIMIT_BYTES or len(files) > LIMIT_FILES:
            return _refuse("too many or too large inputs to check again")
        try:
            files[name] = [info.st_size, _digest(name)]
        except OSError:
            return _refuse(f"{name} could not be read again")
    folders: dict[str, Any] = {}
    for name in _listed:
        if _excluded(name, roots):
            continue
        try:
            folders[name] = sorted(os.listdir(name))
        except OSError:
            folders[name] = None
    probed: dict[str, Any] = {}
    for name in _probed:
        if name in files or _excluded(name, roots):
            continue
        probed[name] = _kind(name)
    try:
        images = image_ids(_image_refs_used)
    except (OSError, subprocess.SubprocessError) as exc:
        return _refuse(f"images could not be resolved: {exc}")
    return {"files": files, "folders": folders, "probed": probed, "images": images}


def _refuse(reason: str) -> None:
    global _refusal
    _refusal = reason
    return None


def difference(before: dict[str, Any], after: dict[str, Any] | None) -> str:
    """What changed between two snapshots (for the runtime's log)."""
    if after is None:
        return _refusal or "inputs could not be checked again"
    for section, label in (
        ("images", "an image the sandbox comes from"),
        ("files", "a file read while preparing"),
        ("probed", "a path probed while preparing"),
        ("folders", "a folder listed while preparing"),
    ):
        old, new = before.get(section, {}), after.get(section, {})
        for name in sorted(set(old) | set(new)):
            if old.get(name) != new.get(name):
                return f"{label} changed: {name}"
    return ""


# Dockerfile instructions, as BuildKit reads them: continuation lines joined.
_ARG = re.compile(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", re.IGNORECASE)
_FROM = re.compile(r"^\s*FROM\s+(.*)$", re.IGNORECASE)
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _dockerfile_bases(path: Path, args: dict[str, str]) -> list[str] | None:
    """The images a Dockerfile builds from (FROM, after ARG defaults and the
    build's own args), or None when one cannot be read exactly."""
    try:
        text = path.read_text()
    except OSError:
        return None
    lines = text.replace("\\\n", " ").splitlines()
    values: dict[str, str] = {}
    stages: set[str] = set()
    bases: list[str] = []
    seen_from = False
    for line in lines:
        arg = _ARG.match(line)
        if arg and not seen_from:
            default = (arg.group(2) or "").strip().strip("\"'")
            values[arg.group(1)] = args.get(arg.group(1), default)
            continue
        found = _FROM.match(line)
        if not found:
            continue
        seen_from = True
        words = [word for word in found.group(1).split() if not word.startswith("--")]
        if not words:
            return None
        ref = _VARIABLE.sub(
            lambda m: values.get(m.group(1) or m.group(2), "\0"), words[0]
        )
        if "\0" in ref or "$" in ref:
            return None
        if len(words) >= 3 and words[1].lower() == "as":
            stages.add(words[2].lower())
        if ref.lower() == "scratch" or ref.lower() in stages:
            continue
        bases.append(ref)
    return bases


def image_refs(evaluation: Any) -> list[str] | None:
    """A fast pre-check only (the hand-over's rebuild decides):
    images the task's sandbox is made from, as a cold start would name it now: the prebuilt reference (after config_override), the base image
    override, each Compose service's image and every FROM of every build.
    None when one cannot be named exactly (the call is then not prepared)."""
    import yaml

    from benchflow.evaluation import _task_image_ref

    cfg = evaluation._config
    task_dir = evaluation._get_task_dirs()[0]
    refs: list[str] = []
    prebuilt = _task_image_ref(task_dir, cfg)
    if prebuilt:
        refs.append(prebuilt)
    if cfg.base_image_override:
        refs.append(cfg.base_image_override)
    env_dir = task_dir / "environment"
    builds: list[tuple[Path, dict[str, str]]] = []
    composed = False
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        path = env_dir / name
        if not path.exists():
            continue
        composed = True
        try:
            document = yaml.safe_load(path.read_text()) or {}
            services = document.get("services") or {}
        except Exception:
            return None
        for service in services.values():
            image = service.get("image")
            if isinstance(image, str):
                if "$" in image:
                    return None
                refs.append(image)
            build = service.get("build")
            if build is None:
                continue
            if isinstance(build, str):
                build = {"context": build}
            context = env_dir / str(build.get("context", "."))
            dockerfile = context / str(build.get("dockerfile", "Dockerfile"))
            raw = build.get("args") or {}
            if isinstance(raw, list):
                raw = dict(item.split("=", 1) for item in raw if "=" in item)
            args = {str(k): str(v) for k, v in raw.items() if v is not None}
            if any("$" in value for value in args.values()):
                return None
            builds.append((dockerfile, args))
    if not prebuilt and not composed and (env_dir / "Dockerfile").exists():
        builds.append((env_dir / "Dockerfile", {}))
    for dockerfile, args in builds:
        bases = _dockerfile_bases(dockerfile, args)
        if bases is None:
            return None
        refs.extend(bases)
    return sorted(set(refs))


def image_ids(refs: list[str]) -> dict[str, str | None]:
    """Each reference's image ID on the Docker daemon now (None: absent)."""
    ids: dict[str, str | None] = {}
    for ref in refs:
        found = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if found.returncode == 0:
            ids[ref] = found.stdout.strip()
        elif "no such" in (found.stderr + found.stdout).lower():
            ids[ref] = None
        else:
            raise subprocess.SubprocessError(found.stderr.strip()[-200:])
    return ids


def eligible(evaluation: Any) -> bool:
    """Whether this job can be prepared ahead exactly (see the module text)."""
    from benchflow._warm_pool import task_shares_nothing, warm_pool_requested
    from benchflow.rollout import Rollout, lifecycle_is_stock

    cfg = evaluation._config
    if not (
        cfg.environment == "docker"
        and cfg.concurrency == 1
        and cfg.job_mode != "sequential-shared"
        and cfg.budget is None
        and cfg.warm_pool != 0
        and warm_pool_requested()
        and evaluation._learner_skills_dir is None
        and evaluation._runs_stock_trials()
        and lifecycle_is_stock(Rollout)
    ):
        return False
    if (evaluation._jobs_dir / evaluation._job_name).exists():
        return False
    tasks = evaluation._get_task_dirs()
    # Whether a cold build would give the same images is decided by
    # repeating it at the hand-over, not by reading Dockerfiles.
    return len(tasks) == 1 and task_shares_nothing(tasks[0])


def note_job(evaluation: Any) -> None:
    """Called by ``Evaluation.run()`` before its job folder exists.

    An attached call records whether it could be prepared; a parked call
    that cannot ends here, having made nothing.
    """
    global _parkable, _job_dir, _jobs_dir_made
    if _role is None:
        return
    ok = _parkable = eligible(evaluation)
    if _role == "run":
        return
    if not ok:
        _tell({"discarded": "this call cannot be prepared ahead"})
        _end_quietly()
    global _image_refs_used, _images_at_start
    _image_refs_used = image_refs(evaluation) or []
    try:
        _images_at_start = image_ids(_image_refs_used)
    except (OSError, subprocess.SubprocessError):
        _tell({"discarded": "images could not be resolved"})
        _end_quietly()
    _job_dir = (evaluation._jobs_dir / evaluation._job_name).absolute()
    jobs_dir = evaluation._jobs_dir.absolute()
    _jobs_dir_made = None if jobs_dir.exists() else jobs_dir
    # Marked before anything else exists in it (see PREPARED_MARKER).
    _job_dir.mkdir(parents=True)
    (_job_dir / PREPARED_MARKER).touch()
    # Recorded by the runtime: what to remove if this process is killed.
    _tell({"job": str(_job_dir), "jobs_dir_made": str(_jobs_dir_made or "")})


def _tell(message: dict[str, Any]) -> None:
    """A best-effort note to the runtime (it may be gone)."""
    from benchflow._daemon_wire import send

    if _channel is not None:
        with contextlib.suppress(OSError):
            send(_channel, message)


def _end_quietly() -> NoReturn:
    atexit._run_exitfuncs()
    os._exit(0)


async def _discard(pool: Any, reason: str) -> NoReturn:
    """Remove everything the parked call made, then end without output."""
    global _recording
    _recording = False
    _tell({"discarded": reason})
    from benchflow.providers.runtime import drain_host_proxies

    with contextlib.suppress(Exception):
        if pool is not None:
            await pool.drain()
    with contextlib.suppress(Exception):
        await drain_host_proxies()
    with contextlib.suppress(Exception):
        from benchflow.task.staging import remove_released

        remove_released()
    job_dir = _job_dir
    # Made by this call (note_job saw it absent), holding no result.
    if (
        job_dir is not None
        and job_dir.is_dir()
        and not any(job_dir.rglob("result.json"))
    ):
        shutil.rmtree(job_dir, ignore_errors=True)
    if _jobs_dir_made is not None:
        with contextlib.suppress(OSError):
            _jobs_dir_made.rmdir()
    # Nobody may be left to remove the held output (the runtime may be gone).
    for path in _spool or ():
        with contextlib.suppress(OSError):
            path.unlink()
    _end_quietly()


def _receive_fds(count: int) -> list[int]:
    from benchflow._daemon_wire import receive_fds

    assert _channel is not None
    return receive_fds(_channel, count)


def _receive() -> Any:
    from benchflow._daemon_wire import receive

    assert _channel is not None
    return receive(_channel)


async def hold(pool: Any) -> None:
    """Prepare the job's one trial, then wait for an identical call.

    Returns once a caller is admitted (the job then runs for it); otherwise
    cleans up and ends the process.
    """
    global _recording, _waited
    from benchflow._daemon_wire import send

    entry = getattr(pool, "_entries", {}).get(0) if pool is not None else None
    if pool is not None and entry is None:
        pool.ready()
        entry = pool._entries.get(0)
    if entry is None:
        await _discard(pool, "no trial to prepare")
    try:
        await entry.task
    except BaseException as exc:
        # A failed preparation is never handed to a caller: it starts cold.
        await _discard(pool, f"preparation failed: {type(exc).__name__}")
    _recording = False
    sandbox = getattr(getattr(entry, "rollout", None), "env", None)
    project = getattr(sandbox, "compose_project_name", None)
    if isinstance(project, str):
        _tell({"project": project})
    baseline = snapshot(since_ns=_started_ns)
    if baseline is None:
        await _discard(pool, difference({}, None))
    if baseline["images"] != _images_at_start:
        await _discard(pool, "an image the sandbox comes from changed while preparing")
    # What a cold call's build would give now, by the cold path's own build
    # the images every service of the resolved Compose model names.
    try:
        refusal = _rebuild_unsupported(sandbox)
        if refusal:
            await _discard(pool, refusal)
        built = await _service_images(sandbox)
    except _Unpreparable as exc:
        await _discard(pool, str(exc))
    except Exception as exc:
        await _discard(
            pool, f"the Compose model could not be resolved: {type(exc).__name__}"
        )
    try:
        assert _channel is not None
        send(_channel, {"parked": True})
    except OSError:
        await _discard(pool, "the local runtime is gone")
    began = time.monotonic()
    while True:
        try:
            message = await asyncio.to_thread(_receive)
        except (OSError, EOFError, ValueError):
            await _discard(pool, "the local runtime is gone")
        action = message.get("action")
        if action == "validate":
            checked = time.monotonic()
            # The fast checks first: inputs read, probed and listed, images.
            now = snapshot()
            changed = difference(baseline, now) if now != baseline else ""
            timing = {"checks_s": round(time.monotonic() - checked, 3)}
            if not changed:
                rebuilt = time.monotonic()
                changed = await _rebuild_changes(sandbox, built)
                timing["rebuild_s"] = round(time.monotonic() - rebuilt, 3)
            try:
                assert _channel is not None
                send(_channel, {"valid": not changed, "reason": changed, **timing})
            except OSError:
                changed = changed or "the local runtime is gone"
            if changed:
                await _discard(pool, changed)
        elif action == "admit":
            fds = await asyncio.to_thread(_receive_fds, 4)
            _waited = time.monotonic() - began
            _admit(fds)
            return
        else:
            await _discard(pool, f"removed by the runtime ({action})")


class _Unpreparable(Exception):
    """This call's sandbox cannot be checked by a rebuild (it stays cold)."""


def _rebuild_unsupported(sandbox: Any) -> str:
    """Why a cold call's build cannot be repeated for this sandbox, or ""."""
    if sandbox is None or not hasattr(sandbox, "_build_task_image"):
        return "this sandbox has no build to repeat"
    if getattr(sandbox, "_verifier_layer", None) is not None:
        return "the verifier image comes from a cached build context"
    if getattr(sandbox, "_recovery_baseline", None) is not None:
        return "a recovered sandbox starts from its original image"
    return ""


async def _image_content(sandbox: Any, ref: str) -> str | None:
    """What a container started from *ref* runs: its filesystem layers and
    configuration (not its ID: a fully cached rebuild of the same inputs gets
    a new creation time, so a new ID, with the same layers and config)."""
    found = await sandbox._run_docker_process(
        [
            "docker",
            *sandbox._docker_compose_global_args(),
            "image",
            "inspect",
            "--format",
            "{{json .RootFS.Layers}} {{json .Config}} {{.Os}} {{.Architecture}}",
            ref,
        ],
        check=False,
        timeout_sec=60,
        what="Docker image inspect",
        separate_stderr=True,
    )
    if found.return_code == 0:
        return hashlib.sha256((found.stdout or "").strip().encode()).hexdigest()
    if "no such" in ((found.stderr or "") + (found.stdout or "")).lower():
        return None
    raise RuntimeError("docker image inspect failed")


async def _service_images(sandbox: Any) -> dict[str, list[Any]]:
    """Each service's image in the resolved Compose model (include, extends,
    additional contexts, inline Dockerfiles and interpolation applied by
    Compose itself) and the content (layers and config) that name has now. A build without the
    cache or a forced pull is never prepared: its result is not repeatable."""
    result = await sandbox._run_docker_compose_command(
        ["config", "--format", "json"], timeout_sec=60, raw_output=True
    )
    config = json.loads(result.stdout or "{}")
    services: Any = config.get("services")
    if not isinstance(services, dict) or not services:
        raise _Unpreparable("Compose resolved no services")
    project = config.get("name") or sandbox.compose_project_name
    images: dict[str, list[Any]] = {}
    for name, spec in sorted(cast("dict[str, Any]", services).items()):
        build = spec.get("build")
        if spec.get("pull_policy") in ("always", "daily", "weekly") or (
            isinstance(build, dict) and (build.get("no_cache") or build.get("pull"))
        ):
            raise _Unpreparable(
                f"service {name} builds without the cache or always pulls"
            )
        ref = spec.get("image") or f"{project}-{name}"
        images[name] = [ref, await _image_content(sandbox, ref)]
    return images


async def _rebuild_changes(sandbox: Any, built: dict[str, list[Any]]) -> str:
    """Run the cold path's build again (cached) and say what it changed: "" when
    every service's image is the one this call was prepared from."""
    try:
        # As DockerSandbox.start(): a build unless main is prebuilt without
        # Compose (the verifier-context path is never prepared).
        if not sandbox._use_prebuilt or sandbox._uses_compose:
            # `compose build` creates, replaces and removes no container, but
            # the sandbox forgets its main container after any compose command
            # it does not list as keeping them: every later exec of the trial
            # would then go through `docker compose exec` (twice as slow).
            known = getattr(sandbox, "_main_container", None)
            await sandbox._build_task_image()
            if known is not None and getattr(sandbox, "_main_container", None) is None:
                sandbox._main_container = known
        now = await _service_images(sandbox)
    except _Unpreparable as exc:
        return str(exc)
    except Exception as exc:
        return f"the cold build could not be repeated: {type(exc).__name__}"
    for name in sorted(set(built) | set(now)):
        if built.get(name) != now.get(name) or now[name][1] is None:
            before, after = (
                built.get(name) or [None, None],
                now.get(name) or [None, None],
            )
            return (
                f"a cold build would give service {name} another image "
                f"({before[0]} {str(before[1])[:19]} -> {after[0]} {str(after[1])[:19]})"
            )
    return ""


def _admit(fds: list[int]) -> None:
    """Take the caller's streams and connection; write out the held output."""
    global _caller
    from benchflow._daemon import _watch_caller
    from benchflow._daemon_wire import send

    if _job_dir is not None:
        (_job_dir / PREPARED_MARKER).unlink(missing_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    assert _spool is not None
    held = [path.read_bytes() for path in _spool]
    for target, fd in enumerate(fds[:3]):
        os.dup2(fd, target)
        os.close(fd)
    _caller = socket.socket(fileno=fds[3])
    _caller.settimeout(None)
    send(_caller, {"pid": os.getpid()})
    threading.Thread(target=_watch_caller, args=(_caller,), daemon=True).start()
    for target, data in zip((1, 2), held, strict=True):
        view = memoryview(data)
        while view:
            view = view[os.write(target, view) :]
    for path in _spool:
        path.unlink(missing_ok=True)
