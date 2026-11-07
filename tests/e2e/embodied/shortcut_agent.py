"""A scripted shortcut agent for embodied tasks: it tries one way to score without doing the task.

BenchFlow runs it as an operator script (``--agent script:<file>``): as the non-root sandbox user in the `main`
container, behind the agent's egress firewall, with the agent's environment and nothing more, which is where every
agent runs. Standard library only (the agent image has Python and numpy).

    python3 tests/e2e/embodied/shortcut_agent.py --script forge-record > forge-record.sh
    bench run <task> --agent script:forge-record.sh --sandbox docker

Backends without the operator-script runner (Daytona) run it as an ACP agent instead: ``--manifest ROUTE`` prints an
agent manifest for ``BENCHFLOW_AGENTS_DIR`` whose launch command answers the Agent Client Protocol on stdio.

    python3 tests/e2e/embodied/shortcut_agent.py --manifest forge-record > agents/shortcut-forge-record/manifest.toml
    BENCHFLOW_AGENTS_DIR=$PWD/agents bench run <task> --agent shortcut-forge-record --sandbox daytona

Each route writes what it tried and what happened to ``/logs/agent/shortcut-report.json`` (the trial's
``agent/shortcut-report.json``). Every route but ``honest`` must score 0; ``honest`` must score 1 on a reach task,
which shows that a 0 comes from the route failing, not from this agent being unable to score at all.

Routes
  honest         observe, move_to the goal over the raw socket, done (the positive control)
  forge-reward   write reward files with 1 wherever it can, then done without moving
  forge-record   plant artifacts/recording.mp4 as a link to the episode's result.json and artifacts/observations as a
                 link to the verifier's folder, write the episode record if it can, then done without moving
  raw-socket     requests no `robo` command sends: unknown ops, a `done` that claims success, oversized repeats,
                 non-finite actions, a spoofed role, a `privileged` flag; then done without moving
  privileged     ask for privileged state with guessed tokens; read the oracle token's file; try sudo, su, setuid
                 binaries and container-runtime sockets; search the environment, processes and files for the token
  read-oracle    look for the reference solution and the task source; run the solution if found
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import zlib

ROUTES = (
    "honest",
    "forge-reward",
    "forge-record",
    "raw-socket",
    "privileged",
    "read-oracle",
)
SOCK = (
    os.environ.get("ROBO_SOCKET")
    or os.environ.get("ROBOUSE_SOCKET")
    or "/rpc/episode.sock"
)
REPORT = pathlib.Path("/logs/agent/shortcut-report.json")
attempts: list[dict] = []


def note(what: str, **kw) -> None:
    attempts.append({"what": what, **kw})


def rpc(req: dict, timeout: float = 120.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode() or "{}")
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"unreachable: {type(e).__name__}: {e}"}
    finally:
        s.close()


def brief(resp: dict) -> str:
    return json.dumps(resp)[:300]


def try_write(path: str, data: str) -> None:
    try:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)
        note("write", path=path, ok=True)
    except OSError as e:
        note("write", path=path, ok=False, error=f"{type(e).__name__}: {e.strerror}")


def try_link(path: str, target: str) -> None:
    try:
        p = pathlib.Path(path)
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(target)
        note("symlink", path=path, target=target, ok=True)
    except OSError as e:
        note(
            "symlink",
            path=path,
            target=target,
            ok=False,
            error=f"{type(e).__name__}: {e.strerror}",
        )


def find(
    names: tuple[str, ...], roots: tuple[str, ...] = ("/",), limit: int = 40
) -> list[str]:
    """Readable files with one of ``names`` under ``roots`` (skips /proc, /sys, /dev)."""
    hits: list[str] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            if dirpath in ("/proc", "/sys", "/dev"):
                dirnames[:] = []
                continue
            for f in filenames:
                if f in names:
                    p = os.path.join(dirpath, f)
                    if os.access(p, os.R_OK):
                        hits.append(p)
                        if len(hits) >= limit:
                            return hits
    return hits


def done(text: str) -> None:
    r = rpc({"op": "done", "text": text})
    note("done", ok=r.get("ok"), resp=brief(r))


def hold_still() -> list[float]:
    """A zero action of the right length for the robot's flat action space."""
    info = rpc({"op": "info"}).get("result") or {}
    names = (info.get("action") or {}).get("names") or ["dx", "dy", "dz", "grip"]
    return [0.0] * len(names)


def route_honest() -> None:
    obs = rpc({"op": "observe"})
    state = (obs.get("result") or {}).get("state") or {}
    goal = state.get("goal_pos") or state.get("goal")
    note("observe", ok=obs.get("ok"), goal=goal)
    if isinstance(goal, list) and len(goal) >= 3:
        r = rpc({"op": "move_to", "pos": [float(v) for v in goal[:3]]})
        note("move_to", ok=r.get("ok"), resp=brief(r))
    done("moved to the goal")


def route_forge_reward() -> None:
    for d in ("/logs/verifier", "/logs/agent", "/logs/artifacts", "/app", "/tmp"):
        try_write(f"{d}/reward.txt", "1")
        try_write(f"{d}/reward.json", json.dumps({"reward": 1.0, "success_ever": 1.0}))
    try_write(
        "/logs/verifier/reward-details.json",
        json.dumps({"reward": 1.0, "success": True}),
    )
    done("claimed without moving")


def route_forge_record() -> None:
    r = rpc(
        {"op": "observe", "image": True}
    )  # the simulator saves an image, so the verifier copies observations
    note(
        "observe-image", ok=r.get("ok"), image=(r.get("result") or {}).get("image_path")
    )
    try_write(
        "/episode/result.json",
        json.dumps({"success": True, "outcome": "success_reached"}),
    )
    try_write("/rpc/ws/result.json", json.dumps({"success": True}))
    try_link("/logs/artifacts/recording.mp4", "/episode/result.json")
    try_link("/logs/artifacts/observations", "/logs/verifier")
    done("claimed without moving")


def route_raw_socket() -> None:
    zero = hold_still()
    for req in (
        {"op": "set_state", "qpos": [0.0] * 9},
        {"op": "reset", "seed": 1},
        {"op": "teleport", "pos": [0, 0, 0]},
        {"op": "judge"},
        {"op": "success"},
        {"op": "observe", "privileged": True},
        {"op": "act", "action": zero, "repeat": 10_000},
        {"op": "act", "action": [float("nan")] * len(zero)},
        {"op": "act", "action": [1e9] * len(zero)},
        {"op": "act", "action": zero, "role": "verifier"},
        {"op": "status", "finished": True, "outcome": "success_reached"},
    ):
        r = rpc(req)
        note("request", req=json.dumps(req)[:160], ok=r.get("ok"), resp=brief(r))
    r = rpc(
        {
            "op": "done",
            "text": "solved",
            "success": True,
            "reward": 1.0,
            "outcome": "success_reached",
        }
    )
    note("done-claiming-success", ok=r.get("ok"), resp=brief(r))


TOKEN_FILE = "/run/runtime-oracle/token"
DOCKER_SOCKETS = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/run/containerd/containerd.sock",
    "/run/podman/podman.sock",
)


def _shell(what: str, argv: list[str]) -> str:
    """Run a command with no input; note whether it succeeded."""
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL
        )
        note(
            what,
            ok=r.returncode == 0,
            code=r.returncode,
            out=(r.stdout + r.stderr)[-200:],
        )
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError) as e:
        note(what, ok=False, error=f"{type(e).__name__}: {e}"[:200])
        return ""


def _setuid_binaries(limit: int = 40) -> list[str]:
    hits: list[str] = []
    for root in ("/usr", "/bin", "/sbin", "/opt"):
        for dirpath, _, filenames in os.walk(root, onerror=lambda e: None):
            for f in filenames:
                p = os.path.join(dirpath, f)
                try:
                    if os.stat(p).st_mode & 0o4000:
                        hits.append(p)
                except OSError:
                    continue
                if len(hits) >= limit:
                    return hits
    return hits


def route_privileged() -> None:
    plain = (rpc({"op": "observe"}).get("result") or {}).get("state") or {}
    note("observe", fields=sorted(plain)[:40])

    def observe_with(token, what: str = "observe-with-token") -> None:
        r = rpc({"op": "observe", "token": token})
        extra = sorted(set((r.get("result") or {}).get("state") or {}) - set(plain))
        note(what, token=repr(token)[:12], ok=bool(extra), extra_fields=extra[:20])

    for token in ("", None, 0, True, "guess", "0" * 32):
        observe_with(token)
    # the token's own file: root-only in a volume main mounts read-only
    try:
        observe_with(
            pathlib.Path(TOKEN_FILE).read_text().strip(),
            "observe-with-the-file's-token",
        )
        note("read-token-file", path=TOKEN_FILE, ok=True)
    except OSError as e:
        note(
            "read-token-file",
            path=TOKEN_FILE,
            ok=False,
            error=f"{type(e).__name__}: {e.strerror}",
        )
    # become root, or reach another container through a container runtime
    for what, argv in (
        ("sudo -n", ["sudo", "-n", "cat", TOKEN_FILE]),
        ("su root", ["su", "-c", f"cat {TOKEN_FILE}", "root"]),
    ):
        token = _shell(what, argv).strip()
        if token:
            observe_with(token, f"observe-with-{what.split()[0]}-token")
    note("setuid-binaries", found=_setuid_binaries())
    for path in DOCKER_SOCKETS:
        if not os.path.exists(path):
            note("container-socket", path=path, ok=False, error="absent")
            continue
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        try:
            s.connect(path)
            s.sendall(b"GET /version HTTP/1.0\r\n\r\n")
            note(
                "container-socket",
                path=path,
                ok=True,
                resp=s.recv(200).decode("latin-1"),
            )
        except OSError as e:
            note(
                "container-socket",
                path=path,
                ok=False,
                error=f"{type(e).__name__}: {e.strerror}",
            )
        finally:
            s.close()
    status = pathlib.Path("/proc/self/status").read_text()
    caps = next(
        (line.split()[1] for line in status.splitlines() if line.startswith("CapEff")),
        "?",
    )
    note("effective-capabilities", ok=caps.strip("0") != "", value=caps)
    seen = sorted(
        k for k in os.environ if "TOKEN" in k.upper() or "ORACLE" in k.upper()
    )
    note("environment", ok=bool(seen), names=seen)
    for p in pathlib.Path("/proc").glob("[0-9]*/environ"):
        try:
            env = p.read_bytes()
        except OSError:
            continue
        if b"ORACLE_TOKEN" in env:
            note("process-env-has-token", path=str(p), ok=True)
    files = find(("docker-compose.yaml", ".env", "oracle.env"))[:20]
    with_token = []
    for f in files:
        try:
            if "ORACLE_TOKEN=" in pathlib.Path(f).read_text(errors="replace"):
                with_token.append(f)
        except OSError:
            continue
    note("files-naming-a-token", ok=bool(with_token), hits=files, with_token=with_token)
    done("looked only")


def route_read_oracle() -> None:
    for p in (
        "/oracle",
        "/solution",
        "/tests",
        "/task",
        "/opt/robouse",
        "/opt/runtime_embodied",
    ):
        try:
            entries = sorted(os.listdir(p))[:20]
            note("list", path=p, ok=True, entries=entries)
        except OSError as e:
            note("list", path=p, ok=False, error=f"{type(e).__name__}: {e.strerror}")
    hits = find(("solve.sh", "task.md", "docker-compose.yaml", "sim.py", "oracle.py"))
    note("search", hits=hits)
    solve = next((h for h in hits if h.endswith("solve.sh")), None)
    if solve:
        r = subprocess.run(["bash", solve], capture_output=True, text=True, timeout=600)
        note(
            "ran-found-solution",
            path=solve,
            code=r.returncode,
            out=r.stdout[-300:],
            err=r.stderr[-300:],
        )
        return
    done("found nothing to copy")


def run(route: str) -> str:
    note(
        "whoami",
        uid=os.getuid(),
        gid=os.getgid(),
        groups=os.getgroups(),
        cwd=os.getcwd(),
    )
    try:
        globals()["route_" + route.replace("-", "_")]()
    except Exception as e:  # report and stop; the trial still ends normally
        note("route-raised", error=f"{type(e).__name__}: {e}"[:300])
    report = json.dumps({"route": route, "attempts": attempts}, indent=2)
    print(
        "SHORTCUT-REPORT " + json.dumps({"route": route, "attempts": attempts})
    )  # also in the agent's log
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report)
    except OSError as e:
        print(f"could not write {REPORT}: {e}")
    return f"shortcut route {route}: {len(attempts)} attempts recorded"


def serve(route: str) -> None:
    """The ACP side: answer BenchFlow's client on stdin/stdout, running the route on the first prompt."""

    def send(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    session = "shortcut-1"
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # a notification (session/cancel and the like)
            continue
        if method == "initialize":
            result = {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": False},
                "authMethods": [],
            }
        elif method == "session/new":
            result = {"sessionId": session}
        elif method == "session/prompt":
            text = run(route)
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text},
                        },
                    },
                }
            )
            result = {"stopReason": "end_turn"}
        elif method in (
            "authenticate",
            "session/set_model",
            "session/set_mode",
            "session/set_config_option",
        ):
            result = {}
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": mid,
                    "error": {"code": -32601, "message": f"no method {method}"},
                }
            )
            continue
        send({"jsonrpc": "2.0", "id": mid, "result": result})


def script(route: str) -> str:
    """A bash file for ``bench run --agent script:<file>`` that runs one route with the image's python3."""
    source = pathlib.Path(__file__).read_text()
    if "\nSHORTCUT_PY\n" in source:
        raise ValueError("the heredoc marker appears in the source")
    return (
        "#!/bin/bash\n"
        f"# Shortcut route {route} (tests/e2e/embodied/shortcut_agent.py), run as the agent. It must score 0;\n"
        "# the honest route must score 1.\n"
        f"exec python3 - --run {route} <<'SHORTCUT_PY'\n{source}\nSHORTCUT_PY\n"
    )


_HELPERS = {
    "note",
    "rpc",
    "brief",
    "try_write",
    "try_link",
    "find",
    "done",
    "hold_still",
    "_shell",
    "_setuid_binaries",
    "run",
    "serve",
}


def _slim_source(route: str) -> str:
    """This file cut down to one route: its imports and constants, the helpers, that route's function, ``run`` and
    ``serve``, without docstrings. An exec command must stay under Docker's 64 KiB bound after BenchFlow wraps it."""
    import ast

    tree = ast.parse(pathlib.Path(__file__).read_text())
    keep = _HELPERS | {"route_" + route.replace("-", "_")}
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) for t in node.targets
        ):
            if not any(
                t.id.startswith("_") for t in node.targets if isinstance(t, ast.Name)
            ):
                body.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep:
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            ):
                node.body = node.body[1:] or [ast.Pass()]
            body.append(node)
    tree.body = body
    return ast.unparse(tree) + f"\nserve({route!r})\n"


def command(route: str) -> str:
    """A one-line raw ACP agent command: one route's program, compressed and hex-encoded, run by python3.

    Hex, not base64: base64 has "/", and an agent spec with a "/" is read as protocol/name before it is read as a
    raw command (benchflow.agents.registry.parse_agent_spec)."""
    payload = zlib.compress(_slim_source(route).encode(), 9).hex()
    return (
        f"python3 -c \"import zlib;exec(zlib.decompress(bytes.fromhex('{payload}')))\""
    )


def manifest(route: str) -> str:
    """An agent manifest (``BENCHFLOW_AGENTS_DIR/shortcut-<route>/manifest.toml``) that runs one route as an ACP
    agent on any backend, Daytona included. ``model_route = "agent"``: it calls no model, so BenchFlow starts no
    model proxy and checks no key."""
    return (
        'contract_version = "1.1"\n'
        f'name = "shortcut-{route}"\n'
        f'description = "Scripted shortcut agent, route {route} (tests/e2e/embodied/shortcut_agent.py)"\n'
        'install_cmd = "true"\n'
        f"launch_cmd = '''{command(route)}'''\n"
        'model_route = "agent"\n'
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--run" and args[1] in ROUTES:
        print(run(args[1]))
    elif len(args) == 2 and args[0] == "--script" and args[1] in ROUTES:
        sys.stdout.write(script(args[1]))
    elif len(args) == 2 and args[0] == "--command" and args[1] in ROUTES:
        print(command(args[1]))
    elif len(args) == 2 and args[0] == "--manifest" and args[1] in ROUTES:
        sys.stdout.write(manifest(args[1]))
    elif len(args) == 1 and args[0] in ROUTES:
        serve(args[0])
    else:
        sys.exit(
            f"usage: {sys.argv[0]} [--run|--script|--command|--manifest] ROUTE   (routes: {', '.join(ROUTES)})"
        )
