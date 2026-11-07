"""The two-layer parity check of the direct Docker path (PERF-E).

Input: the recording proxy's logs (``tests/_docker_api_recorder.py``) of trials
of one task, each a ``bench eval run`` through the proxy. For each pair:

1. **Request bodies.** The NetworkCreate bodies and the ContainerCreate bodies
   (the stopped prototype, then main with the bootstrap) are compared after
   masking names, IDs and per-trial label values, under the order
   rules (:func:`order_rules`: Env, Binds and Mounts as multisets unless a
   key or destination repeats) and nothing else. Also every call that
   changes one of the project's own containers or networks (:func:`effects`:
   start, kill and its signal, remove and its force/volumes query, network
   remove, each with its reply status), each named by its target's role
   (prototype, main, network) and compared in order (B-R3-L1). Left out, as
   Compose's mechanism rather than an effect: a stop answered 304 (the
   prototype was never started), a rename (names are masked), exec
   creation (BenchFlow's own hardening steps, the same code on both paths).
2. **Inspect.** The proxy's snapshot of main (and of its network) taken when
   the start call returned, under :func:`normalise_inspect` (D962's three
   rules and D965's), after the same masking; everything else exact.

Masked: the trial's own strings (project name, job and trial folders, the
private temporary files), every container, network, exec and image-layer ID,
timestamps, PIDs, the per-container MAC address, sandbox keys, and the values
of ``benchflow.process``, ``com.docker.compose.config-hash``,
``com.docker.compose.project.config_files`` and
``com.docker.compose.project.working_dir`` (they hash or list per-trial paths;
the hashes themselves are checked against Compose in
``tests/test_direct_docker.py``).

Usage: ``python -I tools/perf/direct_docker_diff.py A.jsonl B.jsonl`` prints a
JSON verdict; exit 0 only when both layers are identical.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchflow.sandbox._direct_docker import (
    differences,
    normalise_inspect,
    order_rules,
)

_PER_TRIAL_LABELS = (
    "benchflow.process",
    "com.docker.compose.config-hash",
    "com.docker.compose.project.config_files",
    "com.docker.compose.project.working_dir",
)
_TIME_KEYS = {"Created", "StartedAt", "FinishedAt"}
_RANDOM_KEYS = {"Pid", "MacAddress", "SandboxKey", "SandboxID", "EndpointID"}
_HEX_ID = re.compile(r"\b[0-9a-f]{64}\b|\b[0-9a-f]{12}\b")
_TMP = re.compile(r"/tmp/benchflow-[A-Za-z0-9_.-]+")
# A trial's job and trial folder names (bench's timestamp and random suffix),
# and the matrix's own folder per arm.
_JOB_TRIAL = re.compile(
    r"/jobs/\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}/[^/:]+__[0-9a-f]{8}"
)
# The private name of an image BenchFlow builds for the trial
# (_compose_runtime.private_image_repository: "b" + 26 base32 letters).
_BUILT_IMAGE = re.compile(r"\bb[a-z2-7]{26}\b")
_ARM = re.compile(r"/(compose-a|compose-b|direct)/")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def creates(log: list[dict], kind: str) -> list[dict]:
    return [
        r
        for r in log
        if r.get("kind") == "request"
        and r["method"] == "POST"
        and re.search(rf"/{kind}/create(\?|$)", r["path"])
        and r.get("status") in (200, 201)
    ]


def project_of(log: list[dict]) -> str:
    for r in creates(log, "networks"):
        return r["body"]["Name"].removesuffix("_default")
    for r in creates(log, "containers"):
        return r["body"]["Labels"]["com.docker.compose.project"]
    raise ValueError("no project in log")


def snapshot(log: list[dict], what: str, project: str) -> dict | None:
    """The last snapshot of the trial's main (or network) before teardown:
    the start of main with the bootstrap."""
    found = None
    for r in log:
        if r.get("kind") != "snapshot" or r.get("what") != what:
            continue
        body = r.get("body") or {}
        labels = body.get("Labels") or (body.get("Config") or {}).get("Labels") or {}
        if labels.get("com.docker.compose.project") == project:
            found = body
    return found


def mask(value: Any, project: str, trial_root: str | None) -> Any:
    def text(s: str) -> str:
        if trial_root:
            s = s.replace(trial_root, "<trial>")
        s = s.replace(project, "<project>")
        s = _TMP.sub("<tmp>", s)
        s = _JOB_TRIAL.sub("/jobs/<job>/<trial>", s)
        s = _ARM.sub("/<arm>/", s)
        s = _BUILT_IMAGE.sub("<built-image>", s)
        return _HEX_ID.sub("<id>", s)

    def walk(v: Any, key: str = "") -> Any:
        if isinstance(v, dict):
            out = {}
            for k, item in v.items():
                k2 = text(k)
                if k in _TIME_KEYS or k in _RANDOM_KEYS or k in _PER_TRIAL_LABELS:
                    out[k2] = "<masked>" if item not in (None, "", 0) else item
                else:
                    out[k2] = walk(item, k)
            return out
        if isinstance(v, list):
            return [walk(item, key) for item in v]
        if isinstance(v, str):
            return text(v)
        return v

    return walk(value)


def trial_root(log: list[dict]) -> str | None:
    """The trial's folder: the common parent of main's bind sources."""
    for r in creates(log, "containers"):
        binds = (r["body"].get("HostConfig") or {}).get("Binds") or []
        sources = [b.split(":")[0] for b in binds]
        if sources:
            parts = [p.split("/") for p in sources]
            common = []
            for column in zip(*parts, strict=False):
                if len(set(column)) != 1:
                    break
                common.append(column[0])
            return "/".join(common)
    return None


_OBJECT = re.compile(r"^/v[0-9.]+/(containers|networks)/([^/?]+)(/[a-z]+)?(?:\?(.*))?$")


def _roles(log: list[dict]) -> dict[str, str]:
    """Each object the trial created, by every way a call can name it: the
    first container is the stopped prototype, the second main (on both
    paths), and the network is the network (B-R3-L1)."""
    roles: dict[str, str] = {}
    for kind in ("containers", "networks"):
        for index, r in enumerate(creates(log, kind)):
            role = (
                "network"
                if kind == "networks"
                else ("prototype", "main")[min(index, 1)]
            )
            reply_id = (r.get("reply") or {}).get("Id")
            if isinstance(reply_id, str):
                roles[reply_id] = role
            name = (r.get("body") or {}).get("Name") or dict(
                parse_qsl(r["path"].partition("?")[2])
            ).get("name")
            if isinstance(name, str):
                roles[name] = role
    return roles


def _role_of(ref: str, roles: dict[str, str], project: str) -> str | None:
    if ref in roles:
        return roles[ref]
    if re.fullmatch(r"[0-9a-f]{4,64}", ref):
        hits = {role for key, role in roles.items() if key.startswith(ref)}
        if len(hits) == 1:
            return hits.pop()
        return "an ambiguous prefix" if hits else None
    if project in ref:
        # Compose's temporary name for main (<id12>_<project>-main-1).
        return (
            "main" if ref.endswith(f"{project}-main-1") else "another of the project's"
        )
    return None


def effects(log: list[dict], project: str) -> list[str]:
    """What the calls did to this project's own containers and networks, in
    order, each with its target's role (prototype, main, network)."""
    roles = _roles(log)
    out: list[str] = []
    for r in log:
        if r.get("kind") != "request" or r["method"] not in ("POST", "DELETE"):
            continue
        m = _OBJECT.match(r["path"])
        if not m:
            continue
        kind, ref, verb, query = m.group(1), unquote(m.group(2)), m.group(3), m.group(4)
        role = _role_of(ref, roles, project)
        if role is None:
            continue
        q = dict(parse_qsl(query or ""))
        status = r.get("status")
        if r["method"] == "DELETE" and verb is None:
            what = "remove " + " ".join(
                f"{k}={q[k]}" for k in sorted(q) if k in ("force", "v")
            )
        elif kind == "containers" and verb == "/start":
            what = "start"
        elif kind == "containers" and verb == "/kill":
            what = f"kill {q.get('signal', 'SIGKILL')}"
        elif (kind == "containers" and verb == "/stop" and status == 304) or verb in (
            "/rename",
            "/exec",
            "/create",
        ):
            continue
        else:
            what = f"{verb} " + " ".join(f"{k}={q[k]}" for k in sorted(q))
        out.append(f"{role}: {what.strip()} -> {status}")
    return out


def compare(a_path: Path, b_path: Path) -> dict:
    a_log, b_log = load(a_path), load(b_path)
    sides = []
    for log in (a_log, b_log):
        project = project_of(log)
        sides.append((log, project, trial_root(log)))
    verdict: dict[str, Any] = {"a": str(a_path), "b": str(b_path)}
    layer1: list[str] = []
    for kind in ("networks", "containers"):
        bodies = [
            [mask(order_rules(r["body"]), p, t) for r in creates(log, kind)]
            for log, p, t in sides
        ]
        if len(bodies[0]) != len(bodies[1]):
            layer1.append(f"{kind}: {len(bodies[0])} creates != {len(bodies[1])}")
            continue
        for i, (x, y) in enumerate(zip(*bodies, strict=True)):
            layer1 += [f"{kind}[{i}]{d}" for d in differences(x, y)]
    starts = [
        sum(
            1
            for r in log
            if r.get("kind") == "request"
            and r["method"] == "POST"
            and r["path"].endswith("/start")
            and "/containers/" in r["path"]
            and r.get("status") == 204
        )
        for log, _, _ in sides
    ]
    verdict["container_starts"] = starts
    layer2: list[str] = []
    for what in ("container", "network"):
        snaps = [snapshot(log, what, p) for log, p, _ in sides]
        if snaps[0] is None and snaps[1] is None:
            continue  # neither made one (network_mode none: no network)
        if snaps[0] is None or snaps[1] is None:
            layer2.append(f"{what}: snapshot missing ({[s is None for s in snaps]})")
            continue
        normal = [
            mask(normalise_inspect(s) if what == "container" else s, p, t)
            for s, (_, p, t) in zip(snaps, sides, strict=True)
            if s is not None
        ]
        layer2 += [f"{what}{d}" for d in differences(*normal)]
    done = [effects(log, p) for log, p, _ in sides]
    verdict["effects"] = done
    if done[0] != done[1]:
        layer1.append(f"effects: {done[0]} != {done[1]}")
    verdict["layer1"] = layer1
    verdict["layer2"] = layer2
    verdict["identical"] = not layer1 and not layer2
    return verdict


def main(argv: list[str]) -> int:
    verdict = compare(Path(argv[1]), Path(argv[2]))
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["identical"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
