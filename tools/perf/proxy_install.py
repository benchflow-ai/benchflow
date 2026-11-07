"""B9: the LiteLLM proxy's install, uv against pip, cold, in a fresh container.

Each run starts a fresh container from the image named by digest (no mounts,
empty caches), adds Python 3 and ``python3-venv`` by apt (not timed), copies
in the host's ``uv`` binary for the uv runs (not timed), then times:

- ``uv``: ``uv venv`` plus ``uv pip install`` of the proxy's requirements;
- ``pip``: ``python3 -m venv`` plus ``pip install`` of the same.

The requirements are BenchFlow's own (``LITELLM_VERSION_SPEC`` and
``boto3>=1.40``); ``--vertex`` adds the Vertex extra the proxy installs for
Vertex routes. The two tools alternate run by run; the container is removed
after each run.

    uv run python -m tools.perf.proxy_install \\
        --image ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 \\
        --runs 3 --out $W/perf/b9.csv
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from benchflow.providers.litellm_runtime import LITELLM_VERSION_SPEC
from tools.perf._common import (
    append_rows,
    image_digest,
    output,
    provenance,
    report_provenance,
)

FIELDS = ("tool", "round", "vertex", "seconds", "exit_code")
VERTEX_EXTRA = "google-cloud-aiplatform>=1.133.0,<2.0"
TOOLS = ("uv", "pip")


def requirements(vertex: bool) -> list[str]:
    reqs = [LITELLM_VERSION_SPEC, "boto3>=1.40"]
    return [*reqs, VERTEX_EXTRA] if vertex else reqs


def timed_command(tool: str, vertex: bool) -> str:
    reqs = " ".join(shlex.quote(r) for r in requirements(vertex))
    if tool == "uv":
        return (
            "set -e; /usr/local/bin/uv venv --python /usr/bin/python3 /opt/proxy-venv "
            f"&& /usr/local/bin/uv pip install --python /opt/proxy-venv/bin/python -q {reqs}"
        )
    return (
        "set -e; python3 -m venv /opt/proxy-venv "
        f"&& /opt/proxy-venv/bin/pip install -q {reqs}"
    )


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


def run_once(image: str, tool: str, vertex: bool, uv_binary: str) -> tuple[float, int]:
    name = f"perf-b9-{uuid.uuid4().hex[:10]}"
    _docker("run", "-d", "--name", name, image, "sleep", "3600")
    try:
        _docker(
            "exec",
            name,
            "sh",
            "-c",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install "
            "-y -qq --no-install-recommends python3 python3-venv ca-certificates "
            ">/dev/null",
        )
        if tool == "uv":
            _docker("cp", uv_binary, f"{name}:/usr/local/bin/uv")
        started = time.perf_counter()
        done = _docker(
            "exec", name, "sh", "-c", timed_command(tool, vertex), check=False
        )
        return time.perf_counter() - started, done.returncode
    finally:
        # The container this run made.
        _docker("rm", "-f", name, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, help="an image named by digest")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--vertex", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if "@sha256:" not in args.image:
        raise SystemExit("--image must be named by digest (name@sha256:...)")
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        raise SystemExit("uv is not on PATH (the pinned uv the README installs)")
    _docker("pull", "-q", args.image)
    report_provenance(
        provenance(
            "proxy_install",
            images={"base": image_digest(args.image) or args.image},
            extra={
                "requirements": requirements(args.vertex),
                "uv_in_container": output([uv_binary, "--version"]),
            },
        ),
        args.out,
    )
    for round_index in range(args.runs):
        order = TOOLS if round_index % 2 == 0 else tuple(reversed(TOOLS))
        for tool in order:
            seconds, code = run_once(args.image, tool, args.vertex, uv_binary)
            append_rows(
                args.out,
                FIELDS,
                [
                    {
                        "tool": tool,
                        "round": round_index,
                        "vertex": int(args.vertex),
                        "seconds": seconds,
                        "exit_code": code,
                    }
                ],
            )
            print(f"{tool} round {round_index}: {seconds:.1f} s (exit {code})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
