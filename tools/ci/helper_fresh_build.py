"""Build BenchFlow's metadata firewall helper fresh, as a new Docker host does.

    python tools/ci/helper_fresh_build.py build OUT
    python tools/ci/helper_fresh_build.py pins OUT

.github/workflows/helper-fresh-build.yml runs both once a day. ``build``
writes the Dockerfile from src/benchflow/sandbox/_metadata_helper.py (the
module's own text, never a copy), builds it with no build cache and the base
pulled again, runs the image's tools (the firewall tools' ``--version``,
python3's ssl, ``openssl version``) and reads which tier of
``HELPER_TIERS`` it installed (``/etc/apk/world``) and the builds it holds
(``apk info -v``). ``pins`` then reads that record.

The two fail for different reasons, and each writes its own heading to the
job summary (``$GITHUB_STEP_SUMMARY``):

- ``build`` fails: the recipe no longer builds, or its tools don't run. A
  host with no helper image yet refuses every Docker start that keeps the
  cloud metadata block.
- ``pins`` fails: the helper still builds, from a later tier, but Alpine no
  longer serves the exact builds in ``HELPER_PACKAGES``. Fresh hosts are
  served; the pins should move to the builds it lists.

Nothing is tagged or pushed, and the image is removed at the end. Writes
OUT/helper.Dockerfile, OUT/build.log, OUT/packages.txt and OUT/result.json.
Exit 0: passed; 1: failed as above; 2: it could not run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from benchflow.sandbox._metadata_helper import (
    HELPER_DOCKERFILE,
    HELPER_LABEL,
    HELPER_PACKAGES,
    HELPER_RECIPE,
    HELPER_RECIPE_LABEL,
    HELPER_TIERS,
)

BUILD_SEC = 900
RUN_SEC = 120
# The tools the helper's scripts run, python3's ssl (the verifier's egress
# proxy) and the readlink the recipe links where the custody walk looks.
SMOKE = (
    "set -e; iptables --version; iptables-legacy --version; nft --version; "
    "python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)'; openssl version; "
    'test "$(readlink /bin/readlink)" = /bin/busybox'
)
_INSTALLED = re.compile(r"(.+)-([^-]+-r\d+)")
_NAMES = frozenset(pin.split("=", 1)[0] for pin in HELPER_PACKAGES)


def docker(
    args: list[str], *, stdin: str | None = None, timeout: float = RUN_SEC
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _in_image(image: str, command: str) -> subprocess.CompletedProcess[str]:
    """``command`` in the image's shell, with no network."""
    shell = ["run", "--rm", "--network", "none", "--entrypoint", "/bin/sh"]
    return docker([*shell, image, "-c", command])


def summary(text: str) -> None:
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def tier_of(world: list[str]) -> int | None:
    """Which tier of HELPER_TIERS (from 1) the image's /etc/apk/world names
    for the helper's packages (the base's own are there too). ``apk add``
    writes the world only when it installs, so a tier that failed leaves no
    line in it."""
    found = sorted(
        line.strip()
        for line in world
        if re.split(r"[=~<>]", line.strip(), maxsplit=1)[0] in _NAMES
    )
    for n, tier in enumerate(HELPER_TIERS, 1):
        if found == sorted(tier):
            return n
    return None


def installed(lines: list[str]) -> dict[str, str]:
    """``apk info -v`` lines (``name-version-rN``) as {name: version-rN}."""
    found = {}
    for line in lines:
        match = _INSTALLED.fullmatch(line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _table(builds: dict[str, str]) -> list[str]:
    rows = ["| package | pinned | installed |", "|---|---|---|"]
    for pin in HELPER_PACKAGES:
        name, version = pin.split("=", 1)
        got = builds.get(name, "missing")
        rows.append(f"| {name} | {version} | {got}{'' if got == version else ' *'} |")
    return rows


def _fence(text: str) -> str:
    return "```\n" + text.replace("```", "'''") + "\n```"


def build(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    dockerfile = out / "helper.Dockerfile"
    dockerfile.write_text(HELPER_DOCKERFILE)
    digest = hashlib.sha256(dockerfile.read_bytes()).hexdigest()[:16]
    if digest != HELPER_RECIPE:
        print(f"the Dockerfile written ({digest}) is not the recipe ({HELPER_RECIPE})")
        return 2
    daemon = docker(["info", "--format", "{{.ID}} {{.ServerVersion}} {{.Images}}"])
    print(f"daemon (id, version, images): {daemon.stdout.strip() or daemon.stderr}")
    print(f"recipe {HELPER_RECIPE}:\n{HELPER_DOCKERFILE}")
    iidfile = out / "image-id"
    iidfile.unlink(missing_ok=True)
    try:
        built = docker(
            [
                "build",
                "--no-cache",
                "--pull",
                "--progress=plain",
                "--iidfile",
                str(iidfile),
                "--label",
                f"{HELPER_LABEL}=metadata",
                "--label",
                f"{HELPER_RECIPE_LABEL}={HELPER_RECIPE}",
                "-",
            ],
            stdin=HELPER_DOCKERFILE,
            timeout=BUILD_SEC,
        )
        log = built.stdout + built.stderr
        ok = built.returncode == 0
    except subprocess.TimeoutExpired as exc:
        log, ok = (
            f"{exc.stdout or ''}{exc.stderr or ''}\nno result in {BUILD_SEC} s",
            False,
        )
    (out / "build.log").write_text(log)
    print(log)
    image = iidfile.read_text().strip() if ok and iidfile.exists() else ""
    result: dict = {"recipe": HELPER_RECIPE, "built": bool(image), "ok": False}
    if not image:
        (out / "result.json").write_text(json.dumps(result, indent=1))
        tail = "\n".join(log.strip().splitlines()[-25:])
        summary(
            "## Fresh hosts can't build the metadata helper\n\n"
            f"Recipe `{HELPER_RECIPE}` did not build from Alpine's repository "
            "at any tier. Until it builds, a Docker host with no helper image "
            "refuses every start that keeps the cloud metadata block. The "
            f"build's last lines:\n\n{_fence(tail)}"
        )
        return 1
    try:
        smoke = _in_image(image, SMOKE)
        world = _in_image(image, "cat /etc/apk/world")
        info = _in_image(image, "apk info -v")
    finally:
        docker(["image", "rm", "--force", image])
    print(smoke.stdout + smoke.stderr)
    (out / "packages.txt").write_text(info.stdout)
    builds = installed(info.stdout.splitlines())
    tier = tier_of(world.stdout.splitlines()) if world.returncode == 0 else None
    result.update(
        image=image,
        tier=tier,
        world=world.stdout.split(),
        packages=builds,
        smoke=smoke.returncode == 0,
        ok=smoke.returncode == 0 and tier is not None and info.returncode == 0,
    )
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("installed:\n" + info.stdout)
    if not result["ok"]:
        summary(
            "## The metadata helper builds, but its image is not what the "
            "recipe says\n\n"
            f"Recipe `{HELPER_RECIPE}`, image `{image}`: the tools "
            f"{'ran' if smoke.returncode == 0 else 'failed'}, and its "
            f"/etc/apk/world names {'tier ' + str(tier) if tier else 'no tier'} "
            "of HELPER_TIERS. Fresh hosts run this image.\n\n"
            f"{_fence((smoke.stdout + smoke.stderr).strip())}\n\n"
            + "\n".join(_table(builds))
        )
        return 1
    summary(
        "## The metadata helper builds fresh\n\n"
        f"Recipe `{HELPER_RECIPE}`, image `{image}`, from tier {tier} of "
        f"{len(HELPER_TIERS)}.\n\n{_fence(smoke.stdout.strip())}\n\n"
        + "\n".join(_table(builds))
    )
    return 0


def pins(out: Path) -> int:
    try:
        result = json.loads((out / "result.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"no build record in {out}: {exc}")
        return 2
    if not result.get("ok") or result.get("recipe") != HELPER_RECIPE:
        print(f"no passing build of recipe {HELPER_RECIPE} recorded in {out}")
        return 2
    builds = result["packages"]
    if result["tier"] == 1:
        summary(
            "## The exact pins still resolve\n\n"
            "Every package in HELPER_PACKAGES installed at its pinned build."
        )
        return 0
    moved = [
        f'    "{name}={builds.get(name, "?")}",'
        for name, version in (pin.split("=", 1) for pin in HELPER_PACKAGES)
        if builds.get(name) != version
    ]
    summary(
        "## The exact pins no longer resolve; fresh hosts are served by the "
        "fallback\n\n"
        f"Fresh hosts still build the helper, from tier {result['tier']} of "
        "HELPER_TIERS, but Alpine no longer serves every build pinned in "
        "HELPER_PACKAGES (src/benchflow/sandbox/_metadata_helper.py). Move "
        "these pins to the builds served now, and follow the release "
        "checklist's helper image steps for the new recipe:\n\n"
        + _fence("\n".join(moved))
        + "\n\n"
        + "\n".join(_table(builds))
    )
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("build", "pins"):
        print(__doc__)
        return 2
    out = Path(argv[2])
    return build(out) if argv[1] == "build" else pins(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
