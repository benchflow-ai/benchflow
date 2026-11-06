"""PERF-H: the kit's build does less work (its tool copies one
walk on the image's python3, the rest side by side) and its seal's proofs
run as one walk on the kit's own Python, in a real container.

- The seal on the walk and the seal in sh (the specification it mirrors,
  ``kit_prepare_command(..., walk=False)``) give the same bytes from the
  same built kit: status, output, RECORD, MANIFEST, IMAGE_ENV_OK and every
  entry of the kit, on an honest image and on PT's adversarial ones.
- What the walk does not reproduce exactly goes to the sh seal, with the
  same result; a kit without a Python is sealed in sh.
- The build's tool copies on the image's python3 and the loop they mirror
  leave the same kit, byte for byte.
- Behaviour (fails on 5653de15): the seal and the build start no ``find``
  per path component, and the build runs one ``ldd``.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import contextlib
import shlex
import shutil
import subprocess
import uuid

import pytest

from benchflow._runtime_tree import KIT, kit_prepare_command

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"
# musl, busybox's find and cp, a Python the kit copies.
MUSL_IMAGE = "python:3.12-alpine"
KIT_FILES = (
    f"cd {KIT} && for f in RECORD MANIFEST IMAGE_ENV_OK IMAGE_PYTHON_PATHS.json "
    'IMAGE_PYTHON_PATHS.jsonl; do echo "== $f"; cat "$f" 2>/dev/null; done'
)
# Every entry of the kit: type, mode with the set-id bits, owner, link
# target, a file's sha256; listed by the image's own python3 (busybox find has
# no -printf), sorted.
_TREE_PY = (
    "import hashlib, os, stat, sys\n"
    "out = []\n"
    "for root, dirs, files in os.walk(sys.argv[1]):\n"
    "    for n in dirs + files:\n"
    "        p = os.path.join(root, n); st = os.lstat(p)\n"
    "        t = 'l' if stat.S_ISLNK(st.st_mode) else 'd' if stat.S_ISDIR(st.st_mode) "
    "else 'f' if stat.S_ISREG(st.st_mode) else '?'\n"
    "        extra = os.readlink(p) if t == 'l' else hashlib.sha256(open(p, 'rb')"
    ".read()).hexdigest() if t == 'f' else ''\n"
    "        out.append('%s %o %d:%d %s %s' % (t, st.st_mode & 0o7777, st.st_uid, "
    "st.st_gid, os.path.relpath(p, sys.argv[1]), extra))\n"
    "print('\\n'.join(sorted(out)))\n"
)
KIT_TREE = f"python3 -I -S -c {shlex.quote(_TREE_PY)} {KIT}"


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def _fresh(prep: str = "", image: str = IMAGE):
    """A fresh container of *image*, *prep* run as root first (the image as
    built), an ``agent`` user (uid 1000) in it."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "perf-h-kit-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "perf-h=test",
         "--network", "none", image, "sleep", "900"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    try:
        made = _dx(
            name,
            "{ useradd -m -u 1000 agent || adduser -D -u 1000 agent; } >/dev/null 2>&1 "
            f"&& mkdir -p /srv; {prep or 'true'}",
        )
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _seal_both(name: str) -> dict[str, dict]:
    """Build once, then seal a copy of the built kit on the walk and in sh."""
    built = _dx(name, kit_prepare_command("build"))
    assert built.returncode == 0, built.stderr
    assert _dx(name, f"cp -a {KIT} /opt/runtime/rt.built").returncode == 0
    out = {}
    for mode, walk in (("walk", True), ("sh", False)):
        assert (
            _dx(name, f"rm -rf {KIT} && cp -a /opt/runtime/rt.built {KIT}").returncode
            == 0
        )
        done = _dx(name, kit_prepare_command("seal", walk=walk))
        lines = done.stdout.splitlines()
        out[mode] = {
            "rc": done.returncode,
            "stdout": [line for line in lines if not line.startswith("BF kit-seal")],
            "path": [line for line in lines if line.startswith("BF kit-seal")],
            "stderr": done.stderr,
            "files": _dx(name, KIT_FILES).stdout,
            "tree": _dx(name, KIT_TREE).stdout,
        }
    return out


def _same(out: dict[str, dict]) -> None:
    for key in ("rc", "stdout", "stderr", "files", "tree"):
        assert out["walk"][key] == out["sh"][key], key


# The folder libc is loaded from, as $d.
LIBC_DIR = "d=$(dirname $(readlink -f /bin/ls | xargs ldd | awk '/libc.so/{print $3}'))"

# The adversarial images of PT's own kit tests (test_c8_protected_runtime_
# container.py), and the honest one.
SCENARIOS = {
    "honest": "",
    "foreign-file-in-usr-local-bin": (
        "printf '#!/bin/sh\\n' > /usr/local/bin/bf-foreign && "
        "chmod 755 /usr/local/bin/bf-foreign && chown 1000 /usr/local/bin/bf-foreign"
    ),
    "foreign-link-in-usr-bin": (
        "mkdir -p /srv/w && chown 1000 /srv/w && printf '#!/bin/sh\\n' > /srv/w/f && "
        "chmod 755 /srv/w/f && ln -s /srv/w/f /usr/bin/bf-foreign"
    ),
    "writable-loader-folder": "chmod o+w /usr/local/lib",
    "loader-file-naming-agent-folder": (
        "mkdir -p /srv/lib && chown 1000 /srv/lib && "
        "echo /srv/lib > /etc/ld.so.conf.d/zz-agent.conf"
    ),
    "commented-loader-configuration": (
        "printf '  # a comment\\n/usr/local/lib # trailing\\n\\n"
        "include extra.d/*.conf\\n' > /etc/ld.so.conf.d/zz-test.conf && "
        "mkdir -p /etc/ld.so.conf.d/extra.d /opt/vendor/lib && "
        "echo /opt/vendor/lib > /etc/ld.so.conf.d/extra.d/v.conf"
    ),
    "included-writable-folder": (
        "printf 'include extra.d/*.conf\\n' > /etc/ld.so.conf.d/zz-test.conf && "
        "mkdir -p /etc/ld.so.conf.d/extra.d /opt/vendor/lib && "
        "echo /opt/vendor/lib > /etc/ld.so.conf.d/extra.d/v.conf && "
        "chown 1000 /opt/vendor/lib"
    ),
    "comma-and-space-folder-names": (
        "mkdir -p /opt/lib /opt/my '/opt/lib,64' '/opt/my libs' && "
        "chmod 777 /opt/lib /opt/my && "
        "printf '/opt/lib,64\\n/opt/my libs\\n' > /etc/ld.so.conf.d/zz-names.conf"
    ),
    "type-suffix-and-musl-file": (
        "mkdir -p /opt/typed && printf '/opt/typed=libc6\\n' > /etc/ld.so.conf.d/zz-t.conf "
        "&& printf '/opt/musl1:/opt/musl2\\n' > /etc/ld-musl-x86_64.path"
    ),
    "writable-capability-folder": (
        "d=$(dirname $(readlink -f /bin/ls | xargs ldd | awk '/libc.so/{print $3}')); "
        'mkdir -p "$d/glibc-hwcaps/x86-64-v3" && chmod 777 "$d/glibc-hwcaps/x86-64-v3"'
    ),
    "dangling-link-in-a-library-folder": (
        "d=$(dirname $(readlink -f /bin/ls | xargs ldd | awk '/libc.so/{print $3}')); "
        'ln -s /nowhere "$d/libbf-dangling.so"'
    ),
    # PERF-H round 2, the reviewers' cases (perf-h-review-claude-a.md,
    # -b.md): a link in a loader folder whose stat fails other than for a
    # missing target is the sh's to decide (M-1).
    "a-link-that-loops-in-libcs-folder": f'{LIBC_DIR}; ln -s libbf-loop.so "$d/libbf-loop.so"',
    "a-link-loop-in-a-capability-folder": (
        f'{LIBC_DIR}; mkdir -p "$d/glibc-hwcaps/x86-64-v3" && '
        'ln -s b "$d/glibc-hwcaps/x86-64-v3/a" && ln -s a "$d/glibc-hwcaps/x86-64-v3/b"'
    ),
    "a-link-through-a-file-in-libcs-folder": (
        f'{LIBC_DIR}; ln -s /etc/hostname/x "$d/libbf-notdir.so"'
    ),
    "a-link-that-loops-in-usr-local-lib": "ln -s loop /usr/local/lib/loop",
    "a-loop-pair-in-a-usr-local-lib-subfolder": (
        "mkdir -p /usr/local/lib/x86_64 && ln -s b /usr/local/lib/x86_64/a && "
        "ln -s a /usr/local/lib/x86_64/b"
    ),
    "two-include-patterns-and-hwcap": (
        "mkdir -p /etc/ld.so.conf.d/m1 /etc/ld.so.conf.d/m2 /opt/m1 /opt/m2 && "
        "echo /opt/m1 > /etc/ld.so.conf.d/m1/a.conf && "
        "echo /opt/m2 > /etc/ld.so.conf.d/m2/b.conf && chown 1000 /opt/m2 && "
        "printf 'include /etc/ld.so.conf.d/m1/*.conf /etc/ld.so.conf.d/m2/*.conf\\n"
        "hwcap 1 foo\\n' > /etc/ld.so.conf.d/zz-two.conf"
    ),
    "a-loader-file-not-roots": (
        "echo /usr/local/lib > /etc/ld.so.conf.d/zz-own.conf && "
        "chown 1000 /etc/ld.so.conf.d/zz-own.conf"
    ),
    "a-kit-tool-ldd-rejects": "printf '#!/bin/sh\\nexit 0\\n' > /usr/bin/mkfifo",
    "a-path-folder-others-may-write": "chmod o+w /usr/local/sbin",
    "a-link-loop-in-a-path-folder": "ln -s loopy /usr/local/bin/loopy",
    "an-equivalence-class-include": (
        "mkdir -p /etc/ld.so.conf.d/eq /opt/eq && echo /opt/eq > /etc/ld.so.conf.d/eq/a.conf "
        "&& printf 'include /etc/ld.so.conf.d/eq/[[=a=]]*.conf\\n' > /etc/ld.so.conf.d/zz-eq.conf"
    ),
    "foreign-link-and-name-with-a-space": (
        "mkdir -p /srv/w && chown 1000 /srv/w && "
        "ln -s /srv/w/f '/usr/local/bin/a b' && ln -s /bin/true /usr/local/bin/'c d'"
    ),
    # PERF-H round 3 (perf-h-r2-review-claude-b.md L-1): a class that is not
    # first in its set, the included files naming a folder the agent owns.
    "a-class-not-first-in-its-set": (
        "mkdir -p /srv/lib /etc/bfinc && chown 1000 /srv/lib && "
        "echo /srv/lib > /etc/bfinc/1.conf && echo /srv/lib > /etc/bfinc/x.conf && "
        "printf 'include /etc/bfinc/[y[:digit:]].conf\\n' > /etc/ld.so.conf.d/zz.conf"
    ),
    # (perf-h-r2-review-claude-a.md L-1): a link in a PATH folder whose
    # target has a 300-byte name (ENAMETOOLONG): the image's find decides.
    "a-link-with-a-too-long-name-in-a-path-folder": "ln -s $(printf 'x%.0s' $(seq 300)) /usr/local/bin/bf-long",
}


# What the walk leaves to the sh (exit 4, no change): a loader-folder or
# PATH-folder link whose stat fails other than for a missing target (M-1,
# r3 A L-1), a NUL in a loader file, a pattern class Python's glob reads
# differently (r3 B L-1: anywhere in a set).
DEFERS = {
    "a-link-loop-in-a-path-folder",
    "a-link-with-a-too-long-name-in-a-path-folder",
    "a-class-not-first-in-its-set",
    "musl:a-class-not-first-in-its-set",
    "musl:a-link-with-a-too-long-name-in-a-path-folder",
    "a-link-that-loops-in-libcs-folder",
    "a-link-loop-in-a-capability-folder",
    "a-link-through-a-file-in-libcs-folder",
    "a-link-that-loops-in-usr-local-lib",
    "a-loop-pair-in-a-usr-local-lib-subfolder",
    "an-equivalence-class-include",
    "musl:a-link-that-loops-in-lib",
    "musl:a-link-through-a-file-in-lib",
    "musl:musl-path-file-with-a-nul",
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_walk_seals_byte_for_byte_as_the_sh_seal(scenario):
    with _fresh(SCENARIOS[scenario]) as name:
        out = _seal_both(name)
    _same(out)
    if out["walk"]["rc"] == 0:
        assert out["sh"]["path"] == ["BF kit-seal sh"]
        # No kit Python (the build left it out): the sh seals.
        python = "python=1" in out["walk"]["files"]
        path = "walk" if python and scenario not in DEFERS else "sh"
        assert out["walk"]["path"] == [f"BF kit-seal {path}"]
        assert "BF kit-record" in out["walk"]["stdout"]
    else:
        # The walk refused with the sh's own line, before either path ended.
        assert out["walk"]["path"] == out["sh"]["path"] == []
        assert "benchflow: " in out["walk"]["stderr"]


# The same comparison on musl with busybox's find and cp (PERF-H round 2).
MUSL_SCENARIOS = {
    "honest": "",
    "foreign-file-in-usr-local-bin": (
        "printf '#!/bin/sh\\n' > /usr/local/bin/bf-foreign && "
        "chmod 755 /usr/local/bin/bf-foreign && chown 1000 /usr/local/bin/bf-foreign"
    ),
    "foreign-link-in-usr-bin": (
        "mkdir -p /srv/w && chown 1000 /srv/w && printf '#!/bin/sh\\n' > /srv/w/f && "
        "chmod 755 /srv/w/f && ln -s /srv/w/f /usr/bin/bf-foreign"
    ),
    "writable-loader-folder": "chmod o+w /usr/local/lib",
    "musl-path-file-naming-agent-folder": (
        "mkdir -p /srv/lib && chown 1000 /srv/lib && "
        "printf '/lib\\n/usr/local/lib\\n/usr/lib\\n/srv/lib\\n' > /etc/ld-musl-x86_64.path"
    ),
    "musl-path-file-with-a-nul": (
        "mkdir -p /srv/lib && chown 1000 /srv/lib && "
        "printf '/lib:/usr/local/lib:/usr/lib\\000/srv/lib\\n' > /etc/ld-musl-x86_64.path"
    ),
    "a-link-that-loops-in-lib": "ln -s libbf-loop.so /lib/libbf-loop.so",
    "a-link-through-a-file-in-lib": "ln -s /etc/hostname/x /lib/libbf-notdir.so",
    "a-dangling-link-in-lib": "ln -s /nowhere /lib/libbf-dangling.so",
    "a-link-that-loops-in-usr-local-lib": "ln -s loop /usr/local/lib/loop",
    "foreign-link-and-name-with-a-space": (
        "mkdir -p /srv/w && chown 1000 /srv/w && "
        "ln -s /srv/w/f '/usr/local/bin/a b' && ln -s /bin/true /usr/local/bin/'c d'"
    ),
    "a-class-not-first-in-its-set": (
        "mkdir -p /srv/lib /etc/bfinc && chown 1000 /srv/lib && "
        "echo /srv/lib > /etc/bfinc/1.conf && echo /srv/lib > /etc/bfinc/x.conf && "
        "printf 'include /etc/bfinc/[y[:digit:]].conf\\n' >> /etc/ld.so.conf"
    ),
    "a-link-with-a-too-long-name-in-a-path-folder": "ln -s $(printf 'x%.0s' $(seq 300)) /usr/local/bin/bf-long",
}


@pytest.mark.parametrize("scenario", sorted(MUSL_SCENARIOS))
def test_the_walk_seals_byte_for_byte_as_the_sh_seal_on_musl(scenario):
    with _fresh(MUSL_SCENARIOS[scenario], image=MUSL_IMAGE) as name:
        out = _seal_both(name)
    _same(out)
    if out["walk"]["rc"] == 0:
        assert out["sh"]["path"] == ["BF kit-seal sh"]
        python = "python=1" in out["walk"]["files"]
        path = "walk" if python and f"musl:{scenario}" not in DEFERS else "sh"
        assert out["walk"]["path"] == [f"BF kit-seal {path}"]


def test_a_dropped_program_is_the_same_refusal_on_the_walk(tmp_path):
    """PT-C8-R9-4: the image's tar has RUNPATH /opt/vendor/lib, which the
    agent owns: both seals drop it and leave the same refusal."""
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        pytest.skip("needs patchelf on the test host")
    with _fresh() as name:
        tar = tmp_path / "tar"
        subprocess.run(["docker", "cp", f"{name}:/usr/bin/tar", str(tar)], check=True)
        subprocess.run(
            [patchelf, "--set-rpath", "/opt/vendor/lib", str(tar)], check=True
        )
        subprocess.run(["docker", "cp", str(tar), f"{name}:/usr/bin/tar"], check=True)
        made = _dx(
            name,
            "chown 0:0 /usr/bin/tar && chmod 755 /usr/bin/tar && "
            "mkdir -p /opt/vendor/lib && chown 1000 /opt/vendor/lib",
        )
        assert made.returncode == 0, made.stderr
        out = _seal_both(name)
    _same(out)
    assert out["walk"]["path"] == ["BF kit-seal walk"]
    tree = out["walk"]["tree"].splitlines()
    assert not any(" libexec/_usr_bin_tar " in line for line in tree)
    assert any(line.startswith("f 755 0:0 bin/tar ") for line in tree)


def test_an_include_pattern_the_walk_cannot_mirror_goes_to_the_sh_seal():
    """A character class in an include pattern ([[:alnum:]]): Python's glob
    reads it differently, so the walk changes nothing and the sh seals."""
    prep = (
        "mkdir -p /etc/ld.so.conf.d/cls /opt/cls && "
        "echo /opt/cls > /etc/ld.so.conf.d/cls/a.conf && "
        "printf 'include /etc/ld.so.conf.d/cls/[[:alnum:]]*.conf\\n' "
        "> /etc/ld.so.conf.d/zz-cls.conf"
    )
    with _fresh(prep) as name:
        out = _seal_both(name)
    _same(out)
    assert out["walk"]["rc"] == 0
    assert out["walk"]["path"] == ["BF kit-seal sh"]
    assert ".walk-applying" not in out["walk"]["tree"]
    assert ".walk-applying" not in out["walk"]["files"]


def test_a_kit_without_a_python_is_sealed_in_sh():
    """No root-only python3 to copy: no kit Python, so the sh seals."""
    prep = "chown 1000 /usr/local/bin/python3.12"
    with _fresh(prep) as name:
        out = _seal_both(name)
    _same(out)
    assert out["walk"]["rc"] == 0
    assert out["walk"]["path"] == ["BF kit-seal sh"]
    assert "python=0" in out["walk"]["files"]


# The build's tool copies: the walk on the image's python3 against bf_tool.
BUILD_SCENARIOS = {
    "honest": "",
    "a-tool-another-user-owns": "chown 1000 /usr/bin/tar",
    "a-tool-through-a-writable-folder": (
        "mkdir -p /srv/w && chown 1000 /srv/w && mv /usr/bin/xargs /srv/w/xargs && "
        "ln -s /srv/w/xargs /usr/bin/xargs"
    ),
    "a-tool-only-in-usr-local": (
        "mv /usr/bin/od /usr/local/bin/od && mv /usr/bin/cksum /usr/local/sbin/cksum"
    ),
    "a-tool-not-executable": "chmod a-x /usr/bin/tee",
    "setuid-tools-and-links": "chmod 4755 /usr/bin/du && ln -sf ../../bin/ls /usr/bin/dirname",
    # PERF-H round 3 (perf-h-r2-review-claude-a.md L-2): a cp that keeps
    # modes, ahead of coreutils' on PATH, and a set-uid tool.
    "a-mode-keeping-cp-wrapper": (
        "printf '#!/bin/sh\\nexec /bin/cp -p \"$@\"\\n' > /usr/local/bin/cp && "
        "chmod 755 /usr/local/bin/cp && chmod 4755 /usr/bin/du"
    ),
}


# busybox's cp keeps the set-id bits coreutils' drops: the loop copies those.
MUSL_BUILD_SCENARIOS = {
    "honest": "",
    "a-set-id-busybox": "chmod 4755 /bin/busybox",
    # busybox's cp keeps the sticky bit too (r3, A L-2).
    "a-sticky-busybox": "chmod 1755 /bin/busybox",
}


@pytest.mark.parametrize(
    ("image", "scenario"),
    [(IMAGE, s) for s in sorted(BUILD_SCENARIOS)]
    + [(MUSL_IMAGE, s) for s in sorted(MUSL_BUILD_SCENARIOS)],
)
def test_the_tools_walk_builds_byte_for_byte_as_the_loop(image, scenario):
    trees = {}
    prep = (BUILD_SCENARIOS if image == IMAGE else MUSL_BUILD_SCENARIOS)[scenario]
    with _fresh(prep, image=image) as name:
        for mode, walk in (("walk", True), ("sh", False)):
            assert _dx(name, f"rm -rf {KIT}").returncode == 0
            done = _dx(name, kit_prepare_command("build", walk=walk))
            trees[mode] = (
                done.returncode,
                done.stdout,
                done.stderr,
                _dx(name, KIT_TREE).stdout,
            )
    assert trees["walk"] == trees["sh"]
    assert trees["walk"][0] == 0


# Behaviour, on the API 5653de15 has too (kit_prepare_command(phase)).


def _wrap(name: str, program: str, log: str) -> None:
    """*program* (a root-owned file on the system folders) logs its start and
    end, in ns, to *log*, then runs as itself."""
    real = f"{program}.perf-h-real"
    made = _dx(
        name,
        f"mv {program} {real} && cat > {program} <<'PERFH'\n"
        "#!/bin/sh\n"
        f'echo "s $(date +%s%N)" >> {log}\n'
        f'{real} "$@"; rc=$?\n'
        f'echo "e $(date +%s%N)" >> {log}\n'
        "exit $rc\n"
        "PERFH\n"
        f"chmod 755 {program} && : > {log} && chmod 666 {log}",
    )
    assert made.returncode == 0, made.stderr


def _intervals(name: str, log: str) -> list[tuple[int, int]]:
    """(start, end) pairs, matched in the order the starts and ends came."""
    events = [line.split() for line in _dx(name, f"cat {log}").stdout.splitlines()]
    starts = sorted(int(t) for kind, t in events if kind == "s")
    ends = sorted(int(t) for kind, t in events if kind == "e")
    return list(zip(starts, ends, strict=False))


def test_the_seal_starts_no_find_per_path_component():
    """5653de15's seal walked every folder, file and link hop with a
    ``find`` each (hundreds per seal). On the walk, ``find`` is left only
    to the kit's own tree check at the end."""
    with _fresh() as name:
        built = _dx(name, kit_prepare_command("build"))
        assert built.returncode == 0, built.stderr
        _wrap(name, "/usr/bin/find", "/tmp/perf-h-find.log")
        sealed = _dx(name, kit_prepare_command("seal"))
        assert sealed.returncode == 0, sealed.stderr
        calls = _intervals(name, "/tmp/perf-h-find.log")
    assert 0 < len(calls) < 40, len(calls)


def test_the_build_names_every_extension_module_in_one_ldd():
    """5653de15 ran ``ldd`` once for the Python and once for each of its
    extension modules (79 on this image); on glibc the build runs one."""
    with _fresh() as name:
        _wrap(name, "/usr/bin/ldd", "/tmp/perf-h-ldd.log")
        built = _dx(name, kit_prepare_command("build"))
        assert built.returncode == 0, built.stderr
        calls = _intervals(name, "/tmp/perf-h-ldd.log")
    assert len(calls) == 1, len(calls)


def test_the_build_starts_no_find_per_tool():
    """5653de15 proved each tool's route with a ``find`` per path component
    (hundreds per build); the tools' walk proves them in one process."""
    with _fresh() as name:
        _wrap(name, "/usr/bin/find", "/tmp/perf-h-find.log")
        built = _dx(name, kit_prepare_command("build"))
        assert built.returncode == 0, built.stderr
        calls = _intervals(name, "/tmp/perf-h-find.log")
    assert len(calls) < 40, len(calls)
