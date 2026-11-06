"""PT-C8-CONF-2b round 3: the guard capture reads exactly the folders the
guard installer judges.

The installer's folders are PID 1's PATH split on ``:``, then
GUARD_SYSTEM_PATH's, each absolute one once, in that order. The capture
loop of root's kit (``kit_prepare_command``) must visit the same list, or an
interpreter the installer meets is "unrecorded" on an honest image: no copy,
a ``benchflow:`` line and the guard on PYTHONPATH only.

On e4090a15f the loop split ``$bf_ipath:<system folders>``: a shell splits
only what an expansion gives, so PATH's last folder and the literal after
it stayed one word, and no system folder was visited.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from benchflow._image_python_paths import GUARD_SYSTEM_PATH
from benchflow._runtime_tree import kit_prepare_command
from benchflow.sandbox import lockdown

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sh")

SYSTEM = GUARD_SYSTEM_PATH.split(":")

# The guard capture loop, run with each interpreter "capture" printing the
# folder it was asked for, once per folder (its plain ``python``).
_STUBS = (
    "bf_guard_cap() { case $1 in */python) printf '%s\\n' \"${1%/*}\";; esac; }\n"
    "bf_shebang_py() { :; }\n"
)


def _capture_loop() -> str:
    lines = kit_prepare_command("build").split("\n")
    loop = [line for line in lines if 'bf_guard_cap "$py"' in line]
    assert len(loop) == 1, loop
    # The folder list's own function, where the kit has one.
    helpers = [line for line in lines if line.lstrip().startswith("bf_guard_dirs()")]
    return "\n".join([*helpers, loop[0]])


def _visited(path: str) -> list[str]:
    done = subprocess.run(
        ["/bin/sh", "-c", "bf_ipath=$1\n" + _STUBS + _capture_loop(), "sh", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.splitlines()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # The usual PATH: its last folder (/bin) is one of the system ones.
        (
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            SYSTEM,
        ),
        # PATH's last folder holds the image's only Python.
        (
            "/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin",
            [
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
                "/usr/local/bin",
                "/usr/local/sbin",
            ],
        ),
        # A folder off the system list, last.
        (
            "/usr/bin:/opt/py/bin",
            ["/usr/bin", "/opt/py/bin", *[d for d in SYSTEM if d != "/usr/bin"]],
        ),
        # One folder only.
        ("/opt/py/bin", ["/opt/py/bin", *SYSTEM]),
        # Duplicates (fix-code-vulnerability's PATH names /usr/local/bin twice).
        (
            "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            [
                "/usr/local/bin",
                "/usr/local/sbin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            ],
        ),
        ("/opt/a:/opt/b:/opt/a", ["/opt/a", "/opt/b", *SYSTEM]),
        # Empty fields, and relative ones, which the installer skips too.
        ("::/opt/a::bin:.:/opt/b:", ["/opt/a", "/opt/b", *SYSTEM]),
        # No PATH read from PID 1: the system folders alone.
        ("", SYSTEM),
        # A space, and glob characters, are part of the folder's name.
        ("/opt/my env/bin:/opt/[x]*", ["/opt/my env/bin", "/opt/[x]*", *SYSTEM]),
        # The kit's own folders are never the image's.
        ("/opt/benchflow/bin:/opt/a", ["/opt/a", *SYSTEM]),
    ],
)
def test_the_capture_visits_the_installers_folders_once_in_order(path, expected):
    assert _visited(path) == expected


def test_every_python_name_of_the_last_path_folder_is_asked_for():
    asked = subprocess.run(
        [
            "/bin/sh",
            "-c",
            "bf_ipath=$1\nbf_guard_cap() { printf '%s\\n' \"$1\"; }\n"
            "bf_shebang_py() { :; }\n" + _capture_loop(),
            "sh",
            "/usr/sbin:/usr/local/bin",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    # (python3.N names are the folder's own files, as the glob finds them.)
    for name in ("python", "python3"):
        assert f"/usr/local/bin/{name}" in asked, asked
        assert f"/bin/{name}" in asked, asked


# A word list (for ... in, set --) whose unquoted expansion runs straight on
# into more text: the shell splits the expansion, not what follows it.
_WORDS = re.compile(r"(?:\bfor\s+\w+\s+in|\bset\s+--)[ \t]+([^;\n]*)")
_GLUED = re.compile(r"(\$\{[^}]*\}|\$\w+)(?!\w)(?=[^\s;)|&$])")


def _glued_expansions(script: str) -> list[str]:
    found = []
    for words in _WORDS.findall(script):
        unquoted = re.sub(r'"(?:[^"\\]|\\.)*"', '""', words)
        if _GLUED.search(unquoted):
            found.append(words)
    return found


def test_the_scan_finds_the_round_2_loop():
    assert _glued_expansions(
        "IFS=:; set -f; for d in $bf_ipath:/usr/bin:/bin; do :; done"
    )
    assert _glued_expansions("set -- $p/x")
    assert not _glued_expansions('for d in $bf_gdirs; do "$d"/python; done')
    assert not _glued_expansions("set -- ${line#??}")


def test_no_kit_or_installer_word_list_glues_an_expansion_to_text():
    scripts = {phase: kit_prepare_command(phase) for phase in ("build", "seal", "all")}
    scripts["restore"] = kit_prepare_command(
        "restore", stage="/.benchflow-kit-" + "0" * 32
    )
    scripts["installer"] = lockdown._INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE
    scripts["installer kit branch"] = lockdown._install_guard_kit_cmd(
        "/g/x.py", "_bf_guard_r3"
    )
    scripts["installer script"] = lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT
    for name, script in scripts.items():
        assert _glued_expansions(script) == [], name


def test_the_installer_judges_its_path_split_once():
    # Its folders come from one value (the verifier's PATH), split once.
    template = lockdown._INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE
    assert "set -- $verifier_path\n" in template
    assert "verifier_path=__PATH__\n" in template
