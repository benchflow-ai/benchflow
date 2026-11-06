"""U5a-CONF-5 coverage regression at 9a7e0268 and the explicit CONF-2 gap."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown
from tests.test_verifier_prep_path import backend as backend


def _sandbox_user_site(image_env):
    """The user site-packages folder of the sandbox's ``python3`` (the one on
    ``image_env``'s PATH), under ``image_env``'s PYTHONUSERBASE."""
    python3 = shutil.which("python3", path=image_env["PATH"])
    assert python3, "no python3 on the sandbox PATH"
    done = subprocess.run(
        [python3, "-c", "import site; print(site.getusersitepackages())"],
        env=image_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return Path(done.stdout.strip())


def _hook_command(workspace):
    # Only run the final site-hook clean-up, never the rootfs/tmp sweeps: the
    # step _build_cleanup_cmd appends last.
    command = lockdown._cleanup_site_cmd(str(workspace))
    full = lockdown._build_cleanup_cmd(
        {"cleanup_conftests": False}, workspace=str(workspace)
    )
    assert full.endswith(command)
    return command


@pytest.mark.parametrize("route", ["batch", "fallback", "individual"])
@pytest.mark.parametrize(
    "folder",
    [
        "workspace",
        "relative",
        "empty",
        "dot",
        "spaces",
        "newline",
        "glob",
        "symlink",
        "user-site",
        "pth-path",
        "pth-code",
    ],
)
async def test_hook_cleanup_keeps_original_directory_coverage(
    backend, tmp_path, monkeypatch, route, folder
):
    """Guards the CONF-5 rollback of 0f3d5f32, without hiding the CONF-2 gap.

    These hooks are dormant until the later launch, as in Astra's report.
    Their files must be gone, not merely skipped by isolated cleanup. The
    ordinary interpreter's original search set includes executable .pth path
    additions; discovering that set in isolation is not equivalent.
    """
    box, image_env, _ = backend
    image_env["PATH"] = lockdown._SAFE_VERIFIER_PATH
    box.verifier_prep_script = route != "individual"
    names = {"spaces": "space ' quote", "newline": "line\nend\n", "glob": "[x]*?"}
    planted = tmp_path / names.get(folder, "workspace")
    if folder in ("empty", "dot"):
        planted = tmp_path
    planted.mkdir(exist_ok=True)
    marker = tmp_path / "later-hook-ran"
    userbase = tmp_path / "userbase"
    image_env["PYTHONUSERBASE"] = str(userbase)
    image_env.pop("PYTHONNOUSERSITE", None)
    # The user site of the interpreter the sandbox runs (python3 on the
    # verifier's PATH), not the test runner's: their versions can differ, as
    # on a 3.13 runner whose image python3 is the distro's 3.12.
    site = _sandbox_user_site(image_env)
    site.mkdir(parents=True)
    pythonpath = str(planted)
    if folder == "relative":
        pythonpath = planted.name
    elif folder in ("empty", "dot"):
        pythonpath = ":" if folder == "empty" else "."
    elif folder == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(planted, target_is_directory=True)
        pythonpath = str(alias)
    elif folder == "user-site":
        planted = site
        pythonpath = ""
    elif folder in ("pth-path", "pth-code"):
        pythonpath = ""
        (site / "path-to-workspace.pth").write_text(
            str(planted) + "\n"
            if folder == "pth-path"
            else f"import sys; sys.path.append({str(planted)!r})\n"
        )
    # This is task-supplied persistent data, not hostile image configuration.
    # The pre-protected-launch image-only PYTHONPATH was already cleared by
    # the backend and is not claimed as previously covered.
    box._persistent_env = {"PYTHONPATH": pythonpath}
    payload = (
        "import sys\n"
        "if 'u5a-later' in sys.argv:\n"
        f"    open({str(marker)!r}, 'w').write('dummy')\n"
    )
    for name in ("sitecustomize.py", "usercustomize.py"):
        (planted / name).write_text(payload)
    (planted / "injection.pth").write_text(
        f"import sys; 'u5a-later' in sys.argv and open({str(marker)!r}, 'w').write('dummy')\n"
    )
    package = planted / "honest_package.py"
    package.write_text("VALUE = 'honest'\n")
    cache = planted / "__pycache__" / "honest-cache"
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(b"honest baseline cache")
    parent = tmp_path / "sticky"
    parent.mkdir(mode=0o1700)
    parent.chmod(0o1700)
    monkeypatch.setattr(lockdown, "_PREP_DIR_PREFIX", str(parent / "prep-"))
    if route == "fallback":
        parent.chmod(0o700)
    step = lockdown._PrepStep("hooks", "cleanup_conftests", _hook_command(tmp_path), 10)
    # The interpreter the host judged before the step: the
    # sandbox's python3 on the verifier's PATH.
    token = lockdown._root_python.set(shutil.which("python3", path=image_env["PATH"]))
    try:
        await lockdown._run_prep_steps(box, [step])
    finally:
        lockdown._root_python.reset(token)
    assert not marker.exists(), "delayed hook ran during cleanup"
    if folder in ("pth-path", "pth-code"):
        # PT-R16-R2-3: the folders .pth files name are read only from the
        # .pth files the clean-up keeps. The new one naming *planted* goes,
        # so nothing puts *planted* on the search path and its hooks stay
        # dormant: the later interpreter below shows they never run.
        assert not (site / "path-to-workspace.pth").exists()
    else:
        assert not (planted / "sitecustomize.py").exists()
        assert not (planted / "usercustomize.py").exists()
    # PT-R16's HOOK_RULE: Python runs a .pth only in a
    # site folder, so only there is it a hook; in a PYTHONPATH folder it is
    # data and stays (PT-R15-B3, PT-INTEG-A-1). The later interpreter below
    # shows it never runs.
    assert (planted / "injection.pth").exists() is (folder != "user-site")
    later = await box.exec("python3 -c 'pass' u5a-later", user="root")
    assert later.return_code == 0, later.stderr
    assert not marker.exists(), "hook survived into the later interpreter"
    assert package.read_text() == "VALUE = 'honest'\n"
    assert cache.read_bytes() == b"honest baseline cache"


def test_hook_cleanup_preserves_image_packages_and_caches(tmp_path):
    """Guards CONF-5's rollback without broadening the historical deletion set."""
    import hashlib

    # Map the image-library exemption to a hermetic fixture: CI need not be
    # root or have any third-party packages in the system interpreter.
    trusted = tmp_path / "image-library"
    trusted.mkdir()
    (trusted / "__pycache__").mkdir()
    for name in (
        "sitecustomize.py",
        "usercustomize.py",
        "honest_package.py",
        "honest.pth",
        "__pycache__/honest-cache",
    ):
        (trusted / name).write_text("# honest image content\n")
    files = [p for p in trusted.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    subprocess.run(
        [
            "/bin/sh",
            "-c",
            _hook_command(tmp_path).replace("/usr/local/lib", str(trusted)),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": lockdown._SAFE_VERIFIER_PATH,
            "PYTHONPATH": str(trusted),
        },
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before
