"""BenchFlow's protected verifier runtime and the guard's use of it."""

import hashlib
import importlib.util
import io
import json
import os
import py_compile
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import _verifier_runtime as runtime
from benchflow.sandbox import lockdown
from benchflow.task.config import TaskConfig
from tests._receipt_fakes import receipt_aware
from tests.test_verifier_uv_state import CTRF_PLUGIN, hostile_ctrf

LINUX_312 = runtime.Interpreter(
    exe="/usr/local/bin/python3.12",
    version=(3, 12),
    machine="x86_64",
    glibc=(2, 36),
    musl=None,
    markers={
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_system": "Linux",
        "python_full_version": "3.12.7",
        "python_version": "3.12",
        "sys_platform": "linux",
    },
    stdlib=["/usr/local/lib/python3.12", "/usr/local/lib/python3.12/lib-dynload"],
)


def wheel(
    name, version, *, requires=(), entry=None, module=None, source="", python=None
):
    """A wheel's bytes: METADATA, optional pytest11 entry point and one module."""
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if python:
        metadata += f"Requires-Python: {python}\n"
    metadata += "".join(f"Requires-Dist: {r}\n" for r in requires)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(f"{info}/METADATA", metadata + "\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"{info}/RECORD", "")
        if entry:
            archive.writestr(f"{info}/entry_points.txt", f"[pytest11]\n{entry}\n")
        if module:
            archive.writestr(f"{module}.py", source)
    return data.getvalue()


# pytest 8.4.2's own Requires-Dist, as PyPI serves it.
PYTEST_REQUIRES = (
    'exceptiongroup>=1; python_version < "3.11"',
    "iniconfig>=1",
    "packaging>=20",
    "pluggy<2,>=1.5",
    "pygments>=2.7.2",
    'tomli>=1; python_version < "3.11"',
)


class PyPI:
    """A fake PyPI: per release a wheel; per URL how many times it fails."""

    def __init__(self):
        self.releases = {}
        self.blobs = {}
        self.failures = {}
        self.calls = []

    def add(self, name, version, data, filename=None):
        filename = filename or f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        url = f"https://files.example/{filename}"
        self.blobs[url] = data
        self.releases[(name, version)] = [
            {
                "filename": filename,
                "packagetype": "bdist_wheel",
                "url": url,
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
            }
        ]

    def get(self, url):
        self.calls.append(url)
        if self.failures.get(url):
            self.failures[url] -= 1
            return SimpleNamespace(
                status_code=503, raise_for_status=_raise, content=b"", json=dict
            )
        if url.startswith("https://pypi.org/pypi/"):
            name, version = url.split("/")[4:6]
            files = self.releases.get((name, version))
            if files is None:
                return SimpleNamespace(status_code=404)
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"urls": files},
            )
        return SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, content=self.blobs[url]
        )


def _raise():
    raise OSError("DUMMY 503")


@pytest.fixture
def pypi(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
    monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    fake = PyPI()
    for pin in runtime.BENCHFLOW_PYTEST:
        name, version = runtime.parse_pin(pin)
        requires = PYTEST_REQUIRES if name == "pytest" else ()
        fake.add(
            name,
            version,
            wheel(name, version, requires=requires, module=name.replace("-", "_")),
        )
    fake.add(
        "pytest-json-ctrf",
        "0.3.5",
        wheel(
            "pytest-json-ctrf",
            "0.3.5",
            requires=["pytest>6.0.0"],
            entry="ctrf = ctrf_main",
            module="ctrf_main",
            source=CTRF_PLUGIN,
        ),
    )
    monkeypatch.setattr(runtime, "_http_get", fake.get)
    return fake


def test_the_tree_holds_the_plugins_and_benchflows_pytest(pypi, tmp_path):
    resolution = runtime.resolve([], ctrf=True, interpreter=LINUX_312)
    names = {w.name for w in resolution.wheels}
    assert {"pytest", "pluggy", "pytest-json-ctrf"} <= names
    # Python 3.12 needs neither tomli nor exceptiongroup.
    assert "tomli" not in names and "exceptiongroup" not in names
    assert resolution.plugins == {"ctrf": "ctrf_main"}
    assert resolution.plugin_top == ["ctrf_main"]
    manifest = runtime.build_tree(
        resolution, interpreter=LINUX_312, tests_dir="/tests", into=tmp_path / "tree"
    )
    assert (tmp_path / "tree/site/ctrf_main.py").read_text() == CTRF_PLUGIN
    assert manifest["runner"] == runtime.RUNNER
    # C8's kit Python (it adds -I -S) with -B, after the custody check of
    # the kit and of this tree.
    script = (tmp_path / "tree/bin/pytest").read_text()
    assert (
        f"exec {runtime.KIT_PYTHON} -B -X pycache_prefix={runtime.NO_CACHE} " in script
    )
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in script
    assert script.index("sha256sum -c") < script.index("exec ")
    assert f"PATH={runtime.KIT_BIN}" in script
    run = (tmp_path / "tree/run.py").read_text()
    assert "SITE = '/opt/benchflow/verifier/site'" in run
    assert '"--confcutdir=" + TESTS' in run and "TESTS = '/tests'" in run
    assert "PLUGINS = {'ctrf': 'ctrf_main'}" in run
    sums = (tmp_path / "tree/SHA256SUMS").read_text()
    assert "  ./run.py" in sums and "  ./site/ctrf_main.py" in sums


def test_without_a_root_only_python_the_tree_serves_plugins_alone(pypi):
    resolution = runtime.resolve([], ctrf=True, interpreter=None)
    assert [w.name for w in resolution.wheels] == ["pytest-json-ctrf"]
    assert not resolution.runner


def test_a_dependency_the_task_did_not_pin_refuses_before_the_agent(pypi):
    pypi.add(
        "pytest-thing",
        "1.0",
        wheel("pytest-thing", "1.0", requires=["rich>=13"], entry="thing = thing"),
    )
    with pytest.raises(runtime.RuntimePinError, match=r"rich>=13.*does not pin"):
        runtime.resolve(["pytest-thing==1.0"], ctrf=False, interpreter=LINUX_312)


def test_a_release_pypi_does_not_have_is_the_tasks_to_fix(pypi):
    with pytest.raises(runtime.RuntimePinError, match="no release"):
        runtime.resolve(["pytest-nothing==9.9"], ctrf=False, interpreter=LINUX_312)


def test_a_platform_wheel_is_picked_by_the_interpreters_tags(pypi):
    data = wheel("fastplug", "1.0", entry="fast = fastplug", module="fastplug")
    pypi.add(
        "fastplug", "1.0", data, "fastplug-1.0-cp311-cp311-manylinux_2_17_x86_64.whl"
    )
    pypi.releases[("fastplug", "1.0")].append(
        {
            **pypi.releases[("fastplug", "1.0")][0],
            "filename": "fastplug-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        }
    )
    resolution = runtime.resolve(["fastplug==1.0"], ctrf=False, interpreter=LINUX_312)
    (fast,) = [w for w in resolution.wheels if w.name == "fastplug"]
    assert fast.filename == "fastplug-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"
    with pytest.raises(runtime.RuntimePinError, match="no wheel"):
        runtime.resolve(["fastplug==1.0"], ctrf=False, interpreter=None)


def test_offline_pins_keep_working_from_the_cache(pypi, monkeypatch):
    """Astra R3-5: a pin fetched once is built offline from the host's cache."""
    runtime.resolve([], ctrf=True, interpreter=LINUX_312)
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    pypi.calls.clear()
    resolution = runtime.resolve([], ctrf=True, interpreter=LINUX_312)
    assert pypi.calls == []
    assert "pytest-json-ctrf" in {w.name for w in resolution.wheels}


def test_offline_without_a_cache_is_the_networks_failure(pypi, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    with pytest.raises(runtime.RuntimeUnavailable, match="offline"):
        runtime.resolve([], ctrf=True, interpreter=None)


def test_a_failed_download_is_retried_and_never_cached(pypi, monkeypatch):
    """Sol R3-4: a 503 for the wheel is retried; one that keeps failing stops
    the trial as the network's failure and leaves nothing cached, so the next
    trial fetches it again rather than trusting a partial set."""
    (entry,) = pypi.releases[("pytest-json-ctrf", "0.3.5")]
    pypi.failures[entry["url"]] = 2
    runtime.resolve([], ctrf=True, interpreter=None)
    assert pypi.calls.count(entry["url"]) == 3
    cached = Path.home() / ".cache/benchflow/verifier-runtime/wheels"
    assert (cached / f"{entry['digests']['sha256']}.whl").exists()

    for path in cached.iterdir():
        path.unlink()
    pypi.failures[entry["url"]] = 10
    with pytest.raises(runtime.RuntimeUnavailable, match="could not be fetched"):
        runtime.resolve([], ctrf=True, interpreter=None)
    assert not any(cached.iterdir())
    pypi.failures[entry["url"]] = 0
    runtime.resolve([], ctrf=True, interpreter=None)
    assert any(cached.iterdir())


def test_a_wheel_whose_sha256_differs_from_pypis_is_never_used(pypi):
    (entry,) = pypi.releases[("pytest-json-ctrf", "0.3.5")]
    pypi.blobs[entry["url"]] = b"DUMMY tampered"
    with pytest.raises(runtime.RuntimeUnavailable, match="sha256"):
        runtime.resolve([], ctrf=True, interpreter=None)


def test_the_task_asks_for_the_tree_by_its_pins_or_by_ctrf(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("pytest --ctrf /logs/verifier/ctrf.json\n")

    def task(toml):
        return SimpleNamespace(
            task_dir=tmp_path,
            paths=SimpleNamespace(tests_dir=tests, uses_native_verifier_dir=False),
            config=TaskConfig.model_validate_toml('version = "1.0"\n' + toml),
        )

    assert runtime.runtime_requested(task("")) == ([], True)
    pinned = task('[verifier]\npytest_plugin_releases = ["pytest-json-ctrf==0.6.1"]\n')
    assert runtime.runtime_requested(pinned) == (["pytest-json-ctrf==0.6.1"], False)
    (tests / "test.sh").write_text("pytest\n")
    assert runtime.runtime_requested(task("")) == ([], False)


# ---------------------------------------------------------------------------
# The guard: where code may come from.
# ---------------------------------------------------------------------------


def rearm(key):
    """Arm the guard now: after what the test wrote so far, past the coarse
    clock the kernel stamps change times with."""
    time.sleep(0.05)
    key.chmod(0o600)
    time.sleep(0.05)


@pytest.fixture
def armed(tmp_path, monkeypatch):
    key = tmp_path / "guard" / ".key"
    key.parent.mkdir()
    key.write_text("x")
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    return key


def test_runtime_modules_load_from_the_tree_and_read_no_cache(
    tmp_path, monkeypatch, armed
):
    """A runtime plugin is compiled from its source: an unchecked-hash cache the
    agent left under the prefix test.sh set for that path never runs."""
    site = tmp_path / "runtime/site"
    site.mkdir(parents=True)
    (site / "bf_runtime_plugin.py").write_text("SOURCE = 'tree'\n")
    prefix = tmp_path / "agent-bytecache"
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    hostile = tmp_path / "hostile.py"
    hostile.write_text("SOURCE = 'agent cache'\n")
    target = Path(importlib.util.cache_from_source(str(site / "bf_runtime_plugin.py")))
    target.parent.mkdir(parents=True)
    py_compile.compile(
        str(hostile),
        cfile=str(target),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    # The same name earlier on sys.path, where test.sh's environment has it.
    shadow = tmp_path / "env"
    shadow.mkdir()
    (shadow / "bf_runtime_plugin.py").write_text("SOURCE = 'environment'\n")
    monkeypatch.syspath_prepend(str(shadow))
    monkeypatch.setattr(guard, "_BENCHFLOW_RUNTIME_SITE", str(site))
    monkeypatch.setattr(guard, "_BENCHFLOW_RUNTIME_TOP", ("bf_runtime_plugin",))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    sys.modules.pop("bf_runtime_plugin", None)
    guard._install_runtime_finder()
    try:
        module = importlib.import_module("bf_runtime_plugin")
        assert module.SOURCE == "tree"
        assert guard._from_runtime(module)
    finally:
        sys.modules.pop("bf_runtime_plugin", None)


def test_a_file_installed_after_the_arm_is_refused_wherever_it_is(
    tmp_path, monkeypatch, armed
):
    system = tmp_path / "system-site"
    system.mkdir()
    old = system / "image_plugin.py"
    old.write_text("# the image's\n")
    rearm(armed)
    new = system / "new_plugin.py"
    new.write_text("# test.sh's root install\n")
    assert guard.untrusted_reason(str(old), ()) is None
    assert "after the agent stopped" in guard.untrusted_reason(str(new), ())


@pytest.mark.parametrize("honest", [True, False])
def test_an_image_modules_cache_under_the_prefix_must_be_its_compilation(
    tmp_path, monkeypatch, armed, honest
):
    """R3-3: image code is trusted only with the caches Python would read."""
    image = tmp_path / "image"
    image.mkdir()
    source = image / "image_plugin.py"
    source.write_text(CTRF_PLUGIN)
    rearm(armed)
    prefix = tmp_path / "agent-bytecache"
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    compiled = source if honest else tmp_path / "other.py"
    if not honest:
        compiled.write_text(hostile_ctrf(tmp_path / "imported"))
    target = Path(importlib.util.cache_from_source(str(source)))
    target.parent.mkdir(parents=True)
    py_compile.compile(
        str(compiled),
        cfile=str(target),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    reason = guard.untrusted_reason(str(source), (str(tmp_path / "app"),))
    assert (reason is None) is honest, reason


def test_caches_are_looked_up_by_the_spelling_python_imports(
    tmp_path, monkeypatch, armed
):
    """R3-1: ``/x/link/../site/m.py`` opens ``/x/real/site/m.py`` and reads the
    ``__pycache__`` there, not the plain spelling's: each spelling is judged
    where it lands and by its own cache."""
    for site in (tmp_path / "site", tmp_path / "real/site"):
        site.mkdir(parents=True)
        (site / "m.py").write_text(CTRF_PLUGIN)
    (tmp_path / "real/dir").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real/dir", target_is_directory=True)
    hostile = tmp_path / "other.py"
    hostile.write_text(hostile_ctrf(tmp_path / "imported"))
    spelling = f"{tmp_path}/link/../site/m.py"
    target = Path(importlib.util.cache_from_source(spelling))
    target.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(
        str(hostile),
        cfile=str(target),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    assert os.path.realpath(target).startswith(str(tmp_path / "real/site"))
    rearm(armed)
    # The cache the spelling reads is in a folder the agent could write.
    agent = (str(tmp_path / "real/site/__pycache__"),)
    assert guard.untrusted_reason(str(tmp_path / "site/m.py"), agent) is None
    reason = guard.untrusted_reason(spelling, agent)
    assert reason is not None and "byte code" in reason


def test_the_guard_imports_its_dependencies_from_the_standard_library_alone():
    """Sol R3-2: every module the guard uses comes from the standard library."""
    for name in ("hashlib", "hmac", "json", "marshal", "importlib.util", "pathlib"):
        assert guard._standard_origin(sys.modules[name]), name
    planted = SimpleNamespace(__file__="/app/hashlib.py", __spec__=None)
    assert not guard._standard_origin(planted)


def test_the_guard_checks_every_plugin_module_pytest_registers(
    tmp_path, monkeypatch, armed
):
    """R3-4: every module pytest registers is checked, not only those asked
    for; a conftest (registered under its path) is V1's."""
    agent = tmp_path / "app"
    agent.mkdir()
    (agent / "evil.py").write_text("X = 1\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(agent),))
    evil = type(sys)("evil")
    evil.__file__ = str(agent / "evil.py")
    with pytest.raises(guard.Rejected, match="evil"):
        guard._check_registered(evil, SimpleNamespace(get_name=lambda plugin: "evil"))
    conftest = type(sys)("conftest")
    conftest.__file__ = str(agent / "conftest.py")
    guard._check_registered(
        conftest, SimpleNamespace(get_name=lambda plugin: str(agent / "conftest.py"))
    )


@pytest.mark.verifier_runtime
async def test_hardening_refuses_a_task_whose_runtime_is_missing(tmp_path, monkeypatch):
    """A task that needs the tree and has none stops hardening: its plugins
    would otherwise be refused, or loaded from where the agent could write."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("pytest --ctrf /logs/verifier/ctrf.json\n")
    task = SimpleNamespace(
        task_dir=tmp_path,
        paths=SimpleNamespace(tests_dir=tests, uses_native_verifier_dir=False),
        config=TaskConfig.model_validate_toml('version = "1.0"\n'),
    )

    class Sandbox:
        @receipt_aware
        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout="", stderr="", return_code=4)

    with pytest.raises(RuntimeError, match="was not built before the agent"):
        await lockdown._verifier_runtime_manifest(Sandbox(), task, "agent", build=False)
    manifest = {"site": runtime.SITE, "plugins": {"ctrf": "ctrf.main"}}

    class Built:
        @receipt_aware
        async def exec(self, command, **kwargs):
            assert "/opt/benchflow/verifier" in command and "sha256sum -c" in command
            return SimpleNamespace(
                stdout=json.dumps(manifest), stderr="", return_code=0
            )

    assert (
        await lockdown._verifier_runtime_manifest(Built(), task, "agent", build=False)
        == manifest
    )


def test_a_module_the_runtime_holds_is_not_judged_on_test_shs_path(
    tmp_path, monkeypatch, armed
):
    """A conftest's ``pytest_plugins = ['pytest_timeout']``, with the release
    pinned: the import comes from the runtime, so the copy test.sh installed
    in /app (refused there) is never consulted."""
    agent = tmp_path / "app" / "site-packages"
    agent.mkdir(parents=True)
    (agent / "bf_pinned_mod.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(agent))
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(tmp_path / "app"),))
    monkeypatch.setitem(
        sys.modules, "_pytest.config", SimpleNamespace(builtin_plugins=())
    )
    with pytest.raises(guard.Rejected):
        guard._validate(["bf_pinned_mod"], entry_points=False)
    monkeypatch.setattr(guard, "_BENCHFLOW_RUNTIME_TOP", ("bf_pinned_mod",))
    guard._validate(["bf_pinned_mod"], entry_points=False)


async def test_a_conftest_that_names_a_runtime_plugin_module_is_scored(
    tmp_path, monkeypatch
):
    """A verifier conftest's ``pytest_plugins = ['ctrf_model']`` asks pytest
    for the runtime plugin the guard already registered under its pytest11
    name: one module under two names stopped pytest ("Plugin already
    registered under a different name"), so an honest task went unscored."""
    from tests.test_verifier_uv_state import PASSING, score

    def leave(layout):
        tests = layout.tmp / "verifier-tests"
        tests.mkdir()
        (tests / "conftest.py").write_text("pytest_plugins = ['ctrf_model']\n")

    def nothing(env, layout):
        return []

    _, _, rewards, error = await score(
        tmp_path, monkeypatch, PASSING, nothing, before=leave, workspace_name="app"
    )

    assert error is None, error
    assert rewards == {"reward": 1.0}
