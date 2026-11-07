"""Guards V2 round 6 against the regressions at 0055522c (b137)."""

import subprocess

import pytest

from benchflow.sandbox import _verifier_runtime as runtime
from tests.test_verifier_runtime_r5 import Tree


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def test_alias_import_uses_the_authenticated_mapping(tree):
    """V2-R5-C1: ambient pytest11 metadata cannot redirect an authorized alias."""
    (tree.site / "fakeplug-1.0.dist-info/entry_points.txt").unlink()
    info = tree.tests / "ambient-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: ambient\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text("[pytest11]\nfakeplug = ambient_plugin\n")
    marker = tree.root / "ambient-imported"
    (tree.tests / "ambient_plugin.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n"
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n    assert request.config.fakeplug\n"
    )
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("declaration", ["-p", "conftest"])
@pytest.mark.parametrize("parent", ["tests", "protected"])
def test_dotted_plugin_checks_the_child_before_execution(tree, declaration, parent):
    """V2-R5-REV-1: a normal namespace extension cannot admit a foreign child."""
    base = tree.tests if parent == "tests" else tree.site
    package = base / "helpers"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    )
    foreign = tree.app / "helpers"
    foreign.mkdir()
    marker = tree.root / "child-imported"
    (foreign / "extra.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    # Model an honest task adding its project folder for ordinary test imports.
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
        + ("pytest_plugins = ['helpers.extra']\n" if declaration == "conftest" else "")
    )
    if parent == "protected":
        source = tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            "ALLOWED_TOP |= {'helpers'}\nPROTECTED_TOP |= {'helpers'}\nPATH_OPTIONS = (",
        )
        tree.run.write_text(source)
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    # -p is requested after the conftest extended the path, like another plugin.
    if declaration == "-p":
        with (tree.tests / "conftest.py").open("a") as handle:
            handle.write(
                "def pytest_configure(config):\n"
                "    config.pluginmanager.import_plugin('helpers.extra', True)\n"
            )
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("inherited", ["/task/bin:/usr/bin:/bin", "/usr/bin:/bin"])
def test_custody_path_does_not_escape_its_subshell(tmp_path, monkeypatch, inherited):
    """V2-R5-C5/REV-4 (0055522c): pytest and children inherit the caller's PATH."""
    # Execute the real launcher with inert custody and an observing interpreter.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    interpreter = bindir / "python"
    interpreter.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$PATH"\n/bin/sh -c \'printf "%s\\n" "$PATH"\'\n'
    )
    interpreter.chmod(0o755)
    monkeypatch.setattr(runtime, "KIT_SH", "/bin/sh")
    monkeypatch.setattr(runtime, "KIT_BIN", str(bindir))
    monkeypatch.setattr(runtime, "KIT_PYTHON", str(interpreter))
    monkeypatch.setattr(
        runtime, "custody_command", lambda: '[ "$PATH" = ' + str(bindir) + " ]"
    )
    launcher = tmp_path / "pytest"
    launcher.write_text(runtime.runner_script())
    result = subprocess.run(
        ["/bin/sh", str(launcher)],
        env={"PATH": inherited},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [inherited, inherited]


@pytest.mark.parametrize("local", [True, False])
def test_kit_copies_the_verifiers_selected_python(tmp_path, local):
    """V2-R5-C6 (0055522c): keep the image's /usr/local Python ahead of distro Python."""
    from benchflow._runtime_tree import kit_prepare_command

    selection = kit_prepare_command().split("bf_py=;", 1)[1].split("\npy=0", 1)[0]
    for folder in ("usr/bin", "bin", "usr/local/bin"):
        destination = tmp_path / folder / "python3"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if folder != "usr/local/bin" or local:
            destination.write_text("#!/bin/sh\nexit 0\n")
            destination.chmod(0o755)
    selection = selection.replace(
        "/usr/local/bin/python3", str(tmp_path / "usr/local/bin/python3")
    )
    selection = selection.replace("/usr/bin/python3", str(tmp_path / "usr/bin/python3"))
    selection = selection.replace(" /bin/python3", " " + str(tmp_path / "bin/python3"))
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            "bfc_chain() { return 0; }; bf_py=;"
            + selection
            + '\nprintf "%s\\n" "$bf_py"',
        ],
        capture_output=True,
        text=True,
    )
    expected = tmp_path / ("usr/local/bin/python3" if local else "usr/bin/python3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected)


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("spelling", ["image_helper", "image_pkg.helper"])
def test_image_helper_uses_its_preagent_copy(tree, changed, spelling):
    """V2-R5-C7 (0055522c): unchanged image helpers load; changed helpers never execute."""
    import hashlib
    import shutil

    snapshot = tree.root / "baseline"
    if "." in spelling:
        (tree.app / "image_pkg").mkdir()
        (tree.app / "image_pkg/__init__.py").write_text("")
    source = tree.app / (spelling.replace(".", "/") + ".py")
    source.write_text("def pytest_configure(config):\n    config.image_helper = True\n")
    shutil.copytree(tree.app, snapshot)
    files = {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    # runner_source gets these pre-agent facts from the preparation probe.
    original = tree.run.read_text()
    tree.run.write_text(
        original.replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\nIMAGE_FILES = {files!r}\nPATH_OPTIONS = (",
        )
    )
    marker = tree.root / "changed-helper-imported"
    if changed:
        source.write_text(f"open({str(marker)!r}, 'w').close()\n")
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\npytest_plugins = [{spelling!r}]\n"
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n    assert request.config.image_helper\n"
    )
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode == (4 if changed else 0), result.stdout + result.stderr


@pytest.mark.parametrize(
    "option,value",
    [
        ("-k", "selected"),
        ("-m", "selected"),
        ("--override-ini", "addopts=-q"),
        ("--capture", "sys"),
    ],
)
def test_nonpath_option_values_are_never_normalized(tree, option, value):
    """V2-R5-C8 (0055522c): a value that names an existing file remains a value."""
    (tree.tests / value).write_text("")
    (tree.tests / "test_ok.py").write_text(
        "import pytest\n@pytest.mark.selected\ndef test_selected():\n    pass\n"
    )
    result = tree.pytest("test_ok.py", option, value)
    assert result.returncode == 0, result.stdout + result.stderr


async def test_separate_verifier_prepares_its_kit_before_runtime(tmp_path, monkeypatch):
    """V2-R5-REV-5 (0055522c): the fresh verifier gets its own protected runner."""
    from benchflow.sandbox import lockdown
    from tests.test_verifier_runtime_r5 import _task

    task = _task(
        tmp_path,
        'version="1.0"\n[verifier]\nsandbox_mode="separate"\npytest_plugin_releases=["fakeplug==1.0"]\n',
    )
    calls = []
    env = type("Sandbox", (), {})()

    async def kit(sandbox, **kwargs):
        assert sandbox is env
        calls.append(("kit", kwargs.get("service", "main")))
        sandbox.ready = True

    async def prepare(sandbox, task, **kwargs):
        assert getattr(sandbox, "ready", False), (
            "runtime probed without a protected kit"
        )
        calls.append(("runtime", kwargs["service"]))
        return {"runner": runtime.RUNNER}

    monkeypatch.setattr(lockdown, "prepare_runtime_tree", kit)
    monkeypatch.setattr(runtime, "prepare_verifier_runtime", prepare)
    result = await lockdown.prepare_verifier_runtime(
        env, task, None, service="grader", built_after_agent=True
    )
    assert calls == [("kit", "grader"), ("runtime", "grader")]
    assert result["runner"] == runtime.RUNNER


@pytest.mark.parametrize("name", ["bf_dep_mod", "colorsys"])
def test_assertion_rewriter_defers_to_protected_imports(tree, name):
    """V2-R5-C3 (0055522c): rewriting must not re-resolve a protected name on sys.path."""
    marker = tree.root / "rewriter-imported"
    (tree.app / (name + ".py")).write_text(f"open({str(marker)!r}, 'w').close()\n")
    (tree.tests / "conftest.py").write_text(
        f"import sys, pytest\nsys.path.insert(0, {str(tree.app)!r})\n"
        f"pytest.register_assert_rewrite({name!r})\n"
    )
    (tree.tests / "test_ok.py").write_text(f"def test_ok():\n    import {name}\n")
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_stdlib_inventory_does_not_need_sys_stdlib_module_names(tree):
    """V2-R5-C4/REV-2 (0055522c): Python 3.9 still enforces the kit's inventory."""
    marker = tree.root / "stdlib-39-imported"
    (tree.app / "colorsys.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    # Model Python 3.9's sys API while using the real protected runner.
    source = tree.run.read_text().replace(
        "SITE = ",
        "if hasattr(sys, 'stdlib_module_names'):\n    del sys.stdlib_module_names\n\nSITE = ",
        1,
    )
    tree.run.write_text(source)
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
    )
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    import colorsys\n")
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("attached", [False, True])
def test_plugin_option_values_follow_their_declared_arity(tree, attached):
    """V2-R5-C8 (0055522c): conftest-defined text values are opaque, too."""
    (tree.tests / "label").write_text("")
    (tree.tests / "conftest.py").write_text(
        "def pytest_addoption(parser):\n    parser.addoption('--label')\n"
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n    assert request.config.getoption('--label') == 'label'\n"
    )
    result = tree.pytest(
        "test_ok.py", *(["--label=label"] if attached else ["--label", "label"])
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dependency", ["absolute", "relative", "namespace"])
@pytest.mark.parametrize("changed", [False, True])
def test_image_helper_dependency_is_checked_against_the_same_copy(
    tree, dependency, changed
):
    """V2-R5-C7 (0055522c): a helper's unchanged parent cannot vouch for an edited dependency."""
    import hashlib
    import shutil

    package = tree.app / "image_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    dep = (
        (tree.app / "image_dep.py") if dependency == "absolute" else package / "dep.py"
    )
    if dependency == "namespace":
        (tree.app / "image_ns").mkdir()
        dep = tree.app / "image_ns/dep.py"
    dep.write_text("VALUE = 1\n")
    (package / "helper.py").write_text(
        {
            "absolute": "import image_dep\n",
            "relative": "from . import dep\n",
            "namespace": "from image_ns import dep\n",
        }[dependency]
    )
    snapshot = tree.root / "baseline"
    shutil.copytree(tree.app, snapshot)
    files = {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\nIMAGE_FILES = {files!r}\nPATH_OPTIONS = (",
        )
    )
    marker = tree.root / "dependency-imported"
    if changed:
        dep.write_text(f"open({str(marker)!r}, 'w').close()\n")
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\npytest_plugins = ['image_pkg.helper']\n"
    )
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert (result.returncode != 0) == changed, result.stdout + result.stderr


def test_image_native_helper_is_refused_before_loading_a_writable_copy(tree):
    """V2-R5-C7 (0055522c): a native loader needs custody as well as a digest."""
    import hashlib
    import shutil

    source = tree.app / "image_native.so"
    source.write_bytes(b"dummy extension, never loaded")
    snapshot = tree.root / "baseline"
    shutil.copytree(tree.app, snapshot)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\nIMAGE_FILES = {{'image_native.so': {digest!r}}}\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\npytest_plugins = ['image_native']\n"
    )
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    result = tree.pytest("test_ok.py")
    assert result.returncode == 4, result.stdout + result.stderr
    assert "extension is not root's alone" in result.stderr


def test_preagent_probe_records_the_physical_workspace(tmp_path):
    """V2-R5-C7 (0055522c): lexical workspace aliases map to the same baseline."""
    import sys

    workspace = tmp_path / "project"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"WORKSPACE = {str(alias)!r}\n" + runtime._PROBE,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    info = runtime._parse_probe(result.stdout, runner=True)
    assert info.workspace == str(workspace)
    assert "colorsys" in info.stdlib_names


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize(
    "needed,custom",
    [
        ("libfixture.so", None),
        ("/tmp/fixture/libfixture.so", None),
        ("../fixture/libfixture.so", None),
        ("fixture/libfixture.so", None),
        *[
            ("libfixture.so", tag)
            for tag in (
                15,
                29,
                0x6FFFFEFA,
                0x6FFFFEFB,
                0x6FFFFEFC,
                0x7FFFFFFD,
                0x7FFFFFFF,
            )
        ],
    ],
)
def test_native_helper_cannot_select_a_library_path(tmp_path, wide, needed, custom):
    """V2-R5-C7 (0055522c): native helpers cannot bypass custody through ELF search paths."""
    import ast
    import struct

    source = runtime.runner_source(
        site="/site",
        stdlib=[],
        tests="/tests",
        plugins={},
        allowed_top=[],
        protected_top=[],
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_system_linked_extension"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "native-custody", "exec"),
        namespace,
    )
    # Minimal data-only ELF program headers and dynamic table; never loaded.
    data = bytearray(1024)
    data[:16] = b"\x7fELF" + bytes([2 if wide else 1, 1, 1]) + b"\0" * 9
    phoff, phsize, step = (64, 56, 16) if wide else (52, 32, 8)
    struct.pack_into(
        "<HHIQQQIHHHHHH" if wide else "<HHIIIIIHHHHHH",
        data,
        16,
        3,
        62 if wide else 3,
        1,
        0,
        phoff,
        0,
        0,
        phoff,
        phsize,
        2,
        0,
        0,
        0,
    )
    for index, kind in enumerate((1, 2)):
        position, length = (0, len(data)) if kind == 1 else (256, step * 4)
        values = (
            (kind, 4, position, position, position, length, length, 8)
            if wide
            else (kind, position, position, position, length, length, 4, 4)
        )
        struct.pack_into(
            "<IIQQQQQQ" if wide else "<IIIIIIII", data, phoff + index * phsize, *values
        )
    tags = [(5, 512), (1, 1)] + ([(custom, 1)] if custom else []) + [(0, 0)]
    for index, pair in enumerate(tags):
        struct.pack_into("<qQ" if wide else "<iI", data, 256 + index * step, *pair)
    names = b"\0" + needed.encode() + b"\0"
    data[512 : 512 + len(names)] = names
    candidate = tmp_path / "helper.so"
    candidate.write_bytes(data)
    assert namespace["_system_linked_extension"](str(candidate)) == (
        custom is None and "/" not in needed
    )


@pytest.mark.parametrize("broken", [False, True])
async def test_separate_runtime_is_ready_before_any_agent_transfer(
    tmp_path, monkeypatch, broken
):
    """V2-R5-REV-5 (0055522c): startup must authenticate the image before transfer."""
    from benchflow.rollout import _separate_verifier as separate
    from benchflow.sandbox import lockdown
    from tests.test_sv_custody import _separate_trial, _writes_reward

    _, rollout, box = await _separate_trial(tmp_path)
    events = []
    original_install = separate._install_payload

    async def prepare(env, task, *, workspace):
        assert env is box and env.started
        assert not env.uploads
        events.append("prepare")
        if broken:
            raise RuntimeError("kit preparation unavailable")

    async def install(*args, **kwargs):
        events.append("transfer")
        return await original_install(*args, **kwargs)

    async def verify(*args, **kwargs):
        events.append("verify")
        return await _writes_reward("0")(*args, **kwargs)

    monkeypatch.setattr(
        lockdown, "prepare_separate_verifier_runtime", prepare, raising=False
    )
    monkeypatch.setattr(separate, "_install_payload", install)
    rewards, error = await separate.run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=verify
    )
    if broken:
        assert rewards is None and "kit preparation unavailable" in error
        assert events == ["prepare"]
        assert not box.uploads
    else:
        assert (rewards, error) == ({"reward": 0.0}, None)
        assert events == ["prepare", "transfer", "verify"]


def test_authenticated_dotted_plugin_keeps_normal_parent_attributes(tree):
    """Guards V2-R5-REV-1 and 8e07b241: explicit spec loading preserves Python imports."""
    package = tree.site / "safeplug"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "child.py").write_text("VALUE = True\n")
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            "ALLOWED_TOP |= {'safeplug'}\nPROTECTED_TOP |= {'safeplug'}\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "conftest.py").write_text("pytest_plugins = ['safeplug.child']\n")
    (tree.tests / "test_ok.py").write_text(
        "def test_ok():\n    import safeplug.child\n    assert safeplug.child.VALUE\n"
    )
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--maybe"], None),
        (["--words", "one", "two"], ["one", "two"]),
        (["--words", "-1", "2"], ["-1", "2"]),
    ],
)
def test_option_consumption_uses_pytests_parser(tree, extra, expected):
    """V2-R5-C8 (0055522c): optional and multi-value options preserve every text value."""
    for name in ("selected", "one", "two", "-1", "2"):
        (tree.tests / name).write_text("")
    (tree.tests / "conftest.py").write_text(
        "def pytest_addoption(parser):\n    parser.addoption('--maybe', nargs='?', const='const')\n    parser.addoption('--words', nargs='+')\n"
    )
    (tree.tests / "test_ok.py").write_text(
        f"def test_selected(request):\n    assert request.config.getoption('words') == {expected!r}\n"
    )
    result = tree.pytest("test_ok.py", *extra, "-k", "selected")
    assert result.returncode == 0, result.stdout + result.stderr


def test_pyargs_module_names_are_not_filesystem_paths(tree):
    """V2-R5-C8 (0055522c): --pyargs keeps module names even when a directory exists."""
    package = tree.site / "verifier_examples"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "test_ok.py").write_text("def test_ok():\n    pass\n")
    (tree.tests / "verifier_examples").mkdir()
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            "PROTECTED_TOP |= {'verifier_examples'}\nPATH_OPTIONS = (",
        )
    )
    result = tree.pytest("--pyargs", "verifier_examples")
    assert result.returncode == 0, result.stdout + result.stderr


def test_path_option_actions_keep_the_callers_ignore_path(tree):
    """V2-R5-C8 (0055522c): explicit path options still resolve before leaving cwd."""
    ignored = tree.tests / "ignored"
    ignored.mkdir()
    (ignored / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    result = tree.pytest("test_ok.py", "ignored", "--ignore", "ignored/test_bad.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_image_helper_keeps_the_submitted_package_visible_to_tests(tree):
    """V2-R5-C7 (0055522c): helper custody must not replace the solution with its baseline."""
    import hashlib
    import shutil

    package = tree.app / "image_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper.py").write_text(
        "def pytest_configure(config):\n    config.image_helper = True\n"
    )
    subject = package / "subject.py"
    subject.write_text("ANSWER = False\n")
    snapshot = tree.root / "baseline"
    shutil.copytree(tree.app, snapshot)
    files = {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\nIMAGE_FILES = {files!r}\nPATH_OPTIONS = (",
        )
    )
    subject.write_text("ANSWER = True\n")
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\npytest_plugins = ['image_pkg.helper']\n"
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok():\n    from image_pkg import subject\n    assert subject.ANSWER\n"
    )
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_image_dependency_imports_and_metadata_come_from_the_sealed_copy(tree):
    """V2-R5-C7 (0055522c): ordinary image packages are available without ambient imports."""
    import hashlib
    import shutil

    original = tree.root / "original-image-site"
    copied = tree.root / "copied-image-site"
    original.mkdir()
    (original / "image_dependency.py").write_text("ANSWER = True\n")
    info = original / "image_dependency-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: image-dependency\nVersion: 1.0\n")
    shutil.copytree(original, copied)
    (tree.app / "image_helper.py").write_text(
        "import image_dependency\nimport importlib.metadata\ndef pytest_configure(config):\n    assert image_dependency.ANSWER\n    assert importlib.metadata.version('image-dependency') == '1.0'\n"
    )
    snapshot = tree.root / "baseline"
    shutil.copytree(tree.app, snapshot)
    files = {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    marker = tree.root / "ambient-image-dependency-imported"
    (original / "image_dependency.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nANSWER = False\n"
    )
    (info / "METADATA").write_text("Name: image-dependency\nVersion: 9.0\n")
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\nIMAGE_FILES = {files!r}\nIMAGE_SITES = [{str(copied)!r}]\nIMAGE_TOP = frozenset(['image_dependency'])\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(original)!r})\nsys.path.insert(0, {str(tree.app)!r})\npytest_plugins = ['image_helper']\n"
    )
    (tree.tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_image_alias_uses_only_the_sealed_image_metadata(tree):
    """V2-R5-C1/C7 (0055522c): image entry-point aliases use the authenticated copy."""
    image = tree.root / "copied-image-site"
    image.mkdir()
    (image / "image_plugin.py").write_text(
        "def pytest_configure(config):\n    config.image_alias = True\n"
    )
    info = image / "image_plugin-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: image-plugin\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text("[pytest11]\nimage_alias = image_plugin\n")
    ambient = tree.tests / "ambient-image-1.0.dist-info"
    ambient.mkdir()
    (ambient / "METADATA").write_text("Name: ambient-image\nVersion: 1.0\n")
    (ambient / "entry_points.txt").write_text(
        "[pytest11]\nimage_alias = ambient_plugin\n"
    )
    marker = tree.root / "ambient-alias-imported"
    (tree.tests / "ambient_plugin.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n"
    )
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"IMAGE_SITES = [{str(image)!r}]\nIMAGE_TOP = frozenset(['image_plugin'])\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n    assert request.config.image_alias\n"
    )
    result = tree.pytest("-p", "image_alias", "test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
