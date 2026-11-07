"""SV4 image-only cache; guards the fix at 0612ec19.

The old parser/bake/seed contract is deliberately retired. Tests retain the
image identity, locking, inventory and backend cases and check that every
historical shell family is opaque input with no installer side effects.
"""

import asyncio
import copy
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow.sandbox import _layer_lock, env_layers
from benchflow.sandbox import verifier_layers as vl
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.env_layers import BaseImage, DockerLayers, LayerError, layer_tag
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task import Task
from benchflow.task.config import SandboxConfig
from tests.test_env_layers_agent import FakeDocker

LAYER_ENV = "BENCHFLOW_VERIFIER_LAYER"
VERIFIER = env_layers.VERIFIER
IMAGE_UVX = "uvx -p 3.13 pytest==8.4.1 /tests/test_outputs.py\n"
BASE = BaseImage(
    ref="task:1",
    image_id="sha256:" + "1" * 64,
    digest="sha256:" + "2" * 64,
    arch="amd64",
    libc="glibc",
)


@pytest.fixture(autouse=True)
def isolated_layers(tmp_path, monkeypatch):
    monkeypatch.setenv(LAYER_ENV, "1")
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_MIN_FREE_GB", "0")
    for state in (
        DockerLayers._claims,
        DockerLayers._libc_cache,
    ):
        state.clear()
    _layer_lock.clear_process_locks()
    yield
    DockerLayers._claims.clear()
    _layer_lock.clear_process_locks()


def _task(root: Path, name: str = "bake-task", *, test_sh: str = IMAGE_UVX,
          verifier: str = "", sandbox: str = "") -> Path:  # fmt: skip
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "solution").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "tests" / "test.sh").write_text(test_sh)
    (task / "tests" / "test_outputs.py").write_text("def test_x():\n    assert True\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\necho 42 > /app/a\n")
    (task / "instruction.md").write_text("Write the answer.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 60\n'
        "[verifier]\ntimeout_sec = 60\n" + verifier + "[environment]\n" + sandbox
    )
    return task


def _key(task_dir: Path, base: BaseImage = BASE, *, separate: bool = True) -> str:
    recipe, record = vl.plan_verifier_layer(Task(task_dir), separate=separate)
    assert recipe is not None, record
    return vl.verifier_layer_key(recipe, base)


class TestKey:
    def test_two_copies_of_one_task_share_a_key(self, tmp_path):
        a = _task(tmp_path / "a", "one-name")
        b = _task(tmp_path / "b", "another-name")
        assert _key(a) == _key(b)

    @pytest.mark.parametrize(
        "change",
        [
            lambda t: (t / "tests" / "test_outputs.py").write_text(
                "def test_y(): pass\n"
            ),
            lambda t: (t / "tests" / "data.txt").write_text("more\n"),
            lambda t: (t / "tests" / "test_outputs.py").chmod(0o755),
            lambda t: (t / "tests" / "test.sh").write_text(
                IMAGE_UVX.replace("pytest==8.4.1", "pytest==8.4.2")
            ),
            lambda t: (t / "task.toml").write_text(
                (t / "task.toml")
                .read_text()
                .replace("[verifier]\ntimeout_sec = 60", "[verifier]\ntimeout_sec = 61")
            ),
        ],
        ids=[
            "test-file",
            "new-test-file",
            "test-file-mode",
            "toolchain-pin",
            "verifier-config",
        ],
    )
    def test_each_verifier_input_changes_the_key(self, tmp_path, change):
        task = _task(tmp_path)
        before = _key(task)
        change(task)
        assert _key(task) != before

    @pytest.mark.parametrize(
        "change",
        [
            {"digest": "sha256:" + "3" * 64},
            {"libc": "musl"},
            {"arch": "arm64"},
        ],
    )
    def test_the_task_image_changes_the_key(self, tmp_path, change):
        task = _task(tmp_path)
        base = BaseImage(**{**BASE.__dict__, **change})
        assert _key(task, base) != _key(task)

    @pytest.mark.parametrize(
        "change",
        [
            lambda t: (t / "instruction.md").write_text("Something else.\n"),
            lambda t: (t / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n"),
            lambda t: (t / "environment" / "notes.txt").write_text("agent side\n"),
            lambda t: (t / "task.toml").write_text(
                (t / "task.toml")
                .read_text()
                .replace("[agent]\ntimeout_sec = 60", "[agent]\ntimeout_sec = 90")
            ),
        ],
        ids=["instruction", "solution", "environment-file", "agent-config"],
    )
    def test_nothing_else_changes_the_key(self, tmp_path, change):
        task = _task(tmp_path)
        before = _key(task)
        change(task)
        assert _key(task) == before

    def test_the_image_id_alone_does_not_change_the_key(self, tmp_path):
        """A rebuilt task image with the same content (a new id) is a hit."""
        task = _task(tmp_path)
        rebuilt = BaseImage(**{**BASE.__dict__, "image_id": "sha256:" + "9" * 64})
        assert _key(task, rebuilt) == _key(task)


class BakeDocker(FakeDocker):
    """Daemon stand-in: declarative contexts and labels preserve the base."""

    def __init__(self):
        super().__init__()
        self.context_builds = []
        self.bake_files = []

    async def run(self, *args, stdin=None, timeout=0):
        if args in (
            ("version", "--format", "{{.Client.Version}}/{{.Server.Version}}"),
            ("compose", "version", "--short"),
            ("buildx", "version"),
        ):
            return self._ok("test-tool-version")
        if args == ("buildx", "inspect"):
            self.calls.append(list(args))
            return self._ok("Name: default\nDriver: docker\n\nNodes:\n")
        if args == ("info", "--format", "{{.OSType}}/{{.Architecture}}"):
            self.calls.append(list(args))
            return self._ok("linux/x86_64\n")
        if args[0] != "build":
            return await super().run(*args, stdin=stdin, timeout=timeout)
        self.calls.append(list(args))
        tag = args[args.index("--tag") + 1]
        if stdin is None:
            text = (Path(args[-1]) / "Dockerfile").read_text()
            self.context_builds.append(text)
            self.add_image(tag, layers=["sha256:" + "5" * 64])
            return self._ok()
        text = stdin.decode()
        self.bake_files.append(text)
        base = self._by_ref(text.splitlines()[0].removeprefix("FROM "))
        assert base is not None
        labels = dict(base.info["Config"]["Labels"])
        for i, arg in enumerate(args):
            if arg == "--label":
                name, _, value = args[i + 1].partition("=")
                labels[name] = value
        built = self.add_image(
            tag, layers=list(base.info["RootFS"]["Layers"]), labels=labels
        )
        built.info["Config"] = {**copy.deepcopy(base.info["Config"]), "Labels": labels}
        self.records.append({"id": f"rec{len(self.records)}", "description": text})
        return self._ok(built.id)


@pytest.fixture
def docker(monkeypatch):
    fake = BakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    return fake


def _recipe(tmp_path: Path, *, separate: bool = True, test_sh: str = IMAGE_UVX):
    recipe, record = vl.plan_verifier_layer(
        Task(_task(tmp_path, test_sh=test_sh)), separate=separate
    )
    assert recipe is not None, record
    return recipe


class _SharedEnv:
    task_env_config = SandboxConfig()

    def __init__(self) -> None:
        self.recipes: list[Any] = []

    def use_verifier_layer(self, recipe) -> None:
        self.recipes.append(recipe)


def _rollout_for(task_dir: Path, env) -> Any:
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._task = Task(task_dir)
    rollout._env = env
    rollout._env_externally_owned = False
    rollout._from_branch_snapshot = False
    rollout._config = SimpleNamespace(
        fresh_install=False,
        primary_agent="oracle",
        skip_agent_install=False,
        environment="docker",
        harness=None,
        sandbox_setup_timeout=None,
    )
    rollout._layer_records = {}
    rollout._litellm_proxy_record = None
    return rollout


# Every spelling from the reported family plus previous parser failures.
SCRIPTS = [
    f"#!/bin/bash\nset -e\n{command} {option} {tail}\necho 1 > /logs/verifier/reward.txt\n"
    for command, tail in [
        ("uv run --no-project", "python /tests/check.py"),
        ("uvx", "pytest==8.4.1 /tests/check.py"),
        ("uv tool run", "pytest==8.4.1 /tests/check.py"),
    ]
    for option in [
        "-p 3.13",
        "-p3.13",
        "--python 3.13",
        "--python=3.13",
        "--python '3.13'",
        '--python="3.13"',
        "",
    ]
] + [
    "pip install pytest==8.4.1\npython /tests/check.py\n",
    "python3.13 -m pip install pytest\n",
    "uv python install 3.13\nuv run --offline python /tests/check.py\n",
    "if false; then\nuv python install 3.13\nfi\n",
    "cat <<EOF\nuv python install 3.13\nEOF\n",
    "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\nsource $HOME/.local/bin/env\nuvx pytest\n",
    "cd /different; PATH=/other HOME=/else uv run -p 3.13 python check.py\n",
    "UV_PYTHON_PREFERENCE=only-system uv run --no-project python /tests/check.py\n",
    "python /tests/\u6d4b\u8bd5.py\n",
    "printf '%s' 'uv python install 3.13'\n",
]


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("backend", [DockerSandbox, RemoteDockerSandbox])
def test_no_installation_is_planned(tmp_path, monkeypatch, script, separate, backend):
    """Neither script spelling nor backend can bake tools."""
    path = _task(tmp_path, test_sh=script)
    task = Task(path)
    box = backend.__new__(backend)
    box.layer_records = {}
    box.task_env_config = task.config.sandbox
    if separate:
        from benchflow.rollout._separate_verifier import _use_verifier_layer

        rollout = SimpleNamespace(
            _config=SimpleNamespace(environment="docker", fresh_install=False)
        )
        _use_verifier_layer(rollout, box, task, task)
    else:
        rollout = _rollout_for(path, box)
        rollout._request_verifier_layer()
    recipe = box._verifier_layer_recipe
    if not separate:
        assert recipe is None
    else:
        assert recipe is not None
        dockerfile = vl.verifier_layer_dockerfile(
            recipe, "pinned:base", "a" * 64, base=BASE
        )
        assert [line.split()[0] for line in dockerfile.splitlines()] == [
            "FROM",
            "LABEL",
        ]
        assert "toolchain" not in recipe.summary()
    assert (path / "tests/test.sh").read_text() == script


@pytest.mark.parametrize("value", [None, "", "0", "off", "no", "1", "YES"])
@pytest.mark.parametrize("separate", [False, True])
def test_opt_in_and_fresh(tmp_path, monkeypatch, value, separate):
    """Guards SV4-R1-B-1: off by default and fresh always bypasses reuse."""
    if value is None:
        monkeypatch.delenv(LAYER_ENV, raising=False)
    else:
        monkeypatch.setenv(LAYER_ENV, value)
    task = Task(_task(tmp_path))
    recipe, record = vl.plan_verifier_layer(task, separate=separate)
    if value in ("1", "YES"):
        assert (recipe is not None) == separate
        recipe, record = vl.plan_verifier_layer(
            task, separate=separate, fresh_install=True
        )
        assert recipe is None and record["status"] == "fresh"
    else:
        assert (recipe, record) == (None, None)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"PATH": "/usr/bin", "HOME": "/app"},
        {"UV_PYTHON": "3.13.0", "PIP_INDEX_URL": "https://example.invalid"},
    ],
)
@pytest.mark.parametrize("allow_internet", [False, True])
def test_image_only_plan_does_not_resolve_tools(tmp_path, env, allow_internet):
    """Discovery inputs remain the verifier's concern."""
    task = Task(_task(tmp_path))
    recipe, record = vl.plan_verifier_layer(
        task, separate=True, sandbox_env=env, allow_internet=allow_internet
    )
    assert recipe is not None and record is None
    assert recipe.summary()["cache"] == "image-only"


async def test_labels_preserve_all_execution_settings(docker, tmp_path):
    """Files, USER, PATH, HOME, cwd and startup stay identical."""
    image = docker.add_image("task:1")
    image.info["Config"].update(
        User="1001:1002",
        Env=["PATH=/usr/bin", "HOME=/app"],
        WorkingDir="/app",
        Entrypoint=["/entry"],
        Cmd=["/command"],
        Shell=["/bin/bash", "-c"],
        Volumes={"/data": {}},
        Healthcheck={"Test": ["CMD", "true"]},
    )
    recipe = _recipe(tmp_path)
    first = await vl.ensure_verifier_layer(
        DockerLayers(), recipe, "task:1", alias="trial:one"
    )
    second = await vl.ensure_verifier_layer(
        DockerLayers(), recipe, "task:1", alias="trial:two"
    )
    assert (first.result.status, second.result.status) == ("miss", "hit")
    assert len(docker.bake_files) == 1
    assert second.record()["cache"] == "image-only"
    actual = docker.tags[second.result.image].info
    assert actual["RootFS"] == image.info["RootFS"]
    assert {k: v for k, v in actual["Config"].items() if k != "Labels"} == {
        k: v for k, v in image.info["Config"].items() if k != "Labels"
    }
    assert not any("verifier-tools" in " ".join(c) for c in docker.calls)


async def test_onbuild_uses_ordinary_image(docker, tmp_path):
    """A metadata wrapper must not execute inherited code."""
    image = docker.add_image("task:1")
    image.info["Config"]["OnBuild"] = ["RUN touch /dummy"]
    with pytest.raises(LayerError, match="ONBUILD"):
        await vl.ensure_verifier_layer(
            DockerLayers(), _recipe(tmp_path), "task:1", alias=None
        )
    assert not docker.bake_files


@pytest.mark.parametrize(
    "instruction",
    [
        "RUN true",
        "RUN --network=none true",
        "ARG VERSION",
        "ADD x /x",
        "ONBUILD RUN true",
        "COPY --from=external:latest /x /x",
    ],
)
async def test_unresolved_builds_use_ordinary_path(docker, tmp_path, instruction):
    """Runtime or external inputs never enter a reusable build."""
    ref = "task@sha256:" + "1" * 64
    docker.add_image(ref)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(f"FROM {ref}\n{instruction}\n")
    with pytest.raises(LayerError):
        await vl.ensure_verifier_layer(
            DockerLayers(),
            _recipe(tmp_path),
            None,
            alias=None,
            context=context,
            **context_build(context),
        )
    assert not docker.context_builds


async def test_hardening_has_no_layer_exemption(tmp_path):
    """A cache-looking folder gets ordinary cleanup."""
    from benchflow.sandbox import lockdown

    commands = []

    from tests._receipt_fakes import receipt_aware

    class Box:
        # The clean-up first judges its interpreter through checked reads:
        # Answer them as a sandbox shell would.
        @receipt_aware
        async def exec(self, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    box = Box()
    await lockdown.cleanup_verifier_python_hooks(box, None)
    plain = commands.pop()
    recipe = _recipe(tmp_path)
    # Also model an old live layer record; it cannot exempt planted files.
    import inspect

    params = {"result": SimpleNamespace(), "recipe": recipe}
    if "tree_digest" in inspect.signature(vl.VerifierLayer).parameters:
        params["tree_digest"] = "d" * 64
    box.verifier_layer = vl.VerifierLayer(**params)
    await lockdown.cleanup_verifier_python_hooks(box, None)
    assert commands.pop() == plain


async def test_a_separate_verifier_image_from_its_context_is_built_once(
    docker, tmp_path
):
    """Four concurrent trials of one pinned context: one build."""
    base_ref = "ubuntu@sha256:" + "1" * 64
    docker.add_image(base_ref)
    recipe = _recipe(tmp_path, separate=True)
    contexts = []
    for n in range(4):
        context = tmp_path / f"trial-{n}" / "environment"
        context.mkdir(parents=True)
        (context / "Dockerfile").write_text(f"FROM {base_ref}\nCOPY x /x\n")
        (context / "x").write_text("same\n")
        contexts.append(context)
    layers = DockerLayers()
    results = await asyncio.gather(
        *(
            vl.ensure_verifier_layer(
                layers,
                recipe,
                None,
                alias=f"bf__layer__t{n}:me",
                context=c,
                **context_build(c),
            )
            for n, c in enumerate(contexts)
        )
    )
    statuses = sorted(r.result.status for r in results)
    assert statuses == ["hit", "hit", "hit", "miss"]
    assert len({r.result.key for r in results}) == 1
    assert len(docker.context_builds) == 1
    assert sum(1 for c in docker.calls if c[0] == "build") == 2  # context + bake
    # The context's pin is gone after the build.
    assert not any(t.startswith("benchflow-layer-base/") for t in docker.tags)

    changed = tmp_path / "trial-changed" / "environment"
    shutil.copytree(contexts[0], changed)
    (changed / "x").write_text("different\n")
    other = await vl.ensure_verifier_layer(
        layers,
        recipe,
        None,
        alias="bf__layer__t9:me",
        context=changed,
        **context_build(changed),
    )
    assert other.result.status == "miss" and other.result.key != results[0].result.key


async def test_a_failed_bake_is_a_layer_error(docker, tmp_path, monkeypatch):
    docker.add_image("task:1")
    original = docker.run

    async def failing(_layers, *args, stdin=None, timeout=0):
        if args[:1] == ("build",):
            return env_layers._Run(stdout="", stderr="no space left", code=1)
        return await original(*args, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(DockerLayers, "_docker", failing)
    with pytest.raises(LayerError, match="no space left"):
        await vl.ensure_verifier_layer(
            DockerLayers(), _recipe(tmp_path), "task:1", alias=None
        )


async def test_bench_cache_lists_and_prunes_verifier_layers(docker, tmp_path):
    docker.add_image("task:1")
    made = await vl.ensure_verifier_layer(
        DockerLayers(), _recipe(tmp_path), "task:1", alias=None
    )
    DockerLayers.release(made.result.key, daemon=made.result.daemon_id)
    inventory = await DockerLayers().inventory()
    (entry,) = inventory.layers
    assert (entry.kind, entry.key, entry.verifier) == (
        VERIFIER,
        made.result.key,
        "bake-task",
    )
    from benchflow.cli import cache as cache_cli

    items = await cache_cli._docker_items((cache_cli.IMAGES,), [])
    (item,) = [i for i in items if i.type == "layer"]
    assert item.detail["layer"] == VERIFIER and item.detail["what"] == "bake-task"
    report = await DockerLayers().remove_selected(layers=[(VERIFIER, made.result.key)])
    assert report.removed == [made.result.key]
    assert layer_tag(VERIFIER, made.result.key) not in docker.tags


def context_build(context):
    """Adapter for historical daemon-only tests, not a production build route.

    Production passes the sandbox's ordinary Compose builder. These parser,
    key and locking tests supply their existing in-memory daemon builder.
    """

    async def build():
        layers = DockerLayers()
        ref = "test-context:" + str(abs(hash(str(context))))
        await layers._docker(
            "build", "--tag", ref, "--platform", "linux/amd64", str(context)
        )
        return await layers.inspect_base(ref)

    async def identity():
        return "test-compose-build"

    return {"build_context": build, "build_identity": identity}
