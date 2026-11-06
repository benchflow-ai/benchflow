"""The model proxy's cached layer: key, build, reuse and prune (env_layers, PY8 U2b).

Against the stand-in ``docker`` of ``tests/test_env_layers_agent.py``
(:class:`FakeDocker`), extended to read a proxy layer's Dockerfile and its
build context. The proxy's use of the layer in a trial is in
``tests/test_proxy_layer_use.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.providers import proxy_requirements as reqs
from benchflow.providers.litellm_runtime import proxy_layer_recipe
from benchflow.sandbox import _layer_lock, env_layers
from benchflow.sandbox.env_layers import (
    LAYER_BASE_LABEL,
    LAYER_FORMAT,
    LAYER_KEY_LABEL,
    LAYER_LABEL,
    LAYER_PROXY_GROUP_LABEL,
    PROXY,
    PROXY_LAYER_REQUIREMENTS_DIR,
    PROXY_LAYER_STAMP,
    PROXY_LAYER_VENV,
    BaseImage,
    DockerLayers,
    LayerError,
    ProxyLayerRecipe,
    layer_tag,
    proxy_layer_dockerfile,
    proxy_layer_key,
)
from tests.test_env_layers_agent import FakeDocker

RECIPE = ProxyLayerRecipe(
    group="litellm-proxy",
    requirements_name="litellm-proxy.txt",
    requirements_text="litellm==1.91.5 \\\n    --hash=sha256:" + "a" * 64 + "\n",
    install_cmd="set -eu\necho install",
    python="3.12",
    uv_version="0.13.4",
)
BASE = BaseImage(
    ref="bf__task",
    image_id="sha256:" + "1" * 64,
    digest="sha256:" + "2" * 64,
    arch="amd64",
    libc="glibc",
)


@pytest.fixture(autouse=True)
def _clean_class_state(tmp_path_factory, monkeypatch):
    # Each test's layer locks (_layer_lock) in a folder of its own.
    locks = tmp_path_factory.mktemp("layer-locks")
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(locks))
    DockerLayers._claims.clear()
    DockerLayers._libc_cache.clear()
    _layer_lock.clear_process_locks()
    yield
    DockerLayers._claims.clear()
    DockerLayers._libc_cache.clear()
    _layer_lock.clear_process_locks()


# -- the key -------------------------------------------------------------


class TestKey:
    """Every input of a proxy layer's bytes makes a new key."""

    @pytest.mark.parametrize(
        "change",
        [
            {"requirements_text": RECIPE.requirements_text + "boto3==1.40.0\n"},
            {"extra_requirements": ("humanize==4.11.0",)},
            {"python": "3.13"},
            {"uv_version": "0.13.2"},
            {"install_cmd": RECIPE.install_cmd + "\ntrue"},
            {"group": "litellm-proxy-vertex"},
        ],
        ids=["requirements", "extras", "python", "uv", "recipe", "group"],
    )
    def test_recipe_inputs_change_the_key(self, change):
        other = dataclasses.replace(RECIPE, **change)
        assert proxy_layer_key(other, BASE) != proxy_layer_key(RECIPE, BASE)

    @pytest.mark.parametrize(
        "change",
        [
            {"digest": "sha256:" + "3" * 64},
            {"libc": "musl"},
            {"arch": "arm64/v8"},
        ],
        ids=["task-image", "libc", "arch"],
    )
    def test_task_image_inputs_change_the_key(self, change):
        other = dataclasses.replace(BASE, **change)
        assert proxy_layer_key(RECIPE, other) != proxy_layer_key(RECIPE, BASE)

    def test_the_layer_format_changes_the_key(self, monkeypatch):
        before = proxy_layer_key(RECIPE, BASE)
        monkeypatch.setattr(env_layers, "LAYER_FORMAT", LAYER_FORMAT + 1)
        assert proxy_layer_key(RECIPE, BASE) != before

    def test_what_does_not_change_the_bytes_keeps_the_key(self):
        """The order and repeats of the extras, the time limit, the image id
        and its tag are not part of the key."""
        a = dataclasses.replace(RECIPE, extra_requirements=("b==1", "a==2"))
        b = dataclasses.replace(
            RECIPE, extra_requirements=("a==2", "b==1", "a==2"), timeout_sec=5
        )
        same_base = dataclasses.replace(
            BASE, ref="other-tag", image_id="sha256:" + "9" * 64
        )
        assert proxy_layer_key(a, BASE) == proxy_layer_key(b, same_base)

    def test_the_stamp_holds_the_install_and_not_the_task_image(self):
        """The trial compares the stamp: it is the same on every task image
        and changes with what is installed."""
        assert RECIPE.stamp == dataclasses.replace(RECIPE).stamp
        assert RECIPE.stamp != dataclasses.replace(RECIPE, python="3.13").stamp
        assert (
            RECIPE.stamp
            != dataclasses.replace(RECIPE, extra_requirements=("x==1",)).stamp
        )

    def test_the_proxy_and_agent_kinds_never_share_a_tag(self):
        key = proxy_layer_key(RECIPE, BASE)
        assert layer_tag(PROXY, key) == f"benchflow-layer:proxy-{key}"

    @pytest.mark.parametrize(
        "change",
        [
            {"requirements_name": "../etc/passwd"},
            {"requirements_text": " "},
            {"install_cmd": ""},
            {"python": ""},
            {"uv_version": ""},
        ],
    )
    def test_an_incomplete_recipe_is_refused(self, change):
        with pytest.raises(ValueError):
            dataclasses.replace(RECIPE, **change)


class TestRealRecipe:
    """The recipe BenchFlow builds from a proxy install (litellm_runtime)."""

    def test_it_installs_the_shipped_hashed_file_with_the_pins(self):
        install = reqs.proxy_install(None, {})
        recipe = proxy_layer_recipe(install)
        assert recipe.group == reqs.PROXY_GROUP
        assert recipe.requirements_text == reqs.requirements_text(reqs.PROXY_GROUP)
        assert recipe.python == "3.12.15" and recipe.uv_version == "0.13.4"
        uploaded = f"{PROXY_LAYER_REQUIREMENTS_DIR}/litellm-proxy.txt"
        assert f"--require-hashes -r {uploaded}" in recipe.install_cmd
        # A reused venv compiles its bytecode once, in the layer.
        assert "UV_COMPILE_BYTECODE=1" in recipe.install_cmd
        assert f"venv -q --python 3.12.15 {PROXY_LAYER_VENV}" in recipe.install_cmd
        assert "/releases/download/0.13.4" in recipe.install_cmd

    def test_a_vertex_route_gets_the_vertex_set(self):
        install = reqs.proxy_install("google-vertex", {})
        recipe = proxy_layer_recipe(install)
        assert recipe.group == reqs.VERTEX_GROUP
        assert recipe.requirements_name == "litellm-proxy-vertex.txt"
        assert recipe.stamp != proxy_layer_recipe(reqs.proxy_install(None, {})).stamp

    def test_extras_install_under_the_locked_set(self):
        install = reqs.proxy_install(
            None, {reqs.EXTRA_REQUIREMENTS_ENV: "humanize==4.11.0"}
        )
        recipe = proxy_layer_recipe(install)
        assert recipe.extra_requirements == ("humanize==4.11.0",)
        uploaded = f"{PROXY_LAYER_REQUIREMENTS_DIR}/litellm-proxy.txt"
        assert f"-c {uploaded} humanize==4.11.0" in recipe.install_cmd


# -- the Dockerfile ------------------------------------------------------


class TestDockerfile:
    def test_one_run_step_with_the_requirements_and_no_cache(self):
        key = proxy_layer_key(RECIPE, BASE)
        text = proxy_layer_dockerfile(RECIPE, "base:tag", key)
        lines = text.splitlines()
        assert lines[0] == "FROM base:tag"
        (run,) = [line for line in lines if line.startswith("RUN ")]
        assert (
            f"--mount=type=bind,source=litellm-proxy.txt,"
            f"target={PROXY_LAYER_REQUIREMENTS_DIR}/litellm-proxy.txt" in run
        )
        argv = json.loads(run[run.index("[") :])
        assert argv[:2] == ["/bin/sh", "-c"]
        script = argv[2]
        assert script.startswith(f": {LAYER_LABEL}={PROXY} key={key};")
        assert "export UV_NO_CACHE=1" in script
        assert f"printf '%s\\n' {RECIPE.stamp} > {PROXY_LAYER_STAMP}" in script
        assert RECIPE.install_cmd in script

    @pytest.mark.parametrize("shell", ["sh", "bash"])
    def test_a_failed_install_step_stops_the_build_before_the_stamp(
        self, shell, tmp_path, monkeypatch
    ):
        """Guards U2b-R1-A-1: the step's script ran the install as the left
        side of ``( ... ) && printf <stamp>``, where sh ignores ``set -e``, so
        a failed middle step (the extras, a locked-set install cut short)
        still wrote the stamp after ``import litellm`` passed, and a broken
        layer was cached and reused. The script runs here with the stamp
        moved under tmp_path."""
        if shutil.which(shell) is None:
            pytest.skip(f"no {shell} on this machine")
        stamp = tmp_path / "layers" / "litellm-proxy.stamp"
        monkeypatch.setattr(env_layers, "PROXY_LAYER_STAMP", str(stamp))

        def run(install_cmd: str) -> subprocess.CompletedProcess[str]:
            recipe = dataclasses.replace(RECIPE, install_cmd=install_cmd)
            key = proxy_layer_key(recipe, BASE)
            text = proxy_layer_dockerfile(recipe, "b", key)
            (run_line,) = [ln for ln in text.splitlines() if ln.startswith("RUN ")]
            argv = json.loads(run_line[run_line.index("[") :])
            return subprocess.run(
                [shell, "-c", argv[2]],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmp_path,
            )

        failed = run("set -eu\necho locked\nsh -c 'exit 3'\necho import-ok")
        assert failed.returncode != 0
        assert "import-ok" not in failed.stdout
        assert not stamp.exists()
        # Control: the same script with every step passing writes the stamp.
        ok = run("set -eu\necho locked\ntrue\necho import-ok")
        assert ok.returncode == 0, ok.stderr
        assert "import-ok" in ok.stdout
        recipe = dataclasses.replace(
            RECIPE, install_cmd="set -eu\necho locked\ntrue\necho import-ok"
        )
        assert stamp.read_text() == recipe.stamp + "\n"

    def test_a_task_image_user_comes_back_after_the_step(self):
        key = proxy_layer_key(RECIPE, BASE)
        text = proxy_layer_dockerfile(RECIPE, "b", key, user="app:app")
        lines = text.splitlines()
        assert lines[1] == "USER root" and lines[-1] == "USER app:app"


# -- build, reuse and prune against the stand-in docker -----------------


class ProxyFakeDocker(FakeDocker):
    """FakeDocker that also builds proxy layers: ``build --file - <context>``."""

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[dict[str, str]] = []

    async def run(self, *args, stdin=None, timeout=0):
        argv = list(args)
        if argv[:1] != ["build"] or "--file" not in argv:
            return await super().run(*args, stdin=stdin, timeout=timeout)
        self.calls.append(argv)
        context = Path(argv[-1])
        self.contexts.append(
            {p.name: p.read_text() for p in context.iterdir() if p.is_file()}
        )
        assert stdin is not None
        text = stdin.decode()
        self.dockerfiles.append(text)
        base = self._by_ref(text.splitlines()[0].removeprefix("FROM "))
        assert base is not None
        tag = argv[argv.index("--tag") + 1]
        labels = dict(base.info["Config"]["Labels"])
        for i, a in enumerate(argv):
            if a == "--label":
                name, _, value = argv[i + 1].partition("=")
                labels[name] = value
        run_line = next(line for line in text.splitlines() if line.startswith("RUN "))
        command = json.loads(run_line[run_line.index("[") :])
        built = self.add_image(
            tag,
            layers=[
                *base.info["RootFS"]["Layers"],
                "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            ],
            labels=labels,
            user=base.info["Config"]["User"],
            size=self.build_size,
        )
        self.records.append(
            {
                "id": f"rec{len(self.records)}",
                "description": "mount / from exec " + " ".join(command),
            }
        )
        self.write_iidfile(argv, built.id)
        return self._ok(built.id)


@pytest.fixture(autouse=True)
def layer_trust(monkeypatch, tmp_path_factory) -> Path:
    """This test's own record of the layers it built (not ~/.cache)."""
    root = tmp_path_factory.mktemp("layer-trust")
    monkeypatch.setattr(env_layers, "_trust_root", lambda: root, raising=False)
    return root


@pytest.fixture
def docker(monkeypatch) -> ProxyFakeDocker:
    fake = ProxyFakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    return fake


class TestBuildAndReuse:
    async def test_a_miss_builds_from_the_task_image_then_a_hit_reuses_it(self, docker):
        task = docker.add_image("bf__task")
        layers = DockerLayers()
        first = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert (first.kind, first.status) == (PROXY, "miss")
        assert first.build_sec is not None
        assert first.tag == layer_tag(PROXY, first.key)
        built = docker.tags[first.tag]
        assert built.info["RootFS"]["Layers"][:1] == task.info["RootFS"]["Layers"]
        labels = built.info["Config"]["Labels"]
        assert labels[LAYER_LABEL] == PROXY
        assert labels[LAYER_KEY_LABEL] == first.key
        assert labels[LAYER_PROXY_GROUP_LABEL] == "litellm-proxy"
        # The build's context is the requirement file and nothing else.
        assert docker.contexts == [{"litellm-proxy.txt": RECIPE.requirements_text}]

        second = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert (second.status, second.key) == ("hit", first.key)
        assert sum(1 for c in docker.calls if c[0] == "build") == 1
        record = second.record()
        assert record["status"] == "hit" and record["key"] == first.key
        # The layer as a base for the agent's layer on top of it.
        assert second.as_base is not None
        assert second.as_base.rootfs == tuple(built.info["RootFS"]["Layers"])
        assert second.as_base.libc == "glibc"

    async def test_a_planted_layer_with_the_right_labels_is_never_reused(self, docker):
        """A task's Compose file builds FROM the
        victim's public task image, plants a changed proxy and the stamp, sets
        the labels (every input of the key is public) and tags it
        benchflow-layer:proxy-<key>. Labels and the RootFS prefix matched, so
        the victim ran the planted proxy with its provider key."""
        task = docker.add_image("bf__task")
        layers = DockerLayers()
        base = await layers.inspect_base("bf__task")
        key = proxy_layer_key(RECIPE, base)
        planted = docker.add_image(
            layer_tag(PROXY, key),
            layers=[*task.info["RootFS"]["Layers"], "sha256:" + "e" * 64],
            labels={
                LAYER_LABEL: PROXY,
                LAYER_KEY_LABEL: key,
                LAYER_BASE_LABEL: base.digest,
                LAYER_PROXY_GROUP_LABEL: RECIPE.group,
            },
        )

        first = await layers.ensure_proxy_layer(
            RECIPE, "bf__task", alias="bf__layer__victim", prune=False
        )

        assert (first.status, first.key) == ("miss", key)
        assert sum(1 for c in docker.calls if c[0] == "build") == 1
        assert docker.tags["bf__layer__victim"].id != planted.id
        assert docker.tags[first.tag].id != planted.id
        assert first.as_base is not None and first.as_base.image_id != planted.id
        # The trial's alias names the built image by its id, not the tag.
        (tag_call,) = [
            c for c in docker.calls if c[:1] == ["tag"] and c[-1] == "bf__layer__victim"
        ]
        assert tag_call == ["tag", docker.tags[first.tag].id, "bf__layer__victim"]

        second = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert second.status == "hit"
        assert sum(1 for c in docker.calls if c[0] == "build") == 1

    async def test_two_task_images_with_the_same_requirements_never_share_a_layer(
        self, docker
    ):
        """A layer built FROM one task image is never the final
        image of a trial on another, however equal their requirements."""
        docker.add_image("bf__task-a", layers=["sha256:" + "a" * 64])
        docker.add_image("bf__task-b", layers=["sha256:" + "b" * 64])
        layers = DockerLayers()
        a = await layers.ensure_proxy_layer(RECIPE, "bf__task-a", prune=False)
        b = await layers.ensure_proxy_layer(RECIPE, "bf__task-b", prune=False)
        assert a.key != b.key and a.tag != b.tag
        assert (a.status, b.status) == ("miss", "miss")
        image_a, image_b = docker.tags[a.tag], docker.tags[b.tag]
        assert image_a.id != image_b.id
        assert image_a.info["RootFS"]["Layers"][0] == "sha256:" + "a" * 64
        assert image_b.info["RootFS"]["Layers"][0] == "sha256:" + "b" * 64
        assert image_a.info["Config"]["Labels"][LAYER_BASE_LABEL] == a.base.digest
        assert image_b.info["Config"]["Labels"][LAYER_BASE_LABEL] == b.base.digest

    async def test_a_proxy_layer_build_mounts_no_cache_another_build_can_reach(
        self, docker
    ):
        """Guards U2b-R1-B-1: the build mounted a BuildKit cache whose id
        (benchflow-uv-<task image digest>) anyone can compute, and a cache id
        is shared by every build on the daemon, so an untrusted task
        Dockerfile could plant a changed litellm that a later proxy layer
        installed. Now the only mount is the requirement file, and uv keeps
        no cache past the step."""
        docker.add_image("bf__task-a", layers=["sha256:" + "a" * 64])
        docker.add_image("bf__task-b", layers=["sha256:" + "b" * 64])
        layers = DockerLayers()
        await layers.ensure_proxy_layer(RECIPE, "bf__task-a", prune=False)
        await layers.ensure_proxy_layer(RECIPE, "bf__task-b", prune=False)
        assert len(docker.dockerfiles) == 2
        for text in docker.dockerfiles:
            (run,) = [line for line in text.splitlines() if line.startswith("RUN ")]
            mounts = re.findall(r"--mount=(\S+)", run)
            assert mounts == [
                f"type=bind,source={RECIPE.requirements_name},"
                f"target={PROXY_LAYER_REQUIREMENTS_DIR}/{RECIPE.requirements_name}"
            ]
            assert "export UV_NO_CACHE=1" in json.loads(run[run.index("[") :])[2]

    async def test_an_image_under_the_cached_tag_from_another_base_is_rebuilt(
        self, docker
    ):
        """A hit needs the layer's labels and its base's layers under it."""
        docker.add_image("bf__task", layers=["sha256:" + "a" * 64])
        layers = DockerLayers()
        first = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        # Someone re-tags an unrelated image as the cached tag.
        docker.add_image(first.tag, layers=["sha256:" + "f" * 64])
        again = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert again.status == "miss"

    async def test_a_failed_build_raises_a_layer_error_with_its_tail(
        self, docker, monkeypatch
    ):
        docker.add_image("bf__task")

        async def failing(_self, *args, stdin=None, timeout=0):
            if args[:1] == ("build",):
                return env_layers._Run(stdout="", stderr="step 1\nNo space", code=1)
            return await ProxyFakeDocker.run(
                docker, *args, stdin=stdin, timeout=timeout
            )

        monkeypatch.setattr(DockerLayers, "_docker", failing)
        with pytest.raises(LayerError, match="litellm-proxy proxy layer failed"):
            await DockerLayers().ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        # The pinned base tag is gone, and no context folder is left.
        assert not [t for t in docker.tags if t.startswith("benchflow-layer-base")]

    async def test_the_build_context_is_removed(self, docker, monkeypatch, tmp_path):
        monkeypatch.setattr(env_layers.tempfile, "tempdir", str(tmp_path))
        docker.add_image("bf__task")
        await DockerLayers().ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert os.listdir(tmp_path) == []

    async def test_a_build_context_that_cannot_be_written_is_a_layer_error(
        self, docker, monkeypatch
    ):
        """Guards U2b-R1-A-2: an OSError writing the build context (a full or
        read-only temp folder on this machine) escaped as itself, so the
        sandbox's start failed instead of recording the layer as failed and
        installing the proxy in the trial."""

        def full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(env_layers.tempfile, "mkdtemp", full)
        docker.add_image("bf__task")
        with pytest.raises(LayerError, match="proxy layer's build context"):
            await DockerLayers().ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        assert not [t for t in docker.tags if t.startswith("benchflow-layer-base")]


class TestPrune:
    """Pruning removes only BenchFlow-labelled proxy layers and their records."""

    async def test_every_prune_command_names_benchflows_label(self, docker):
        layers = DockerLayers()
        keys = []
        for n in range(4):
            docker.add_image(f"bf__task-{n}", layers=[f"sha256:{n:064x}"])
            result = await layers.ensure_proxy_layer(
                RECIPE, f"bf__task-{n}", prune=False
            )
            DockerLayers.release(result.key, daemon=result.daemon_id)
            keys.append(result.key)
        # Someone else's image and build record, which the prune must not touch.
        docker.add_image("someone/else:1", labels={"x": "y"})
        docker.records.append({"id": "theirs", "description": "mount / from exec make"})
        docker.calls.clear()

        report = await layers.prune(PROXY, keep=1)

        assert len(report.removed) == 3
        assert "someone/else:1" in docker.tags
        assert any(r["id"] == "theirs" for r in docker.records)
        listing = [c for c in docker.calls if c[:2] == ["image", "ls"]]
        assert listing and all(f"label={LAYER_LABEL}={PROXY}" in c for c in listing), (
            listing
        )
        removing = docker.prune_commands()
        assert removing
        for command in removing:
            text = " ".join(command)
            if command[:2] == ["image", "rm"]:
                assert re.fullmatch(
                    r"image rm --no-prune benchflow-layer:proxy-[0-9a-f]{64}", text
                ), text
            else:
                assert command[:2] == ["builder", "prune"], text
                assert "description~=" in text
                assert "benchflow[.]layer=proxy key=" in text, text

    async def test_the_agent_and_proxy_bounds_are_counted_apart(self, docker):
        """A proxy prune never removes an agent layer, and the other way."""
        from tests.test_env_layers_agent import RECIPE as AGENT_RECIPE

        layers = DockerLayers()
        docker.add_image("bf__task")
        proxy = await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
        agent = await layers.ensure_agent_layer(
            AGENT_RECIPE, proxy.tag, prune=False, base=proxy.as_base
        )
        DockerLayers.release(proxy.key, daemon=proxy.daemon_id)
        DockerLayers.release(agent.key, daemon=agent.daemon_id)
        await layers.prune(PROXY, keep=0)
        assert agent.tag in docker.tags
        assert proxy.tag not in docker.tags

    async def test_a_build_prunes_proxy_layers_past_the_keep(self, docker, monkeypatch):
        monkeypatch.setenv(env_layers.PROXY_LAYER_KEEP_ENV, "2")
        layers = DockerLayers()
        made = []
        for n in range(3):
            docker.add_image(f"bf__task-{n}", layers=[f"sha256:{n:064x}"])
            result = await layers.ensure_proxy_layer(RECIPE, f"bf__task-{n}")
            DockerLayers.release(result.key, daemon=result.daemon_id)
            made.append(result.tag)
        kept = [tag for tag in made if tag in docker.tags]
        assert len(kept) == 2 and made[0] not in kept


class TestUnsupportedRecords:
    @pytest.mark.parametrize("backend", ["modal", "apple-container", "agentcore"])
    def test_backends_without_layers(self, backend):
        records = env_layers.unsupported_records(backend, [PROXY, "agent"])
        assert set(records) == {PROXY, "agent"}
        for record in records.values():
            assert record["status"] == "unsupported"
            assert backend in record["reason"]

    @pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
    def test_backends_that_record_their_own(self, backend):
        assert env_layers.unsupported_records(backend, [PROXY]) == {}
