"""The agent layer's key, build, reuse and prune (env_layers, PY8 U3a).

Against a stand-in ``docker`` (:class:`FakeDocker`) that keeps images and
BuildKit records in memory and logs every command; the same paths against a
real daemon are in ``tests/test_env_layers_docker.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, ClassVar

import pytest

from benchflow.sandbox import _base, _docker_sweep, _layer_lock, env_layers
from benchflow.sandbox.env_layers import (
    AGENT,
    LAYER_FORMAT,
    LAYER_KEY_LABEL,
    LAYER_LABEL,
    LAYER_REPOSITORY,
    AgentLayerRecipe,
    BaseImage,
    DockerLayers,
    LayerError,
    LayerSkipped,
    agent_layer_dockerfile,
    agent_layer_key,
    alias_owner_state,
    alias_owner_tag,
    image_content_digest,
    layer_tag,
    marker,
)

RECIPE = AgentLayerRecipe(
    agent="fake-agent",
    version="1.2.3",
    install_cmd="mkdir -p /opt/benchflow/bin && echo fake > /opt/benchflow/bin/fake-agent",
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
    """An agent version, recipe, task image, libc or architecture change
    makes a new key."""

    @pytest.mark.parametrize(
        "change",
        [
            {"agent": "other-agent"},
            {"version": "1.2.4"},
            {"install_cmd": RECIPE.install_cmd + " && true"},
            {"uv_version": "0.13.4"},
        ],
        ids=["agent", "version", "recipe", "uv"],
    )
    def test_recipe_inputs_change_the_key(self, change):
        other = dataclasses.replace(RECIPE, **change)
        assert agent_layer_key(other, BASE) != agent_layer_key(RECIPE, BASE)

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
        assert agent_layer_key(RECIPE, other) != agent_layer_key(RECIPE, BASE)

    def test_the_layer_format_changes_the_key(self, monkeypatch):
        before = agent_layer_key(RECIPE, BASE)
        monkeypatch.setattr(env_layers, "LAYER_FORMAT", LAYER_FORMAT + 1)
        assert agent_layer_key(RECIPE, BASE) != before

    def test_the_root_install_environment_changes_the_key(self, monkeypatch):
        """Guards M1 in 13b7aaa58 when root's interpreter search path changes."""
        before = agent_layer_key(RECIPE, BASE)
        monkeypatch.setattr(env_layers, "ROOT_EXEC_PATH", "/usr/bin:/bin")
        assert agent_layer_key(RECIPE, BASE) != before

    def test_the_root_environment_policy_changes_the_key(self, monkeypatch):
        """Guards M1 in 13b7aaa58 when the shared environment wrapper changes."""
        before = agent_layer_key(RECIPE, BASE)
        wrap = env_layers.with_system_path
        monkeypatch.setattr(
            env_layers,
            "with_system_path",
            lambda command, path: "unset NEW_PYTHON_INPUT; " + wrap(command, path),
        )
        assert agent_layer_key(RECIPE, BASE) != before

    @pytest.mark.parametrize("policy", ["flags", "unset"])
    def test_the_pre_exec_shell_policy_changes_the_key(self, monkeypatch, policy):
        """Guards residual M1 in 01727c09e: key the complete shell policy."""
        before = agent_layer_key(RECIPE, BASE)
        if policy == "flags":
            flags = _base.system_shell_env_flags()
            monkeypatch.setattr(
                _base, "system_shell_env_flags", lambda: [*flags, "-e", "NEW_LOADER="]
            )
        else:
            monkeypatch.setattr(
                _base,
                "SYSTEM_SHELL_UNSET_VARS",
                (*_base.SYSTEM_SHELL_UNSET_VARS, "NEW_STARTUP"),
            )
        assert agent_layer_key(RECIPE, BASE) != before

    def test_what_does_not_change_the_bytes_keeps_the_key(self):
        """The image id, its tag and the time limit are not part of the key:
        a rebuild of the same task image reuses the layer."""
        same_base = dataclasses.replace(
            BASE, ref="other-tag", image_id="sha256:" + "9" * 64
        )
        same_recipe = dataclasses.replace(RECIPE, timeout_sec=5)
        assert agent_layer_key(same_recipe, same_base) == agent_layer_key(RECIPE, BASE)

    @pytest.mark.parametrize("version", ["", "latest", "*", " "])
    def test_an_unpinned_agent_has_no_layer(self, version):
        with pytest.raises(ValueError, match="pinned version"):
            AgentLayerRecipe(agent="a", version=version, install_cmd="true")


class TestContentDigest:
    INFO: ClassVar[dict[str, Any]] = {
        "Id": "sha256:" + "a" * 64,
        "Created": "2026-10-16T02:36:59Z",
        "Architecture": "amd64",
        "Os": "linux",
        "RootFS": {"Type": "layers", "Layers": ["sha256:" + "b" * 64]},
        "Config": {
            "Env": ["PATH=/usr/bin"],
            "Cmd": ["sh"],
            "WorkingDir": "/app",
            "User": "",
            "Hostname": "abc",
            "Labels": {"com.docker.compose.project": "trial-1", "x": "1"},
        },
    }

    def test_a_cached_rebuild_has_the_same_digest(self):
        """Teardown removes the built task image and the next trial rebuilds
        it from cache: new id, new Created, another compose project label,
        same bytes. Keyed by id, every trial would miss."""
        rebuilt = json.loads(json.dumps(self.INFO))
        rebuilt["Id"] = "sha256:" + "c" * 64
        rebuilt["Created"] = "2026-10-16T03:00:00Z"
        rebuilt["Config"]["Hostname"] = "def"
        rebuilt["Config"]["Labels"]["com.docker.compose.project"] = "trial-2"
        assert image_content_digest(rebuilt) == image_content_digest(self.INFO)

    @pytest.mark.parametrize(
        "path,value",
        [
            (("RootFS", "Layers"), ["sha256:" + "d" * 64]),
            (("Config", "Env"), ["PATH=/usr/bin", "X=1"]),
            (("Config", "User"), "app"),
            (("Config", "WorkingDir"), "/srv"),
            (("Config", "Labels"), {"x": "2"}),
            (("Architecture",), "arm64"),
        ],
    )
    def test_what_a_container_sees_changes_the_digest(self, path, value):
        other = json.loads(json.dumps(self.INFO))
        target = other
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        assert image_content_digest(other) != image_content_digest(self.INFO)


# -- the Dockerfile ------------------------------------------------------


class TestDockerfile:
    KEY = "f" * 64

    def test_one_run_step_in_exec_form_starting_with_the_marker(self):
        text = agent_layer_dockerfile(RECIPE, "benchflow-layer-base:x", self.KEY)
        lines = text.splitlines()
        assert lines[0] == "FROM benchflow-layer-base:x"
        assert len(lines) == 2 and lines[1].startswith("RUN [")
        argv = json.loads(lines[1].removeprefix("RUN "))
        assert argv[:2] == ["/bin/sh", "-c"]
        assert argv[2].startswith(f"{marker(AGENT, self.KEY)} /bin/sh -c ")
        words = shlex.split(argv[2].removeprefix(f"{marker(AGENT, self.KEY)} "))
        assert words[:3] == [
            "/bin/sh",
            "-c",
            env_layers.with_system_path(RECIPE.install_cmd, env_layers.ROOT_EXEC_PATH),
        ]
        assert words[3:6] == ["&&", "/bin/sh", "-c"]

    def test_nothing_from_outside_the_recipe(self):
        """No ARG, ENV, COPY, ADD or mount: the build gets nothing per trial."""
        text = agent_layer_dockerfile(RECIPE, "base", self.KEY, user="app")
        # The shared route unsets BASH_ENV/ENV inside RUN; those are not
        # Dockerfile ENV instructions and do not persist in the image.
        instructions = {line.split()[0] for line in text.splitlines()}
        assert not instructions.intersection({"ARG", "ENV", "COPY", "ADD"})
        for word in ("--mount", "--secret"):
            assert word not in text

    def test_a_task_user_is_restored_after_the_root_step(self):
        lines = agent_layer_dockerfile(
            RECIPE, "base", self.KEY, user="app:app"
        ).splitlines()
        assert lines[1] == "USER root"
        assert lines[-1] == "USER app:app"

    @pytest.mark.parametrize("user", ["app\nRUN id", "a b", "$(id)"])
    def test_a_user_that_would_add_instructions_is_refused(self, user):
        with pytest.raises(LayerError):
            agent_layer_dockerfile(RECIPE, "base", self.KEY, user=user)

    def test_the_marker_ends_the_key(self):
        assert marker(AGENT, self.KEY) == f": benchflow.layer=agent key={self.KEY};"
        with pytest.raises(ValueError):
            marker(AGENT, "abc")


# -- a stand-in docker ---------------------------------------------------


@dataclasses.dataclass
class _Image:
    id: str
    info: dict[str, Any]
    created: str


class FakeDocker:
    """A ``docker`` that keeps images and BuildKit records in memory.

    Patched over :meth:`DockerLayers._docker`; every command it gets is in
    :attr:`calls`, every Dockerfile in :attr:`dockerfiles`.
    """

    def __init__(self, libc: str = "glibc") -> None:
        self.calls: list[list[str]] = []
        self.dockerfiles: list[str] = []
        self.tags: dict[str, _Image] = {}
        self.records: list[dict[str, str]] = []
        self.used_ids: set[str] = set()
        self.libc = libc
        # What `docker image inspect` says a built layer weighs.
        self.build_size = 1000
        # `docker info`'s root folder and daemon name; None: the call fails.
        self.info: str | None = None
        self.daemon_id = "fake-daemon-1"
        self._n = 0

    def add_image(
        self,
        tag: str,
        *,
        layers: list[str] | None = None,
        labels: dict | None = None,
        user: str = "",
        created: str | None = None,
        size: int = 1000,
    ) -> _Image:
        self._n += 1
        image_id = "sha256:" + hashlib.sha256(f"{tag}{self._n}".encode()).hexdigest()
        info = {
            "Id": image_id,
            "Architecture": "amd64",
            "Os": "linux",
            "Size": size,
            "RootFS": {"Type": "layers", "Layers": layers or ["sha256:" + "0" * 64]},
            "Config": {
                "User": user,
                "Labels": dict(labels or {}),
                "Env": ["PATH=/bin"],
            },
        }
        image = _Image(image_id, info, created or self._stamp())
        self.tags[tag] = image
        return image

    def _stamp(self) -> str:
        n = self._n
        return f"2026-10-16 {n // 3600:02d}:{n // 60 % 60:02d}:{n % 60:02d} +0000 UTC"

    def _ok(self, stdout: str = "") -> env_layers._Run:
        return env_layers._Run(stdout=stdout, stderr="", code=0)

    def _fail(self, text: str) -> env_layers._Run:
        return env_layers._Run(stdout="", stderr=text, code=1)

    def _by_ref(self, ref: str) -> _Image | None:
        if ref in self.tags:
            return self.tags[ref]
        for image in self.tags.values():
            if image.id == ref:
                return image
        return None

    def _repo_tags(self, image: _Image) -> list[str]:
        return [
            t if ":" in t.rpartition("/")[2] else f"{t}:latest"
            for t, i in self.tags.items()
            if i.id == image.id
        ]

    @staticmethod
    def _labels_match(image: _Image, filters: list[str]) -> bool:
        labels = image.info["Config"]["Labels"]
        for f in filters:
            name, _, value = f.removeprefix("label=").partition("=")
            if labels.get(name) != value:
                return False
        return True

    async def run(
        self, *args: str, stdin: bytes | None = None, timeout: float = 0
    ) -> env_layers._Run:
        argv = list(args)
        self.calls.append(argv)
        filters = [argv[i + 1] for i, a in enumerate(argv) if a == "--filter"]
        match argv:
            case ["image", "inspect", "--format", fmt, ref]:
                image = self._by_ref(ref)
                label = re.search(r'"([^"]+)"', fmt)
                assert image is not None and label is not None
                return self._ok(
                    image.info["Config"]["Labels"].get(label.group(1), "") + "\n"
                )
            case ["image", "inspect", *refs] if refs and refs[0] != "--format":
                # Like Docker: what exists is printed, a missing ref fails.
                found = [self._by_ref(ref) for ref in refs]
                infos = [dict(i.info, RepoTags=self._repo_tags(i)) for i in found if i]
                return env_layers._Run(
                    stdout=json.dumps(infos),
                    stderr="" if all(found) else "No such image",
                    code=0 if all(found) else 1,
                )
            case ["image", "ls", "--quiet", "--no-trunc", *_]:
                ids = {
                    i.id for i in self.tags.values() if self._labels_match(i, filters)
                }
                return self._ok("".join(f"{i}\n" for i in sorted(ids)))
            case ["image", "ls", *_]:
                rows = []
                for tag, image in self.tags.items():
                    if self._labels_match(image, filters):
                        repo, _, name = tag.partition(":")
                        rows.append(
                            json.dumps(
                                {
                                    "Repository": repo,
                                    "Tag": name or "latest",
                                    "ID": image.id[7:19],
                                    "CreatedAt": image.created,
                                }
                            )
                        )
                return self._ok("".join(f"{r}\n" for r in rows))
            case ["ps", "--all", "--quiet", "--filter", spec]:
                image = self._by_ref(spec.removeprefix("ancestor="))
                used = image is not None and image.id in self.used_ids
                return self._ok("c0ffee\n" if used else "")
            case ["run", *_]:
                return self._ok(self.libc + "\n")
            case ["rm", "--force", name] if name.startswith("benchflow-libc-probe-"):
                # The libc probe's container, already gone by its --rm.
                return self._fail(f"No such container: {name}")
            case ["volume", "rm", *_]:
                # The libc probe's named volumes, one per image VOLUME.
                return self._ok()
            case ["pull", "--quiet", ref]:
                self.add_image(ref)
                return self._ok(ref)
            case ["tag", src, dst]:
                image = self._by_ref(src)
                if image is None:
                    return self._fail("No such image")
                self.tags[dst] = image
                return self._ok()
            case ["image", "rm", *options, ref] if not options or options == [
                "--no-prune"
            ]:
                # Without --force: a tag goes; the last tag of an image a
                # container uses stays.
                image = self.tags.get(ref)
                if image is None:
                    return self._fail("No such image")
                if image.id in self.used_ids and len(self._repo_tags(image)) == 1:
                    return self._fail("conflict: image is being used")
                del self.tags[ref]
                return self._ok()
            case ["info", "--format", "{{.ID}}"]:
                # The daemon's id names its layer locks (_layer_lock).
                return self._ok(self.daemon_id + "\n")
            case ["info", "--format", _]:
                return self._ok(self.info) if self.info else self._fail("no daemon")
            case ["build", *_]:
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
                run_line = next(
                    line for line in text.splitlines() if line.startswith("RUN ")
                )
                command = json.loads(run_line[4:])
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
            case ["image", "prune", "--all", "--force", *_]:
                # Docker's: every tag of every unused image whose labels
                # match, inherited labels included.
                for tag, image in list(self.tags.items()):
                    if image.id not in self.used_ids and self._labels_match(
                        image, filters
                    ):
                        del self.tags[tag]
                return self._ok()
            case ["builder", "du", "--verbose"]:
                return self._ok(
                    "".join(
                        f"ID:\t{r['id']}\nDescription:\t{r['description']}\n\n"
                        for r in self.records
                    )
                )
            case ["builder", "prune", "--all", "--force", "--filter", spec]:
                quoted = re.fullmatch(r'description~="(.*)"', spec)
                assert quoted is not None, spec
                self.records = [
                    r
                    for r in self.records
                    if not re.search(quoted.group(1), r["description"])
                ]
                return self._ok()
        raise AssertionError(f"stand-in docker got an unexpected command: {argv}")

    @staticmethod
    def write_iidfile(argv: list[str], image_id: str) -> None:
        """Like Docker: ``--iidfile <path>`` gets the built image's id."""
        if "--iidfile" in argv:
            Path(argv[argv.index("--iidfile") + 1]).write_text(image_id)

    def prune_commands(self) -> list[list[str]]:
        """Commands that remove something (call after clearing :attr:`calls`)."""
        return [c for c in self.calls if "prune" in c or c[:2] == ["image", "rm"]]


@pytest.fixture(autouse=True)
def layer_trust(monkeypatch, tmp_path_factory) -> Path:
    """This test's own record of the layers it built (not ~/.cache)."""
    root = tmp_path_factory.mktemp("layer-trust")
    monkeypatch.setattr(env_layers, "_trust_root", lambda: root, raising=False)
    return root


@pytest.fixture
def docker(monkeypatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(DockerLayers, "_docker", fake.run)
    return fake


class TestBuildAndReuse:
    @pytest.mark.parametrize(
        "name",
        sorted(
            set(_base.SYSTEM_SHELL_UNSET_VARS)
            | {flag[:-1] for flag in _base.system_shell_env_flags() if flag != "-e"}
            | {"LD_TRACE_LOADED_OBJECTS", "LD_FUTURE_SWITCH"}
        ),
    )
    @pytest.mark.parametrize("value", ["", "private-image-value"])
    @pytest.mark.parametrize("on_proxy", [False, True], ids=["task", "proxy"])
    async def test_image_shell_inputs_skip_before_cache_lookup(
        self, docker, monkeypatch, name, value, on_proxy
    ):
        """Guards residual M1 in 01727c09e, including empty presence switches."""
        task = docker.add_image("bf__task")
        task.info["Config"]["Env"].append(f"{name}={value}")
        layers = DockerLayers()
        base = await layers.inspect_base("bf__task") if on_proxy else None

        async def no_lookup(*args, **kwargs):
            pytest.fail("an ineligible image must not read or build the cache")

        monkeypatch.setattr(layers, "_ensure", no_lookup)
        monkeypatch.setattr(layers, "_libc", no_lookup)
        with pytest.raises(LayerSkipped) as skipped:
            await layers.ensure_agent_layer(RECIPE, "bf__task", base=base, prune=False)
        assert str(skipped.value) == f"image env sets {name}"
        assert not docker.dockerfiles
        assert not DockerLayers._claims

    @pytest.mark.parametrize(
        "config",
        [
            {"Entrypoint": ["sh", "-c", "echo changed > /opt/benchflow/bin/gemini"]},
            {
                "Healthcheck": {
                    "Test": ["CMD-SHELL", "echo changed > /opt/benchflow/bin/gemini"]
                }
            },
            {"Volumes": {"/opt/benchflow/bin": {}}},
        ],
        ids=["entrypoint", "healthcheck", "volume"],
    )
    @pytest.mark.parametrize("on_proxy", [False, True], ids=["task", "proxy"])
    async def test_image_runtime_overrides_install_in_trial(
        self, docker, config, on_proxy
    ):
        """Guards U3c-CONFIRM-1 at 1c019ec9, including a proxy beneath the agent."""
        task = docker.add_image("bf__task")
        task.info["Config"].update(config)
        layers = DockerLayers()
        # The proxy path supplies its already inspected image as the base.
        base = await layers.inspect_base("bf__task") if on_proxy else None
        with pytest.raises(LayerError, match="runtime configuration"):
            await layers.ensure_agent_layer(RECIPE, "bf__task", base=base, prune=False)
        assert docker.dockerfiles == []
        assert not DockerLayers._claims

    async def test_a_miss_builds_from_the_task_image_then_a_hit_reuses_it(self, docker):
        task = docker.add_image("bf__task")
        layers = DockerLayers()
        first = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert first.status == "miss" and first.build_sec is not None
        assert first.tag == layer_tag(AGENT, first.key)
        built = docker.tags[first.tag]
        assert built.info["RootFS"]["Layers"][:1] == task.info["RootFS"]["Layers"]
        assert len(built.info["RootFS"]["Layers"]) == 2
        # FROM a tag of BenchFlow's own on the task image's id, removed after.
        base_line = docker.dockerfiles[0].splitlines()[0]
        assert base_line.startswith("FROM benchflow-layer-base/agent-")
        assert base_line.removeprefix("FROM ") not in docker.tags

        builds = sum(1 for c in docker.calls if c[0] == "build")
        second = await layers.ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert (second.status, second.key) == ("hit", first.key)
        assert second.build_sec is None
        assert sum(1 for c in docker.calls if c[0] == "build") == builds
        assert second.record()["status"] == "hit"

    async def test_the_build_gets_no_context_arguments_or_secrets(self, docker):
        docker.add_image("bf__task")
        await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        build = next(c for c in docker.calls if c[0] == "build")
        assert build[-1] == "-"  # Dockerfile on stdin, no context folder
        for flag in (
            "--build-arg",
            "--secret",
            "--ssh",
            "--build-context",
            "-f",
            "--file",
        ):
            assert flag not in build
        labels = dict(
            build[i + 1].split("=", 1) for i, a in enumerate(build) if a == "--label"
        )
        assert labels[LAYER_LABEL] == AGENT
        assert labels["com.docker.compose.project"] == ""

    async def test_a_layer_built_on_another_task_image_is_never_a_hit(self, docker):
        """Two task images, the same recipe: two keys, two layers."""
        docker.add_image("bf__one", layers=["sha256:" + "a" * 64])
        docker.add_image("bf__two", layers=["sha256:" + "b" * 64])
        layers = DockerLayers()
        one = await layers.ensure_agent_layer(RECIPE, "bf__one", prune=False)
        two = await layers.ensure_agent_layer(RECIPE, "bf__two", prune=False)
        assert one.key != two.key
        assert two.status == "miss"

    async def test_a_tag_holding_something_else_is_rebuilt(self, docker):
        """An image under the layer's tag whose labels or layers do not match
        the task image is not reused."""
        docker.add_image("bf__task")
        key = (
            await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        ).key
        docker.add_image(
            layer_tag(AGENT, key), labels={LAYER_LABEL: AGENT, LAYER_KEY_LABEL: key}
        )
        again = await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert again.status == "miss"

    async def test_a_musl_task_image_gets_its_own_key(self, docker):
        docker.add_image("bf__task")
        glibc = await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        DockerLayers._libc_cache.clear()
        docker.libc = "musl"
        # The label an earlier layer left says glibc for this digest; a
        # different task image (other bytes) is probed afresh.
        docker.add_image("bf__alpine", layers=["sha256:" + "c" * 64])
        musl = await DockerLayers().ensure_agent_layer(
            RECIPE, "bf__alpine", prune=False
        )
        assert musl.base.libc == "musl" and glibc.base.libc == "glibc"
        assert musl.key != glibc.key

    async def test_libc_is_read_from_an_earlier_layer_without_a_container(self, docker):
        docker.add_image("bf__task")
        await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        DockerLayers._libc_cache.clear()
        docker.calls.clear()
        await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert not any(c[0] == "run" for c in docker.calls)

    @staticmethod
    def _base_listings(docker) -> list[list[str]]:
        return [
            c
            for c in docker.calls
            if c[:2] == ["image", "ls"]
            and any(a.startswith("label=benchflow.layer.base=") for a in c)
        ]

    async def test_libc_is_read_from_the_recipes_own_layer_without_a_listing(
        self, docker
    ):
        """SPEED-FIX: the daemon's filtered image list took 0.5 s of every
        trial's start on a daemon with 207 images; the recipe's own layer is
        read by its tag instead, from the same labels."""
        docker.add_image("bf__task")
        first = await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        DockerLayers._libc_cache.clear()
        docker.calls.clear()
        again = await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert (again.status, again.key, again.base.libc) == ("hit", first.key, "glibc")
        assert self._base_listings(docker) == []
        assert not any(c[0] == "run" for c in docker.calls)

    async def test_another_recipe_reads_libc_from_the_listing_as_before(self, docker):
        docker.add_image("bf__task")
        await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        DockerLayers._libc_cache.clear()
        docker.calls.clear()
        other = dataclasses.replace(RECIPE, version="9.9.9")
        result = await DockerLayers().ensure_agent_layer(other, "bf__task", prune=False)
        assert result.base.libc == "glibc" and result.status == "miss"
        assert len(self._base_listings(docker)) == 1
        assert not any(c[0] == "run" for c in docker.calls)

    async def test_a_layer_tag_naming_another_image_or_libc_is_not_believed(
        self, docker
    ):
        """A candidate tag whose labels name another task image or another
        libc than its key's says nothing; the listing or the probe decides."""
        task = docker.add_image("bf__task")
        base = await DockerLayers().inspect_base("bf__task")
        DockerLayers._libc_cache.clear()
        musl = dataclasses.replace(base, libc="musl")
        docker.add_image(
            layer_tag(AGENT, agent_layer_key(RECIPE, musl)),
            labels={
                env_layers.LAYER_BASE_LABEL: base.digest,
                env_layers.LAYER_LIBC_LABEL: "glibc",
            },
        )
        docker.calls.clear()
        result = await DockerLayers().inspect_base(
            "bf__task", for_agent=True, agent_recipe=RECIPE
        )
        assert result.libc == "glibc" and result.image_id == task.id
        assert len(self._base_listings(docker)) == 1

    async def test_a_prebuilt_task_image_is_pulled_when_absent(self, docker):
        result = await DockerLayers().ensure_agent_layer(
            RECIPE, "ghcr.io/x/task:1", pull=True, prune=False
        )
        assert ["pull", "--quiet", "ghcr.io/x/task:1"] in docker.calls
        assert result.status == "miss"

    async def test_a_failed_build_raises_with_the_log_tail(self, docker, monkeypatch):
        docker.add_image("bf__task")
        real = docker.run

        async def failing(_layers, *args, **kwargs):
            if args[0] == "build":
                docker.calls.append(list(args))
                return env_layers._Run(stdout="", stderr="step 1\nnpm ERR! 404", code=1)
            return await real(*args, **kwargs)

        monkeypatch.setattr(DockerLayers, "_docker", failing)
        with pytest.raises(LayerError, match="npm ERR! 404"):
            await DockerLayers().ensure_agent_layer(RECIPE, "bf__task", prune=False)
        assert not [t for t in docker.tags if t.startswith("benchflow-layer-base")]

    async def test_an_alias_is_the_trials_own_tag_and_the_key_is_claimed(self, docker):
        docker.add_image("bf__task")
        result = await DockerLayers().ensure_agent_layer(
            RECIPE, "bf__task", alias="bf__layer__trial-1", prune=False
        )
        assert result.image == "bf__layer__trial-1"
        assert docker.tags["bf__layer__trial-1"] is docker.tags[result.tag]
        assert DockerLayers._claims[(result.daemon_id, result.key)] == 1
        DockerLayers.release(result.key, daemon=result.daemon_id)
        assert (result.daemon_id, result.key) not in DockerLayers._claims


# -- prune ---------------------------------------------------------------


def _named_label(command: list[str], key: str) -> bool:
    """Whether a prune command names BenchFlow's layer label and ``key``."""
    text = " ".join(command)
    return (
        f"label={LAYER_LABEL}=agent" in text or "benchflow[.]layer=agent" in text
    ) and key in text


class TestPrune:
    """The label-scoped prune covers agent layers: it lists layers by
    BenchFlow's label, removes each by its own tag once its labels are
    checked, prunes records by the key's marker, and touches nothing else."""

    async def _build(self, docker: FakeDocker, n: int) -> list[str]:
        keys = []
        for i in range(n):
            docker.add_image(f"bf__task{i}", layers=[f"sha256:{i:064x}"])
            result = await DockerLayers().ensure_agent_layer(
                RECIPE, f"bf__task{i}", prune=False
            )
            DockerLayers.release(result.key, daemon=result.daemon_id)
            keys.append(result.key)
        return keys

    async def test_every_prune_command_names_the_label_and_a_key(self, docker):
        keys = await self._build(docker, 5)
        other = docker.add_image("someone-else:latest", labels={"maintainer": "x"})
        docker.records.append(
            {"id": "theirs", "description": "mount / from exec /bin/sh -c npm ci"}
        )
        docker.calls.clear()
        report = await DockerLayers().prune(keep=2)
        assert sorted(report.removed) == sorted(keys[:3])
        assert sorted(report.kept) == sorted(keys[3:])
        assert ["image", "ls", "--filter", f"label={LAYER_LABEL}=agent"] == (
            docker.calls[0][:4]
        )
        commands = docker.prune_commands()
        assert commands
        for command in commands:
            if command[:2] == ["image", "rm"]:
                # BenchFlow's own tag for one key, after reading its labels.
                assert command[2] == "--no-prune"
                assert command[3:] in [[layer_tag(AGENT, k)] for k in keys], command
                index = docker.calls.index(command)
                assert ["image", "inspect", command[3]] in docker.calls[:index]
            else:
                assert command[:2] == ["builder", "prune"], command
                assert any(_named_label(command, k) for k in keys), command
                assert "--filter" in command, f"an unfiltered prune: {command}"
        assert docker.tags["someone-else:latest"] is other
        assert any(r["id"] == "theirs" for r in docker.records)
        assert (
            len(
                [
                    r
                    for r in docker.records
                    if "benchflow.layer=agent" in r["description"]
                ]
            )
            == 2
        )

    async def test_a_record_prune_matches_its_key_alone(self, docker):
        keys = await self._build(docker, 2)
        await DockerLayers().prune(keep=0, only=[keys[0]])
        left = [r["description"] for r in docker.records]
        assert not any(keys[0] in d for d in left)
        assert any(keys[1] in d for d in left)

    async def test_layers_a_trial_runs_from_are_kept(self, docker):
        docker.add_image("bf__task", layers=["sha256:" + "f" * 64])
        held = await DockerLayers().ensure_agent_layer(
            RECIPE, "bf__task", alias="bf__layer__t1", prune=False
        )
        others = await self._build(docker, 2)
        report = await DockerLayers().prune(keep=0)
        assert held.key in report.in_use
        assert sorted(report.removed) == sorted(others)
        assert layer_tag(AGENT, held.key) in docker.tags
        # Released, but its trial tag is still there: still in use.
        DockerLayers.release(held.key, daemon=held.daemon_id)
        report = await DockerLayers().prune(keep=0)
        assert report.in_use == [held.key]
        # Teardown removed the trial tag.
        del docker.tags["bf__layer__t1"]
        report = await DockerLayers().prune(keep=0)
        assert report.removed == [held.key]

    async def test_another_processs_trial_tag_keeps_its_layer(self, docker):
        """A trial of another BenchFlow process has tagged the layer and not
        started its container yet: no claim here, no container, but its tag."""
        keys = await self._build(docker, 1)
        docker.tags["bf__layer__other-process-trial"] = docker.tags[
            layer_tag(AGENT, keys[0])
        ]
        report = await DockerLayers().prune(keep=0)
        assert report.in_use == keys and not report.removed

    @staticmethod
    def _dead_owner_tag(monkeypatch) -> str:
        """An alias tag of a process on this host that is gone: another boot
        and a pid that does not exist (gone on Linux and elsewhere)."""
        dead_pid = 4_000_001
        real = _docker_sweep._pid_exists
        monkeypatch.setattr(
            _docker_sweep, "_pid_exists", lambda pid: pid != dead_pid and real(pid)
        )
        host = env_layers._host_hash(_docker_sweep.current_process().host)
        tag = f"{host}-{dead_pid}-123-00000000-0000-0000-0000-000000000000"
        assert alias_owner_state(tag) == "gone"
        return tag

    async def test_a_dead_processs_trial_tag_is_dropped_and_its_layer_pruned(
        self, docker, monkeypatch
    ):
        """A process killed before its trial's teardown left the trial's tag
        on the layer. That tag holds nothing: the prune untags it and removes
        the layer like any other past the bounds."""
        keys = await self._build(docker, 1)
        alias = f"bf__layer__killed-trial:{self._dead_owner_tag(monkeypatch)}"
        docker.tags[alias] = docker.tags[layer_tag(AGENT, keys[0])]
        report = await DockerLayers().prune(keep=0)
        assert report.aliases_removed == [alias]
        assert report.removed == keys and not report.in_use
        assert alias not in docker.tags
        assert layer_tag(AGENT, keys[0]) not in docker.tags

    async def test_a_dead_processs_trial_tag_counts_for_no_bytes(
        self, docker, monkeypatch
    ):
        """Leftover tags must not fill the byte bound: before, each held its
        layer's bytes forever, and every later build pruned all other layers."""
        docker.build_size = 4 * env_layers._GB
        keys = await self._build(docker, 4)
        dead = self._dead_owner_tag(monkeypatch)
        for i, key in enumerate(keys[:2]):
            docker.tags[f"bf__layer__killed-{i}:{dead}"] = docker.tags[
                layer_tag(AGENT, key)
            ]
        report = await DockerLayers().prune(keep=64, max_bytes=10 * env_layers._GB)
        assert len(report.aliases_removed) == 2 and not report.in_use
        assert report.kept == keys[:1:-1]  # the newest two, 8 GB
        assert sorted(report.removed) == sorted(keys[:2])
        assert report.bytes_left == 8 * env_layers._GB

    async def test_a_dead_processs_trial_tag_stays_while_a_container_uses_it(
        self, docker, monkeypatch
    ):
        keys = await self._build(docker, 1)
        alias = f"bf__layer__killed-trial:{self._dead_owner_tag(monkeypatch)}"
        docker.tags[alias] = docker.tags[layer_tag(AGENT, keys[0])]
        docker.used_ids.add(docker.tags[alias].id)
        report = await DockerLayers().prune(keep=0)
        assert not report.aliases_removed and report.in_use == keys
        assert alias in docker.tags

    async def test_a_live_processs_trial_tag_keeps_its_layer(self, docker):
        """This process's own tag (a trial between tag and claim, or one whose
        teardown failed to untag) is never dropped while it runs."""
        keys = await self._build(docker, 1)
        alias = f"bf__layer__running-trial:{alias_owner_tag()}"
        docker.tags[alias] = docker.tags[layer_tag(AGENT, keys[0])]
        report = await DockerLayers().prune(keep=0)
        assert report.in_use == keys and not report.aliases_removed
        assert alias in docker.tags

    def test_an_alias_tag_names_this_process_and_is_a_valid_tag(self):
        tag = alias_owner_tag()
        assert re.fullmatch(r"[\w][\w.-]{0,127}", tag), tag
        assert alias_owner_state(tag) == "this"

    @pytest.mark.parametrize(
        "tag",
        ["latest", "", "zzz", f"{'0' * 12}-1-1-x", "abc:def"],
        ids=["latest", "empty", "junk", "other-host", "colon"],
    )
    def test_an_alias_tag_of_unknown_form_or_host_is_unknown(self, tag):
        assert alias_owner_state(tag) == "unknown"

    async def test_a_layer_a_container_uses_stays(self, docker):
        keys = await self._build(docker, 1)
        docker.used_ids.add(docker.tags[layer_tag(AGENT, keys[0])].id)
        report = await DockerLayers().prune(keep=0)
        assert report.in_use == keys and not report.removed
        assert any(keys[0] in r["description"] for r in docker.records)

    async def test_orphaned_records_are_pruned_by_key(self, docker):
        keys = await self._build(docker, 2)
        del docker.tags[layer_tag(AGENT, keys[0])]  # removed by hand
        docker.calls.clear()
        report = await DockerLayers().prune(keep=10)
        assert report.records_pruned == [keys[0]]
        assert all(_named_label(c, keys[0]) for c in docker.prune_commands())

    async def test_a_build_prunes_past_the_keep(self, docker, monkeypatch):
        monkeypatch.setenv(env_layers.AGENT_LAYER_KEEP_ENV, "3")
        for i in range(10):
            docker.add_image(f"bf__task{i}", layers=[f"sha256:{i:064x}"])
            result = await DockerLayers().ensure_agent_layer(RECIPE, f"bf__task{i}")
            DockerLayers.release(result.key, daemon=result.daemon_id)
        layer_tags = [t for t in docker.tags if t.startswith(f"{LAYER_REPOSITORY}:")]
        assert len(layer_tags) == 3
        assert (
            len(
                [
                    r
                    for r in docker.records
                    if "benchflow.layer=agent" in r["description"]
                ]
            )
            == 3
        )

    async def test_another_kind_is_not_touched(self, docker):
        keys = await self._build(docker, 1)
        docker.add_image(
            f"{LAYER_REPOSITORY}:proxy-{'e' * 64}",
            labels={LAYER_LABEL: "proxy", LAYER_KEY_LABEL: "e" * 64},
        )
        docker.records.append(
            {
                "id": "p",
                "description": f"mount / from exec /bin/sh -c : benchflow.layer=proxy key={'e' * 64}; uv sync",
            }
        )
        await DockerLayers().prune(keep=0)
        assert layer_tag(AGENT, keys[0]) not in docker.tags
        assert f"{LAYER_REPOSITORY}:proxy-{'e' * 64}" in docker.tags
        assert any(r["id"] == "p" for r in docker.records)

    async def test_a_recovery_lease_keeps_the_layer_and_its_tag(self, docker):
        """A trial's verifier-recovery lease tags the image main ran from (the
        layer) and outlives the trial's teardown: no claim, no trial tag, no
        container. The prune must leave the image and the lease tag."""
        keys = await self._build(docker, 1)
        lease = "benchflow-recovery-lease:" + "9" * 32
        docker.tags[lease] = docker.tags[layer_tag(AGENT, keys[0])]
        report = await DockerLayers().prune(keep=0)
        assert report.in_use == keys and not report.removed
        assert docker.tags[lease] is docker.tags[layer_tag(AGENT, keys[0])]
        assert any(keys[0] in r["description"] for r in docker.records)

    async def test_an_image_committed_from_a_layer_survives_its_prune(self, docker):
        """A snapshot (or a user's image) made from a layer's container
        inherits the layer's labels; removing the layer leaves it."""
        keys = await self._build(docker, 1)
        layer = docker.tags[layer_tag(AGENT, keys[0])]
        snap = docker.add_image(
            "bf-snap-task-1",
            layers=[*layer.info["RootFS"]["Layers"], "sha256:" + "5" * 64],
            labels=dict(layer.info["Config"]["Labels"]),
        )
        report = await DockerLayers().prune(keep=0)
        assert report.removed == keys
        assert layer_tag(AGENT, keys[0]) not in docker.tags
        assert docker.tags["bf-snap-task-1"] is snap

    async def test_a_tag_holding_something_else_is_left_alone(self, docker):
        """BenchFlow's tag on an image whose key label is not the tag's key."""
        key = "d" * 64
        docker.add_image(
            layer_tag(AGENT, key),
            labels={LAYER_LABEL: AGENT, LAYER_KEY_LABEL: "e" * 64},
        )
        report = await DockerLayers().prune(keep=0)
        assert report.kept == [key] and not report.removed
        assert layer_tag(AGENT, key) in docker.tags

    async def test_kept_layers_fit_the_byte_bound(self, docker):
        """A layer holds its whole task image: the newest that fit stay."""
        docker.build_size = 3 * env_layers._GB
        keys = await self._build(docker, 5)
        report = await DockerLayers().prune(keep=64, max_bytes=10 * env_layers._GB)
        assert report.kept == keys[:-4:-1]  # the newest three, 9 GB
        assert sorted(report.removed) == sorted(keys[:2])
        assert report.bytes_removed == 6 * env_layers._GB
        assert report.bytes_left == 9 * env_layers._GB

    async def test_layers_in_use_count_toward_the_byte_bound(self, docker):
        docker.build_size = 3 * env_layers._GB
        others = await self._build(docker, 2)
        docker.build_size = 8 * env_layers._GB
        docker.add_image("bf__big", layers=["sha256:" + "8" * 64])
        held = await DockerLayers().ensure_agent_layer(RECIPE, "bf__big", prune=False)
        report = await DockerLayers().prune(keep=64, max_bytes=10 * env_layers._GB)
        assert report.in_use == [held.key]
        assert sorted(report.removed) == sorted(others)

    async def test_an_older_layer_in_use_counts_before_newer_idle_ones(self, docker):
        """A long trial still runs from an old 8 GB layer A while newer
        trials leave B (released) and C (held): A's bytes count before B is
        let in, so B goes, though it is newer than A."""
        gb = env_layers._GB
        layers = DockerLayers()
        sizes = {"a": 8 * gb, "b": 3 * gb, "c": 3 * gb}
        keys = {}
        for i, name in enumerate(sizes):
            docker.build_size = sizes[name]
            docker.add_image(f"bf__{name}", layers=[f"sha256:{i + 1:064x}"])
            result = await layers.ensure_agent_layer(RECIPE, f"bf__{name}", prune=False)
            keys[name] = result.key
        DockerLayers.release(keys["b"], daemon=docker.daemon_id)
        report = await layers.prune(keep=64, max_bytes=10 * gb)
        assert sorted(report.in_use) == sorted([keys["a"], keys["c"]])
        assert report.removed == [keys["b"]]
        assert report.kept == []
        assert report.bytes_left == 11 * gb
        assert layer_tag(AGENT, keys["b"]) not in docker.tags

    async def test_an_older_layer_in_use_takes_a_place_in_the_keep(self, docker):
        """The same order for the count: with keep 2, a held oldest layer and
        a held newest one leave no place for the idle one between them."""
        layers = DockerLayers()
        keys = []
        for i in range(3):
            docker.add_image(f"bf__k{i}", layers=[f"sha256:{i + 1:064x}"])
            result = await layers.ensure_agent_layer(RECIPE, f"bf__k{i}", prune=False)
            keys.append(result.key)
        DockerLayers.release(keys[1], daemon=docker.daemon_id)
        report = await layers.prune(keep=2)
        assert sorted(report.in_use) == sorted([keys[0], keys[2]])
        assert report.removed == [keys[1]]

    async def test_a_low_disk_takes_back_more_layers_oldest_first(
        self, docker, monkeypatch
    ):
        docker.build_size = 3 * env_layers._GB
        keys = await self._build(docker, 5)

        async def free(_self):
            return 2 * env_layers._GB

        monkeypatch.setattr(DockerLayers, "_free_bytes", free)
        report = await DockerLayers().prune(keep=64, min_free=10 * env_layers._GB)
        # 8 GB short: the three oldest (9 GB) go.
        assert sorted(report.removed) == sorted(keys[:3])
        assert report.kept == keys[:2:-1]

    async def test_a_build_prunes_past_the_byte_bound(self, docker, monkeypatch):
        """With delete, teardown removes the task image but not the layer's
        copy of it: what builds keep is bounded by bytes, not only count."""
        monkeypatch.setenv("BENCHFLOW_AGENT_LAYER_MAX_GB", "10")
        monkeypatch.delenv(env_layers.AGENT_LAYER_KEEP_ENV, raising=False)
        docker.build_size = 3_000_000_000  # 3 GB a layer, task image included
        for i in range(10):
            docker.add_image(f"bf__task{i}", layers=[f"sha256:{i:064x}"])
            result = await DockerLayers().ensure_agent_layer(RECIPE, f"bf__task{i}")
            DockerLayers.release(result.key, daemon=result.daemon_id)
            del docker.tags[f"bf__task{i}"]  # teardown with delete
        layer_tags = [t for t in docker.tags if t.startswith(f"{LAYER_REPOSITORY}:")]
        assert len(layer_tags) == 3

    def test_the_byte_bounds_come_from_the_environment(self, monkeypatch):
        gb = env_layers._GB
        for name in (
            env_layers.AGENT_LAYER_MAX_GB_ENV,
            env_layers.AGENT_LAYER_MIN_FREE_GB_ENV,
        ):
            monkeypatch.delenv(name, raising=False)
        assert env_layers.agent_layer_max_bytes() == int(
            env_layers.DEFAULT_AGENT_LAYER_MAX_GB * gb
        )
        assert env_layers.agent_layer_min_free_bytes() == int(
            env_layers.DEFAULT_AGENT_LAYER_MIN_FREE_GB * gb
        )
        monkeypatch.setenv(env_layers.AGENT_LAYER_MAX_GB_ENV, "2.5")
        assert env_layers.agent_layer_max_bytes() == int(2.5 * gb)
        monkeypatch.setenv(env_layers.AGENT_LAYER_MAX_GB_ENV, "0")
        assert env_layers.agent_layer_max_bytes() is None
        for bad in ("lots", "-1", "nan", "inf"):
            monkeypatch.setenv(env_layers.AGENT_LAYER_MIN_FREE_GB_ENV, bad)
            assert env_layers.agent_layer_min_free_bytes() == int(
                env_layers.DEFAULT_AGENT_LAYER_MIN_FREE_GB * gb
            )

    async def test_free_disk_is_read_only_for_a_daemon_on_this_machine(
        self, docker, monkeypatch, tmp_path
    ):
        from benchflow.sandbox import _docker_subnets as subnets

        # Model the CLI's effective Unix endpoint; a FakeDocker info response
        # by itself cannot prove that a same-named daemon is on this machine.
        monkeypatch.setattr(
            subnets, "_uses_local_socket", lambda key: not key.env.get("DOCKER_HOST")
        )
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
        monkeypatch.setattr(env_layers.socket, "gethostname", lambda: "this-host")
        docker.info = f"{tmp_path}\tthis-host\n"
        free = await DockerLayers()._free_bytes()
        assert free is not None and free > 0
        # Docker Desktop: the daemon runs in a VM with another name.
        docker.info = f"{tmp_path}\tdocker-desktop\n"
        assert await DockerLayers()._free_bytes() is None
        docker.info = f"{tmp_path}\tthis-host\n"
        calls = len(docker.calls)
        remote = DockerLayers(env={"DOCKER_HOST": "tcp://10.0.0.5:2376"})
        assert await remote._free_bytes() is None
        assert docker.calls[calls:] == [
            ["info", "--format", "{{.ID}}"]
        ]  # identity only; no disk probe

    def test_the_keep_comes_from_the_environment(self, monkeypatch):
        monkeypatch.delenv(env_layers.AGENT_LAYER_KEEP_ENV, raising=False)
        assert env_layers.agent_layer_keep() == env_layers.DEFAULT_AGENT_LAYER_KEEP
        monkeypatch.setenv(env_layers.AGENT_LAYER_KEEP_ENV, "5")
        assert env_layers.agent_layer_keep() == 5
        monkeypatch.setenv(env_layers.AGENT_LAYER_KEEP_ENV, "many")
        assert env_layers.agent_layer_keep() == env_layers.DEFAULT_AGENT_LAYER_KEEP

    def test_the_record_filter_is_quoted_for_buildkit(self):
        key = "a" * 64
        spec = env_layers._marker_filter(AGENT, key)
        assert spec == f'description~="benchflow[.]layer=agent key={key};"'
        assert shlex.split(spec) == [f"description~=benchflow[.]layer=agent key={key};"]
