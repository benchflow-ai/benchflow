"""Guards the fix at integration commit 363d815b."""

from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.env_layers import DockerLayers, LayerError
from tests.test_env_layers_agent import RECIPE as AGENT_RECIPE
from tests.test_env_layers_agent import FakeDocker
from tests.test_env_layers_proxy import RECIPE as PROXY_RECIPE


@pytest.mark.parametrize("kind", ["agent", "proxy"])
@pytest.mark.parametrize(
    "trigger",
    ["RUN printf changed > /app/input.txt", "COPY . /app", "ENV TASK_ID=other"],
)
async def test_no_layer_consumes_deferred_task_image_instructions(
    monkeypatch, kind, trigger
):
    """Guards 363d815b: trusted ONBUILD scaffolding is not a layer recipe."""
    fake = FakeDocker()
    image = fake.add_image("trusted-task")
    image.info["Config"]["OnBuild"] = [trigger]
    layers = DockerLayers()
    monkeypatch.setattr(layers, "_docker", fake.run)
    ensure = getattr(layers, f"ensure_{kind}_layer")
    build = AsyncMock(side_effect=AssertionError("ONBUILD was allowed into a build"))
    monkeypatch.setattr(layers, "_ensure", build)
    with pytest.raises(LayerError, match=r"(?i)onbuild"):
        await ensure(AGENT_RECIPE if kind == "agent" else PROXY_RECIPE, "trusted-task")
    build.assert_not_awaited()
