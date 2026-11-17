"""What each agent's session offers for an effort, checked against the pinned
packages, and the refusals that follow from it (R383D, EN-383)."""

import json
import shutil
import subprocess

import pytest

from benchflow.agents import opencode_catalog
from benchflow.effort_request import (
    EffortRequestError,
    session_takes_effort,
    settle_effort,
)
from tests.test_effort_request import _bench_run, _shell

#: The OpenCode release BenchFlow installs. Its effort option exists only
#: for a model with variants: ``function Xt(e){if(e.variants.length===0)
#: return;return{id:"effort",...category:"thought_level"...`` (read from
#: opencode-linux-x64 OPENCODE_VERSION, 2026-11-14). A bump fails
#: test_each_declared_effort_option_is_one_the_pinned_agent_offers until
#: that is read again.
OPENCODE_VERSION = "1.18.11"
#: What each agent that declares an effort option offers, at the version it
#: is pinned to, read from the package (R383D and FX383, 2026-11-14):
#: claude-agent-acp's ``EFFORT_CONFIG_ID = "effort"``, codex-acp's
#: ``model[effort]`` ids (no option), OpenCode's ``effort`` (only with model
#: variants), and BenchFlow's own antigravity and muse shims, checked below.
_DECLARED_OFFERS = {
    "claude-agent-acp": ("0.81.2", "effort"),
    "codex-acp": ("2.0.1", None),
    "opencode": (OPENCODE_VERSION, "effort"),
    "antigravity": (None, "thinking"),
    "muse-code": (None, "effort"),
}


def _pinned(name):
    from benchflow.agents.registry import CACHEABLE_INSTALLS, pinned_npm_package

    if name == "opencode":
        return CACHEABLE_INSTALLS["opencode"]
    return pinned_npm_package(name)[1]


def test_each_declared_effort_option_is_one_the_pinned_agent_offers():
    """R383D M1: the completeness test trusted declarations. Each one is
    checked here against what the pinned package offers: a new declaration,
    or a bump of a pinned version, fails until it is read again."""
    from benchflow.agents.antigravity_acp_shim import AgySession
    from benchflow.agents.muse_acp_shim import MuseSession
    from benchflow.agents.registry import _CORE_AGENT_CONFIGS as core

    declared = {name for name, config in core.items() if session_takes_effort(config)}
    assert declared == set(_DECLARED_OFFERS)
    for name, (version, option) in _DECLARED_OFFERS.items():
        assert (core[name].acp_effort_config_id or None) == option, name
        if version is not None:
            assert _pinned(name) == version, name
    agy = {option["id"] for option in AgySession("s", "/tmp").config_options()}
    assert core["antigravity"].acp_effort_config_id in agy
    muse = MuseSession("s", "/tmp", None, None).config_options()
    assert core["muse-code"].acp_effort_config_id in {option["id"] for option in muse}
    # Only OpenCode's option depends on the model it is handed.
    assert [name for name, c in core.items() if c.acp_effort_needs_variants] == [
        "opencode"
    ]


def _at_the_agent(env):
    """The names the agent process sees: 0.9.x passes them as they are."""
    return env


def _registered(tmp_path, model):
    """OpenCode's config as its installed -proxy wrapper writes it for
    *model*, with the env the gateway wiring gives it
    (``_wire_litellm_agent_env``), named as the agent sees it."""
    from benchflow.providers.litellm_config import resolve_litellm_route
    from benchflow.providers.litellm_runtime import _wire_litellm_agent_env
    from tests.test_opencode_family_proxy_tracking import _register_js, _wrapper_script

    route = resolve_litellm_route(model, {})
    env = _wire_litellm_agent_env(
        agent="opencode",
        agent_env={},
        route=route,
        base_url="http://127.0.0.1:4000",
        master_key="fake-master-key",
    )
    env = _at_the_agent({"BENCHFLOW_AGENT_HOME": str(tmp_path), **env})
    register_js = _register_js(_wrapper_script("opencode", "opencode-proxy"))
    run_env = {"PATH": "/usr/bin:/bin", **env}
    node = shutil.which("node")
    subprocess.run([node, "-"], input=register_js, text=True, env=run_env, check=True)
    config = json.loads((tmp_path / ".config/opencode/opencode.json").read_text())
    return route, config["provider"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize(
    ("model", "offered"),
    [
        ("anthropic/claude-opus-5-5", False),
        ("openai/gpt-5.5", False),
        ("openrouter/anthropic/claude-opus-5.5", False),
        ("deepseek/deepseek-v4-pro", False),
        ("aws-bedrock/us.anthropic.claude-opus-5-5", False),
        ("gemini/gemini-4-argon", True),
    ],
)
def test_opencode_takes_an_effort_only_where_the_wrapper_gives_its_model_variants(
    tmp_path, model, offered
):
    """R383D M1: -a opencode -e passed the dry run and exited 1 after install
    on every route but Gemini ("does not expose reasoning effort config
    option 'effort'; available options: ['mode', 'model']"). The wrapper's
    own registration, run here as the trial runs it, decides: the gateway
    alias is ``{ name: alias }`` with no variants, so OpenCode builds no
    option; a gemini/ route goes to OpenCode's google provider, whose catalog
    gives the model its variants."""
    from benchflow.agents.registry import (
        OPENCODE_PROXY_PROVIDER_ID,
        registers_model_variants,
    )

    route, providers = _registered(tmp_path, model)
    alias = providers.get(OPENCODE_PROXY_PROVIDER_ID)
    if alias is not None:
        has_variants = any(entry.get("variants") for entry in alias["models"].values())
    else:
        has_variants = "google" in providers
    assert has_variants is offered
    assert registers_model_variants("opencode", route.upstream_model) is offered
    if offered:
        assert settle_effort("opencode", model, "high", {}).effort == "high"
    else:
        with pytest.raises(EffortRequestError) as caught:
            settle_effort("opencode", model, "high", {})
        assert str(caught.value) == (
            "reasoning effort high can't reach the opencode session: it offers an "
            f"effort option only for a model with variants, and the gateway "
            f"registers {model} for it with none (a gemini/ model, on its own "
            "google provider, has them); drop the effort.\n"
            "Nothing ran, so nothing was recorded."
        )
        assert caught.value.hint == "drop the effort"


@pytest.mark.parametrize(
    ("argv", "keys", "words"),
    [
        (
            ("-a", "opencode", "-m", "anthropic/claude-opus-5-5", "-e", "high"),
            {"ANTHROPIC_API_KEY": "fake-not-a-key"},
            "reasoning effort high can't reach the opencode session",
        ),
        (
            ("-a", "opencode", "-m", "openai/gpt-5.5", "-e", "low"),
            {"OPENAI_API_KEY": "fake-not-a-key"},
            "reasoning effort low can't reach the opencode session",
        ),
        (
            ("-a", "claude", "-m", "anthropic/claude-opus-4-1", "-e", "high"),
            {"ANTHROPIC_API_KEY": "fake-not-a-key"},
            "anthropic/claude-opus-4-1 takes no reasoning effort (claude-agent-acp)",
        ),
        (
            ("-a", "claude", "-m", "anthropic/claude-sonnet-4-20250514", "-e", "max"),
            {"ANTHROPIC_API_KEY": "fake-not-a-key"},
            "anthropic/claude-sonnet-4-20250514 takes no reasoning effort",
        ),
        (
            ("-a", "openhands", "-m", "anthropic/claude-opus-5-5", "-e", "max"),
            {"ANTHROPIC_API_KEY": "fake-not-a-key"},
            "reasoning effort max can't reach the openhands session",
        ),
        (
            ("-a", "opencode", "-m", "gemini/gemini-4-argon", "-e", "max"),
            {"GEMINI_API_KEY": "fake-not-a-key"},
            "gemini/gemini-4-argon takes no reasoning effort max from OpenCode",
        ),
        (
            ("-a", "opencode", "-m", "gemini/gemini-4-hydrogen", "-e", "xhigh"),
            {"GEMINI_API_KEY": "fake-not-a-key"},
            "gemini/gemini-4-hydrogen takes no reasoning effort xhigh from OpenCode",
        ),
        (
            ("-a", "opencode", "-m", "gemini/gemini-2.5-pro", "-e", "low"),
            {"GEMINI_API_KEY": "fake-not-a-key"},
            "gemini/gemini-2.5-pro turns reasoning effort into a thinking budget",
        ),
        (
            ("-a", "acpx/codex", "-m", "openai/gpt-5.5", "-e", "high"),
            {"OPENAI_API_KEY": "fake-not-a-key"},
            "reasoning effort high can't reach the acpx/codex session",
        ),
    ],
)
def test_each_new_refusal_is_the_dry_runs_and_the_runs(
    tmp_path, monkeypatch, argv, keys, words
):
    """R383D: each of these passed the dry run (exit 0) and failed after the
    sandbox and the agent install, exit 1. Now both refuse, word for word,
    before any preflight, and the next: line is the refusal's own."""
    _shell(monkeypatch, tmp_path, **keys)
    texts = []
    for flags in ([], ["--dry-run"]):
        refused, seen = _bench_run(
            tmp_path / ("dry" if flags else "run"), monkeypatch, *argv, *flags
        )
        assert refused.exit_code == 2, refused.output
        assert seen == {"preflight": [], "configs": []}
        texts.append(" ".join(refused.output.split()))
    assert words in texts[0]
    # Only a level the model lacks is answered by asking for another level.
    asks_another = "from OpenCode" in words
    assert ("ask for a level the model takes" in texts[0]) is asks_another
    assert texts[0] == texts[1]


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-1",
        "aws-bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0",
        "openrouter/anthropic/claude-opus-4.1",
        "anthropic/claude-opus-4-20250514",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-3-5-haiku-20241022",
    ],
)
def test_a_claude_model_before_opus_4_5_takes_no_effort(model):
    """R383D L1: Anthropic's reference gives no effort before Opus 4.5;
    Claude Code's -e on these passed the dry run and failed after install."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("claude-agent-acp", model, "high", {})
    assert str(caught.value) == (
        f"{model} takes no reasoning effort (claude-agent-acp): drop high"
    )
    assert caught.value.hint == "drop -e: the model takes no reasoning effort"


def test_openhands_names_its_own_setting_only_for_the_levels_it_carries():
    """R383D L2: at max OpenHands' launch config sends a reasoning field the
    gateway refuses (R383C I1), so the refusal names no setting there."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("openhands", "anthropic/claude-opus-5-5", "max", {})
    assert str(caught.value) == (
        "reasoning effort max can't reach the openhands session: its registry "
        "entry declares no ACP effort option, so BenchFlow can't hand it one; "
        "drop the effort (its own setting, --agent-env "
        "LLM_REASONING_EFFORT=<level>, carries none, low, medium, high, xhigh, "
        "not max).\nNothing ran, so nothing was recorded."
    )
    assert "LLM_REASONING_EFFORT=max" not in caught.value.hint
    for level in ("low", "xhigh"):
        with pytest.raises(EffortRequestError) as caught:
            settle_effort("openhands", "anthropic/claude-opus-5-5", level, {})
        assert f"--agent-env LLM_REASONING_EFFORT={level}," in caught.value.hint


@pytest.mark.parametrize("spec", ["acpx/codex", "oh", "acp:mimo"])
def test_a_refusal_names_the_agent_as_the_run_named_it(tmp_path, monkeypatch, spec):
    """R383D N1, R383E N1: through the real CLI, which resolves the alias
    first, the refusal still names the agent as it was typed."""
    _shell(monkeypatch, tmp_path, OPENAI_API_KEY="fake-not-a-key")
    refused, seen = _bench_run(
        tmp_path, monkeypatch, "-a", spec, "-m", "openai/gpt-5.5", "-e", "high"
    )
    assert refused.exit_code == 2, refused.output
    assert seen == {"preflight": [], "configs": []}
    assert f"can't reach the {spec} session" in " ".join(refused.output.split())


def test_each_bedrock_id_a_refusal_names_is_one_the_gateway_knows():
    """R383D N2: the refusals name a Bedrock model id, each one LiteLLM's
    (the gateway's) model map lists."""
    import litellm

    from benchflow.effort_request import _BEDROCK_CLAUDE_IDS

    for dashed, bedrock in _BEDROCK_CLAUDE_IDS.items():
        assert bedrock in litellm.model_cost, dashed
        assert dashed in bedrock, dashed


def test_the_gemini_effort_table_was_measured_on_the_pinned_opencode():
    """R383E M1: the table BenchFlow refuses Gemini levels from was measured on
    the OpenCode it installs; a bump of the pin fails here until it is
    measured again (tests/test_opencode_catalog.py: the table is what the pinned
    OpenCode sends against the pinned catalog)."""
    from benchflow._utils.config import EFFORT_LEVELS
    from benchflow.agents.registry import (
        CACHEABLE_INSTALLS,
        OPENCODE_GEMINI_EFFORTS,
        OPENCODE_GEMINI_EFFORTS_VERSION,
        OPENCODE_GEMINI_UNSENT,
    )

    assert CACHEABLE_INSTALLS["opencode"] == OPENCODE_GEMINI_EFFORTS_VERSION
    assert OPENCODE_GEMINI_EFFORTS_VERSION == OPENCODE_VERSION
    for model, levels in OPENCODE_GEMINI_EFFORTS.items():
        assert set(levels) <= set(EFFORT_LEVELS), model
    for model, levels in OPENCODE_GEMINI_UNSENT.items():
        assert set(levels) <= set(EFFORT_LEVELS) - set(OPENCODE_GEMINI_EFFORTS[model])


@pytest.mark.parametrize(
    ("model", "level", "taken"),
    [
        ("gemini/gemini-4-argon", "low", True),
        ("gemini/gemini-4-argon", "medium", True),
        ("gemini/gemini-4-argon", "high", True),
        ("gemini/gemini-4-argon", "max", False),
        ("gemini/gemini-4-argon", "minimal", False),
        ("gemini/gemini-3-flash-preview", "minimal", True),
        ("gemini/gemini-4-hydrogen", "high", True),
        ("gemini/gemini-4-hydrogen", "xhigh", False),
        ("gemini/gemma-4-31b-it", "low", True),
        ("gemini/gemma-4-31b-it", "minimal", False),
        ("gemini/gemini-embedding-001", "high", False),
        ("gemini/gemini-2.5-pro", "low", False),
        ("gemini/gemini-2.5-pro", "max", False),
    ],
)
def test_opencode_takes_on_gemini_only_the_levels_it_sends(model, level, taken):
    """R383E M1: -a opencode -m gemini/<model> -e <level> passed the dry run
    at any level, and a level the pinned OpenCode builds no variant for failed
    after install ("effort not found: max"), one it lists but can't send at
    the first call ("invalid google provider options": hydrogen's xhigh);
    Gemini 2.5's levels are thinking budgets (R093 P1's ruling: refused); a
    model with no effort option takes none."""
    if taken:
        assert settle_effort("opencode", model, level, {}).effort == level
    else:
        with pytest.raises(EffortRequestError):
            settle_effort("opencode", model, level, {})


def test_the_gemini_refusals_name_what_opencode_sends():
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/gemini-4-argon", "max", {})
    assert str(caught.value) == (
        "gemini/gemini-4-argon takes no reasoning effort max from OpenCode "
        f"{OPENCODE_VERSION} (opencode): it sends low, medium, high for this model; ask "
        "for one of them"
    )
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/gemini-4-hydrogen", "xhigh", {})
    assert str(caught.value) == (
        "gemini/gemini-4-hydrogen takes no reasoning effort xhigh from OpenCode "
        f"{OPENCODE_VERSION} (opencode): BenchFlow's pinned OpenCode catalog "
        f"{opencode_catalog.DATE} lists xhigh for this model, but its "
        'google provider refuses that level at the first call ("invalid google '
        'provider options"); it sends low, medium, high; ask for one of them'
    )
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("opencode", "gemini/gemini-2.5-pro", "high", {})
    assert str(caught.value) == (
        "gemini/gemini-2.5-pro turns reasoning effort into a thinking budget "
        f"(opencode): OpenCode {OPENCODE_VERSION} offers it high, max, each sent "
        "as generationConfig.thinkingConfig.thinkingBudget and no level, so -e "
        "high can't run as asked"
    )
    assert caught.value.hint == "drop -e: no level reaches this model as asked"


def test_a_trial_no_call_measured_counts_as_unmeasured():
    """R383E L2: summary.json's effort_unmeasured left out the reason a
    trial gets when no call carried an effort."""
    from benchflow.effort_record import effort_unmeasured

    for reason in (
        "unmeasured",
        "unmeasured: no call carried an effort",
        "partly unmeasured",
    ):
        results = {"t": {"effort": {"requested": "high", "reason": reason}}}
        assert effort_unmeasured(results) == 1, reason
