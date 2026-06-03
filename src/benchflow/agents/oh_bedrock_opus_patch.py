# Deployed into the openhands sandbox's litellm (via registry.py install_cmd) to make
# Bedrock Claude Opus/Sonnet/Haiku 4.8+ use the ADAPTIVE thinking contract.
#
# litellm flags only the direct-anthropic id (`claude-opus-4-8`) as adaptive-thinking,
# NOT the Bedrock inference-profile ids (`us./eu./global./bare anthropic.claude-opus-4-8`).
# So on Daytona (direct-bedrock routing) the converse transform emits the legacy
# `thinking.type=enabled` that Bedrock 4.8 rejects (ACP -32603 / 400 ValidationException).
#
# This shim is regex-gated to 4.8+ ids => a strict no-op for every other model
# (gemini, azure, opus-4.6, etc. import-and-call straight through to the originals).
# It is never imported by benchflow itself; it is base64-shipped and auto-loaded in the
# sandbox via a .pth file. Requires litellm>=1.88.0rc1 (the effort + output_config path).
#
# Effort is NOT hardcoded: it is taken from the BENCHFLOW_BEDROCK_THINKING_EFFORT env
# (MAX mode sets it to `max`); unset => the agent's requested effort passes through.
# The model matcher mirrors providers/bedrock_runtime.BEDROCK_ADAPTIVE_THINKING_RE
# (kept identical; tests/test_bedrock_thinking.py pins parity).
import os
import re

_NEW = re.compile(r"claude-(?:opus|sonnet|haiku)-4-(?:8|9|1\d)(?!\d)")

_EFFORT_ENV = "BENCHFLOW_BEDROCK_THINKING_EFFORT"
_VALID_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}

# (1) gate: treat 4.8+ as adaptive-thinking models (the load-bearing fix — Bedrock
# regional ids lack the registry capability flag, so the legacy shape is rejected)
try:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    _orig_gate = AnthropicConfig._is_adaptive_thinking_model

    def _gate(model):
        try:
            if model and _NEW.search(model.lower()):
                return True
        except Exception:
            pass
        return _orig_gate(model)

    AnthropicConfig._is_adaptive_thinking_model = staticmethod(_gate)
except Exception:
    pass

# (2) config-gated effort for 4.8+: override ONLY when BENCHFLOW_BEDROCK_THINKING_EFFORT
# is set (e.g. MAX mode => `max`); otherwise the agent's requested effort is honored.
try:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    _orig_handle = AmazonConverseConfig._handle_reasoning_effort_parameter

    def _handle(self, model, reasoning_effort, optional_params):
        if model and _NEW.search(model.lower()):
            _override = os.environ.get(_EFFORT_ENV, "").strip().lower()
            if _override in _VALID_EFFORTS:
                reasoning_effort = _override
        return _orig_handle(self, model, reasoning_effort, optional_params)

    AmazonConverseConfig._handle_reasoning_effort_parameter = _handle
except Exception:
    pass

# (3) belt-and-suspenders: flip the cost-map flag too
try:
    import litellm

    for _k in list(litellm.model_cost):
        if _NEW.search(_k.lower()):
            litellm.model_cost[_k]["supports_adaptive_thinking"] = True
except Exception:
    pass
