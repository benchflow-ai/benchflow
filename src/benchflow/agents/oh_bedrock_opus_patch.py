# Deployed into the openhands sandbox's litellm (via registry.py install_cmd) to make
# Bedrock Claude Opus/Sonnet/Haiku 4.8+ use ADAPTIVE thinking at MAX effort.
#
# litellm flags only the direct-anthropic id (`claude-opus-4-8`) as adaptive-thinking,
# NOT the Bedrock inference-profile ids (`us./eu./global./bare anthropic.claude-opus-4-8`).
# So on Daytona (direct-bedrock routing) the converse transform emits the legacy
# `thinking.type=enabled` that Bedrock 4.8 rejects (ACP -32603 / 400 ValidationException).
#
# This shim is regex-gated to 4.8+ ids => a strict no-op for every other model
# (gemini, azure, opus-4.6, etc. import-and-call straight through to the originals).
# It is never imported by benchflow itself; it is base64-shipped and auto-loaded in the
# sandbox via a .pth file. Requires litellm>=1.88.0rc1 (the `max` effort + output_config path).
import re

_NEW = re.compile(r"claude-(opus|sonnet|haiku)-4-(8|9|1\d)")

# (1) gate: treat 4.8+ as adaptive-thinking models
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

# (2) force MAX effort for 4.8+ (openhands caps its own effort at xhigh)
try:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    _orig_handle = AmazonConverseConfig._handle_reasoning_effort_parameter

    def _handle(self, model, reasoning_effort, optional_params):
        if model and _NEW.search(model.lower()):
            reasoning_effort = "max"
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
