#!/usr/bin/env bash
# Harbor RewardKit LLM-as-judge (final claim step). Issue #18.
set -euo pipefail
mkdir -p /logs/verifier
node /tests/treatment-gate.mjs

if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

export REWARDKIT_JUDGE="${REWARDKIT_JUDGE:-openai/gpt-5.5}"
# Drop unsupported LiteLLM params for OpenAI chat models.
export LITELLM_DROP_PARAMS="${LITELLM_DROP_PARAMS:-1}"

# documents extra not required when judging .mjs only; keep deps lean.
uvx --from 'harbor-rewardkit==0.1.7' --with pyyaml rewardkit /tests
