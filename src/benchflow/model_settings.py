"""The sampling a model's vendor recommends, where BenchFlow has it on file.

The run card shows the entry for the run's model beside what BenchFlow itself
sets for an agent trial (nothing: the agent sets its own sampling), so a run
that evaluates a model away from its vendor's setting shows before it starts.
Each entry names the page it was read from and when; a model with no entry
gets a line saying so, which is a prompt to read its model card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VendorSetting:
    """One vendor's published sampling for a family of models."""

    vendor: str
    covers: str
    pattern: re.Pattern[str]
    thinking_default: bool
    thinking: dict[str, float] = field(default_factory=dict)
    non_thinking: dict[str, float] = field(default_factory=dict)
    source: str = ""
    read_on: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "covers": self.covers,
            "thinking_default": self.thinking_default,
            "thinking": dict(self.thinking),
            "non_thinking": dict(self.non_thinking),
            "source": self.source,
            "read_on": self.read_on,
        }


# Most specific first: the first pattern that matches the model id wins.
_SETTINGS = (
    VendorSetting(
        vendor="Qwen",
        covers="Qwen3.8-27B",
        pattern=re.compile(r"(?:^|/)qwen3\.8-27b"),
        thinking_default=True,
        thinking={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
        },
        non_thinking={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        },
        source="https://huggingface.co/Qwen/Qwen3.8-27B",
        read_on="2026-11-04",
    ),
    VendorSetting(
        vendor="Qwen",
        covers="Qwen3 (the 2025 models, such as Qwen3-32B)",
        # Not the 2507 updates, whose Instruct models have no thinking mode.
        pattern=re.compile(r"(?:^|/)qwen3-\d[\d.]*b(?:-a\d+b)?(?![\w.-]*2507)"),
        thinking_default=True,
        thinking={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        non_thinking={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
        source="https://huggingface.co/Qwen/Qwen3-32B",
        read_on="2026-11-04",
    ),
)


def vendor_setting(model: str | None) -> VendorSetting | None:
    """The vendor's published sampling for ``model``, or None when none is on file."""
    if not model:
        return None
    name = model.strip().lower()
    for setting in _SETTINGS:
        if setting.pattern.search(name):
            return setting
    return None
