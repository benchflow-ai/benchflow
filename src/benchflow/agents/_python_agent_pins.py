"""Pinned sources and packages for the agents BenchFlow installs from Python.

deepagents and harvey-lab-harness each install into their own uv venv in the
sandbox. Their packages used to be unpinned (``uv pip install deepagents
langchain-openai``; ``pip install anthropic openai ...``) and harvey-lab cloned
the head of ``harveyai/harvey-labs``, so the same BenchFlow release installed
different agents from one week to the next. A changed agent version can change
a score, so every package below is pinned to an exact version, transitive
dependencies included, and each pin change is a release-note item.

The packaged requirements are ``uv pip compile --universal --python-version 3.12
--generate-hashes --no-header --no-annotate`` output (environment markers kept, so one list
serves x86_64 and aarch64 Linux):

- ``DEEPAGENTS_REQUIREMENTS``: ``deepagents==0.7.26`` and
  ``langchain-openai==1.7.2``, the latest releases on 2026-10-16, which is
  what the unpinned install resolved to that day.
- ``HARVEY_LAB_REQUIREMENTS``: the nine packages the harness needs (anthropic,
  openai, google-genai, python-docx, pdfplumber, openpyxl, python-pptx,
  markitdown, pandas), constrained to the versions in harvey-labs' own
  ``uv.lock`` at ``HARVEY_LAB_COMMIT`` (``uv export --frozen --no-hashes
  --no-dev``), with the security updates listed in the release changelog.

The hash locks are compiled from complete version-pin lists. The 0.9.0 update
changes only advisory-affected transitive pins; agent versions and the harness
source commit stay fixed. Installs require the reviewed artifact hashes.
To move a pin, recompile the list the same way, run the agent's install and
smoke on Docker, and list the old and new versions in the changelog.
"""

from __future__ import annotations

# harveyai/harvey-labs at a2b429eb (2026-08-25): the last commit whose
# ``harness/`` package matches the ACP shim (``harness.tools``,
# ``harness.adapters.*``, ``harness/system_prompt.md``, six tools). dfd17ab1
# (2026-09-08) added a seventh ``finish`` tool the shim's loop does not stop
# on, and cd079c29 (2026-09-14) moved ``harness/`` to ``lab_core/``, so a
# clone of the head has had no ``harness`` module since then.
HARVEY_LAB_REPO = "https://github.com/harveyai/harvey-labs.git"
HARVEY_LAB_COMMIT = "a2b429eb6c9683c4fdeced3bc6b3af36edf239a6"
# The folders the shim reads; the rest of the tree is about 3.2 GB of tasks.
HARVEY_LAB_SPARSE_PATHS = ("harness", "sandbox", "utils")


def _requirements(filename: str) -> tuple[str, ...]:
    from importlib.resources import files

    text = files("benchflow.agents").joinpath("requirements", filename).read_text()
    return tuple(
        " ".join(line.split())
        for line in text.replace("\\\n", "").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


DEEPAGENTS_REQUIREMENTS = _requirements("deepagents.txt")
HARVEY_LAB_REQUIREMENTS = _requirements("harvey-lab.txt")
