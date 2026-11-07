"""PT-INTEG-R3-1 (round-3 review, high): a kit without a Python on an image
whose only Python sits outside the kit's candidates.

The kit copies a Python only from ``/usr/bin``, ``/bin`` or ``/usr/local/bin``,
while the pre-agent capture runs the first ``python3`` on the image PATH. On a
conda, pyenv or custom-prefix image the kit therefore holds no Python and the
record lists that interpreter's search path. At 6dcfb062 plugin discovery then
failed hardening for every trial with a sandbox user (the data path raised
before the old 127 fallback); integ 27c9387e and e91a5791 score it 1. Adopted
from the reviewer's scratch test: this round's no-kit-Python image, with the
Python at ``/opt/later`` on the image PATH; solve.sh only writes the answer and
test.sh is bash-only.
"""

from __future__ import annotations

import pytest

from tests import test_pt_integ_r3_no_kit_python_docker as base
from tests.exec_diet_fakes import docker_available

DOCKERFILE = base.DOCKERFILE.replace(
    "WORKDIR /app\n",
    "ENV PATH=/opt/later:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
    "WORKDIR /app\n",
)
SOLVE = "#!/bin/bash\necho done > /app/solved.txt\n"


@pytest.mark.deterministic
async def test_python_only_outside_the_kit_candidates_is_scored(tmp_path, monkeypatch):
    """Fails on 6dcfb062 (verifier crashed in plugin discovery)."""
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    monkeypatch.setattr(base, "DOCKERFILE", DOCKERFILE)
    monkeypatch.setattr(base, "SOLVE", SOLVE)
    result = await base._run(tmp_path, "oracle", "agent")
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 1.0}
