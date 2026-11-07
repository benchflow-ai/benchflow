"""bfx: BenchFlow's static helper for its own root steps in a sandbox.

After an agent has worked in a sandbox, every process BenchFlow starts there
as root for its own purposes is this one program (TRUSTED-RUNTIME.md §3,
I1): static, pinned by SHA-256 here (:mod:`.pins`), started by absolute path
from a place the agent cannot write (:data:`SANDBOX_PATH`), ignoring its
environment, non-dumpable, with inherited descriptors closed, and acting on
files only beneath directories it holds, never following a link or crossing
a mount. The source is ``bfx/`` in the repository.

- :mod:`.locate`: the verified binary for an architecture.
- :mod:`.client`: one-shot argv and answers, and ``bfx serve``.
- :mod:`.delivery`: how each backend puts it where the agent cannot write,
  and which guarantee that gives.
"""

from benchflow._bfx.client import (
    BfxError,
    BfxProtocolError,
    BfxServe,
    one_shot_argv,
    parse_answer,
)
from benchflow._bfx.locate import (
    SANDBOX_DIR,
    SANDBOX_PATH,
    BfxUnavailable,
    available,
    normalize_arch,
    verified_binary,
)

__all__ = [
    "SANDBOX_DIR",
    "SANDBOX_PATH",
    "BfxError",
    "BfxProtocolError",
    "BfxServe",
    "BfxUnavailable",
    "available",
    "normalize_arch",
    "one_shot_argv",
    "parse_answer",
    "verified_binary",
]
