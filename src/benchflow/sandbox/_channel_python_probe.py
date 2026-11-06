"""Shared dependency probe for channel subset derivation and startup."""

from __future__ import annotations

import ast
import textwrap


def import_source(sources: tuple[str, ...]) -> str:
    """Load every dependency and check the attributes referenced by the helpers."""
    imports = []
    attributes: set[str] = set()
    calls: set[str] = set()
    for source in sources:
        tree = ast.parse(source)
        found = [
            n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        if any(n not in tree.body for n in found):
            raise ValueError("channel imports must precede command acceptance")
        imports.extend(found)
        names = {
            alias.asname or alias.name.split(".")[0]
            for node in found
            for alias in node.names
        }
        for node in ast.walk(tree):
            value = node.func if isinstance(node, ast.Call) else node
            root = value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in names:
                continue
            if isinstance(node, ast.Attribute):
                attributes.add(ast.unparse(node))
            elif isinstance(node, ast.Call):
                calls.add(ast.unparse(node.func))
    checks = [ast.unparse(ast.Module(body=imports, type_ignores=[]))]
    checks.extend(sorted(attributes))
    for call in sorted(calls):
        checks.append(
            f"if not callable({call}):\n"
            "    raise ImportError('nonfunctional channel dependency')"
        )
    return "\n".join(checks) + "\n"


# Frozen modules still expose their source paths. Checking those paths also
# detects missing files that Python itself would not need until later.
FILE_SOURCE = r"""
files = {sys.executable}
for name, module in list(sys.modules.items()):
    if name == '__main__':
        continue
    if (not isinstance(module, type(sys)) or module.__spec__ is None
            or module.__spec__.origin is None):
        raise ImportError('nonfunctional channel dependency')
    path = getattr(module, '__file__', None)
    if path:
        files.add(path)
with open('/proc/self/maps') as stream:
    for line in stream:
        parts = line.split()
        if len(parts) == 6 and parts[5].startswith(sys.prefix + '/'):
            files.add(parts[5])
for path in files:
    if not os.path.isfile(path):
        raise ImportError('missing channel dependency')
"""


def startup_source(sources: tuple[str, ...]) -> str:
    """Fail before the PTY marker, journal or READY without leaking probe globals."""
    return (
        "def _bf_check_python():\n"
        + textwrap.indent(import_source(sources) + FILE_SOURCE, "    ")
        + "\n_bf_check_python()\ndel _bf_check_python\n"
    )
