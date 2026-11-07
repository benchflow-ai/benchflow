"""Keep agent-writable provenance across Linux mount aliases and ownership freezes."""

import posixpath
import re
import shlex
from dataclasses import dataclass


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _mounts(mountinfo: str) -> list[tuple[str, str, str]]:
    """Parse one complete namespace, preserving the kernel's path bytes."""
    mounts: list[tuple[str, str, str]] = []

    def unescape(value: str) -> str:
        if re.search(r"\\(?!040|011|012|134)", value):
            raise ValueError("invalid Linux mountinfo escape")
        return re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), value)

    # The kernel escapes space, tab, LF and backslash in paths, but leaves
    # other whitespace literal. Unicode split/splitlines would change the
    # coordinates of a valid mount (CONFIRM-lockdown-5).
    for line in mountinfo.removesuffix("\n").split("\n"):
        fields = line.split(" ")
        separator = fields.index("-", 6) if "-" in fields[6:] else -1
        if (
            len(fields) < 10
            or separator != len(fields) - 4
            or any(not field or "\0" in field or "\t" in field for field in fields)
            or not re.fullmatch(r"[0-9]+", fields[0])
            or not re.fullmatch(r"[0-9]+", fields[1])
            or not re.fullmatch(r"[0-9]+:[0-9]+", fields[2])
        ):
            raise ValueError("invalid Linux mountinfo record")
        root, target = (unescape(f) for f in fields[3:5])
        # A bind-mounted namespace (``ip netns``, Docker's netns files) has
        # no directory root, only ``net:[4026532524]``; it holds no files,
        # so it cannot alias an agent-writable tree.
        if fields[-3] == "nsfs" and re.fullmatch(r"[a-z_]+:\[[0-9]+\]", root):
            if not target.startswith("/"):
                raise ValueError("mountinfo paths must be absolute")
            continue
        if not root.startswith("/") or not target.startswith("/"):
            raise ValueError("mountinfo paths must be absolute")
        mounts.append((fields[2], posixpath.normpath(root), posixpath.normpath(target)))
    if not any(target == "/" for _, _, target in mounts):
        raise ValueError("mountinfo has no root mount")
    return mounts


def _coordinates(
    mounts: tuple[tuple[str, str, str], ...], blocked: tuple[str, ...]
) -> frozenset[tuple[str, str]]:
    coordinates: set[tuple[str, str]] = set()
    for prefix in blocked:
        if not prefix.startswith("/") or "\0" in prefix:
            raise ValueError("excluded paths must be absolute, without NUL")
        prefix = posixpath.normpath("/" + prefix.lstrip("/"))
        enclosing = [m for m in mounts if _under(prefix, m[2])]
        longest = max(len(m[2]) for m in enclosing)
        for device, root, target in enclosing:
            if len(target) == longest:
                coordinates.add(
                    (
                        device,
                        posixpath.normpath(
                            posixpath.join(root, posixpath.relpath(prefix, target))
                        ),
                    )
                )
        for device, root, target in mounts:
            if _under(target, prefix):
                coordinates.add((device, root))
    return frozenset(coordinates)


@dataclass(frozen=True)
class PathExclusions:
    """Captured identities, interpreted only in their owning mount namespace.

    Coordinates are (device, filesystem-root path), never container aliases.
    A transfer carries only coordinates; target-local seeds are resolved with
    the target's own mount table. Ownership freezing cannot change this set.
    """

    mounts: tuple[tuple[str, str, str], ...]
    coordinates: frozenset[tuple[str, str]]
    inodes: frozenset[tuple[int, int]] = frozenset()
    inspector: str = ""
    inode_index: str = ""
    # The kit's Python walked the system route and found every way clear
    # (benchflow.sandbox._system_route_walk); the sh walks are then not run.
    system_route_clear: bool = False

    @classmethod
    def capture(cls, mountinfo: str, blocked: tuple[str, ...]) -> "PathExclusions":
        mounts = tuple(_mounts(mountinfo))
        return cls(mounts, _coordinates(mounts, blocked))

    def for_target(self, mountinfo: str, local: tuple[str, ...]) -> "PathExclusions":
        target = self.capture(mountinfo, local)
        # Device/inode pairs identify hard links across bind mounts, even when
        # the filesystem-root names differ. Inspector paths are namespace-local.
        return PathExclusions(
            target.mounts, self.coordinates | target.coordinates, self.inodes
        )

    def _within_mount(self, device: str, root: str, target: str) -> tuple[str, ...]:
        paths = set()
        for source_device, source in self.coordinates:
            if device != source_device:
                continue
            if _under(root, source):
                paths.add(target)
            elif _under(source, root):
                paths.add(
                    posixpath.normpath(
                        posixpath.join(target, posixpath.relpath(source, root))
                    )
                )
        return tuple(sorted(paths))

    def paths(self) -> tuple[str, ...]:
        """Local spellings for the existing plugin/configuration guard API."""
        return tuple(
            sorted(
                {path for mount in self.mounts for path in self._within_mount(*mount)}
            )
        )

    def predicate_groups(self) -> list[tuple[str, list[str]]]:
        """Each mountpoint with its excluded local paths, in the order
        ``bf_under`` tries them: the longest mountpoint first, and the first
        one enclosing a path decides (an empty list excludes nothing there).
        """
        groups: dict[str, set[str]] = {}
        for device, root, target in self.mounts:
            groups.setdefault(target, set()).update(
                self._within_mount(device, root, target)
            )
        return [
            (target, sorted(groups[target]))
            for target in sorted(groups, key=lambda p: (-len(p), p))
        ]

    def shell_predicate(self) -> str:
        """Compile identity membership for a component of a resolved route.

        The longest enclosing mount determines its filesystem coordinates.
        An unrelated overmount must not inherit its parent's exclusions.
        Stacked records at the same mountpoint retain all candidate devices,
        as mountinfo ordering alone does not prove which one is visible.
        Shell literals are quoted, including whitespace and metacharacters.
        """

        def pattern(path: str) -> str:
            return (
                "/*"
                if path == "/"
                else f"{shlex.quote(path)}|{shlex.quote(path + '/')}*"
            )

        lines = ["bf_under() {"]
        if self.inodes:
            if not self.inode_index:
                raise ValueError(
                    "inode exclusions need a protected namespace-local index"
                )
            lines.extend(
                [
                    '  if [ -L "$1" ] || { [ -e "$1" ] && [ ! -d "$1" ]; }; then',
                    '    bf_inode=$(stat -c "%d:%i" -- "$1") || { printf "unreadable identity"; return 0; }',
                    f'    if [ -e {shlex.quote(self.inode_index)}/"$bf_inode" ]; then printf "captured inode %s" "$bf_inode"; return 0; fi',
                    "  fi",
                ]
            )
        lines.append('  case "$1" in')
        for target, paths in self.predicate_groups():
            lines.append(f"    {pattern(target)})")
            for path in paths:
                lines.append(
                    f"      case \"$1\" in {pattern(path)}) printf '%s\\n' {shlex.quote(path)}; return 0 ;; esac"
                )
            lines.append("      return 1 ;;")
        lines.extend(["  esac", "  return 1", "}"])
        return "\n".join(lines)


def blocked_mount_aliases(
    mountinfo: str,
    blocked: tuple[str, ...],
    *,
    target_mountinfo: str | None = None,
) -> tuple[str, ...]:
    """Map blocked subtrees to filesystem coordinates, then to visible aliases.

    A bind mount's realpath does not name its source. Linux mountinfo supplies
    its device and filesystem-root path; the same coordinates in any mount
    name the same agent-controlled tree, before or after chown. Include nested
    mounts as well as aliases of a parent filesystem or a subtree within it.
    A verifier service may expose the same coordinates at entirely different
    paths; project into its namespace when *target_mountinfo* is supplied.
    """
    captured = PathExclusions.capture(mountinfo, blocked)
    if target_mountinfo is not None:
        return captured.for_target(target_mountinfo, ()).paths()
    # Legacy cleanup/plugin callers also retain absent lexical local roots.
    # Only coordinates may cross into another namespace.
    return tuple(sorted({*blocked, *captured.paths()}))
