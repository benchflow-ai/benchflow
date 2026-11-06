"""A brute-force filesystem model for the integrity access probe's tests.

Independent of ``benchflow.integrity.access``: it resolves paths the way the
kernel does and decides, by trying every operation the agent user may legally
perform, whether it can change what root finds at a protected path. The
permission rules are the kernel's:

- reaching an entry needs search (x) on every folder above its real path;
- renaming or unlinking an entry needs search and write on its folder, and in
  a sticky folder, ownership of the entry or of the folder;
- creating an entry needs search and write on its folder;
- writing a file needs write on it; changing a folder's entries needs write
  on it;
- an owner can always grant itself any mode (chmod), so an owner holds rwx;
- root resolves links (at most ``MAX_LINKS``, else the path is unresolved).

``Tree.probe_output`` prints what the probe script prints for a tree, so the
real script can be compared with it in a container.
"""

from __future__ import annotations

import posixpath
import random
from dataclasses import dataclass, field

MAX_LINKS = 16
AGENT_UID = 1000
AGENT_GIDS = frozenset({1000})
OTHER_UID = 2000


@dataclass
class Node:
    kind: str  # dir | file | link
    uid: int = 0
    gid: int = 0
    mode: int = 0o755
    target: str = ""  # links only
    ident: int = 0  # stays with the object when it is renamed


@dataclass
class Tree:
    nodes: dict[str, Node] = field(default_factory=dict)

    # --- resolution ---------------------------------------------------------

    def resolve(self, path: str) -> tuple[str, str | None, list[str]]:
        """``(end, path, visited)`` as root resolves ``path``: end is found,
        absent or unresolved; visited lists every entry passed, in order."""
        visited = ["/"]
        cur, todo, hops = "/", [p for p in path.split("/") if p], 0
        while todo:
            seg = todo.pop(0)
            if seg == ".":
                continue
            if seg == "..":
                cur = posixpath.dirname(cur) if cur != "/" else "/"
                continue
            nxt = posixpath.join(cur, seg)
            node = self.nodes.get(nxt)
            if node is None or (self.nodes[cur].kind != "dir"):
                return "absent", nxt, visited
            visited.append(nxt)
            if node.kind == "link":
                hops += 1
                if hops > MAX_LINKS or not node.target:
                    return "unresolved", nxt, visited
                target = node.target
                if target.startswith("/"):
                    cur = "/"
                todo = [p for p in target.split("/") if p] + todo
                continue
            cur = nxt
        return "found", cur, visited

    # --- permissions --------------------------------------------------------

    def bits(self, path: str) -> int:
        node = self.nodes[path]
        if node.uid == AGENT_UID:
            return 7
        if node.gid in AGENT_GIDS:
            return (node.mode >> 3) & 7
        return node.mode & 7

    def reachable(self, path: str) -> bool:
        parts = [p for p in path.split("/") if p]
        above = ["/"] + ["/" + "/".join(parts[:i]) for i in range(1, len(parts))]
        return all(
            self.nodes.get(a) is not None
            and self.nodes[a].kind == "dir"
            and self.bits(a) & 1
            for a in above
        )

    def can_unlink(self, path: str) -> bool:
        if path == "/" or path not in self.nodes:
            return False
        parent = posixpath.dirname(path)
        if not self.reachable(path) or self.bits(parent) & 3 != 3:
            return False
        pnode = self.nodes[parent]
        if pnode.mode & 0o1000:
            return self.nodes[path].uid == AGENT_UID or pnode.uid == AGENT_UID
        return True

    def can_create(self, path: str) -> bool:
        parent = posixpath.dirname(path)
        return (
            parent in self.nodes
            and self.nodes[parent].kind == "dir"
            and self.reachable(path)
            and self.bits(parent) & 3 == 3
        )

    def subtree(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return [p for p in self.nodes if p == path or p.startswith(prefix)]

    # --- the reference ------------------------------------------------------

    def reference_write_open(self, path: str) -> bool:
        """Whether the agent user can change what root finds at ``path``."""
        end, real, visited = self.resolve(path)
        if end == "unresolved":
            return True
        # Any entry root passes that the agent user can rename or unlink, it
        # can also recreate with content of its own.
        if any(self.can_unlink(entry) for entry in visited):
            return True
        if end == "absent":
            return real is not None and self.can_create(real)
        assert real is not None
        node = self.nodes[real]
        if not self.reachable(real):
            return False
        if node.kind == "file":
            return bool(self.bits(real) & 2)
        # A folder: its own entries, or anything below it the agent can change.
        if self.bits(real) & 3 == 3:
            return True
        for inner in self.subtree(real):
            if inner == real:
                continue
            if self.can_unlink(inner):
                return True
            if (
                self.nodes[inner].kind == "file"
                and self.reachable(inner)
                and (self.bits(inner) & 2)
            ):
                return True
        return False

    def reference_read_open(self, path: str) -> bool:
        """Whether the agent user can read what root finds at ``path``."""
        end, real, _ = self.resolve(path)
        if end == "unresolved":
            return True
        if end == "absent":
            return False
        assert real is not None
        if not self.reachable(real):
            return False
        if self.nodes[real].kind == "file":
            return bool(self.bits(real) & 4)
        if self.bits(real) & 4:
            return True  # its names can be listed
        return any(
            self.reachable(inner) and self.bits(inner) & 4
            for inner in self.subtree(real)
            if inner != real
        )

    def final_kind(self, path: str) -> str:
        end, real, _ = self.resolve(path)
        if end != "found" or real is None:
            return end
        return self.nodes[real].kind

    # --- what the probe prints ----------------------------------------------

    def probe_output(self, paths: list[str]) -> str:
        lines = [f"U\t{AGENT_UID}\t{' '.join(str(g) for g in sorted(AGENT_GIDS))}"]
        for path in paths:
            lines.append(f"P\t{hexpath(path)}")
            end, real, visited = self.resolve(path)
            for entry in visited:
                node = self.nodes[entry]
                kind = "L" if node.kind == "link" else "S"
                facts = f"{node.uid} {node.gid} {node.mode:o}"
                lines.append(f"{kind}\t{facts}\t{_ls(node)}\t{hexpath(entry)}")
            end_kind = {"found": "F", "absent": "A", "unresolved": "X"}[end]
            lines.append(f"{end_kind}\t{hexpath(real or '/')}")
        return "\n".join(lines) + "\n"


def hexpath(path: str) -> str:
    """A pathname field as the probe prints it: hex of the raw bytes."""
    return path.encode("utf-8", "surrogateescape").hex()


def encode_records(text: str) -> str:
    """Plain-path probe records (as hand-written fixtures spell them) in the
    probe's real format: the path field of P, S, L, F, A and X records as hex."""
    out = []
    for line in text.split("\n"):
        kind, _, rest = line.partition("\t")
        if kind == "P" or kind in {"F", "A", "X"}:
            line = f"{kind}\t{hexpath(rest)}"
        elif kind in {"S", "L"}:
            fields = rest.split("\t")
            if len(fields) >= 3:
                path = "\t".join(fields[2:])
                line = "\t".join([kind, fields[0], fields[1], hexpath(path)])
            else:
                line = f"{kind}\t{hexpath(fields[-1])}"
        out.append(line)
    return "\n".join(out)


def _ls(node: Node) -> str:
    """The first field of ``ls -ldn`` for a node (no ACLs in the model)."""
    kind = {"dir": "d", "file": "-", "link": "l"}[node.kind]
    if node.kind == "link":
        return "lrwxrwxrwx"
    out = kind
    for shift in (6, 3, 0):
        bits = (node.mode >> shift) & 7
        out += ("r" if bits & 4 else "-") + ("w" if bits & 2 else "-")
        x = bool(bits & 1)
        if shift == 0 and node.mode & 0o1000 and node.kind == "dir":
            out += "t" if x else "T"
        else:
            out += "x" if x else "-"
    return out


# Plain names and names a shell or a line-based format can lose: trailing and
# embedded newlines, a tab, spaces, a backslash, a leading dash, and a
# non-UTF-8 byte (0xff, spelt with surrogateescape).
_NAMES = (
    "a",
    "b",
    "c",
    "d",
    "data\n",
    "\nx",
    "e f",
    "g\\h",
    "-i",
    "j\udcff",
    "k\tl",
    " ",
)
_MODES = (
    0o700,
    0o711,
    0o744,
    0o755,
    0o770,
    0o777,
    0o1777,
    0o1770,
    0o622,
    0o644,
    0o666,
    0o600,
)
_OWNERS = (0, 0, 0, AGENT_UID, OTHER_UID)
_GROUPS = (0, 0, AGENT_UID, OTHER_UID)


def random_tree(rng: random.Random, *, size: int = 9) -> Tree:
    """A small tree of folders, files and links under a root-owned 0755 root."""
    tree = Tree({"/": Node("dir", 0, 0, 0o755, ident=0)})
    for ident in range(1, size + 1):
        folders = [p for p, n in tree.nodes.items() if n.kind == "dir"]
        parent = rng.choice(folders)
        name = rng.choice(_NAMES)
        path = posixpath.join(parent, name)
        if path in tree.nodes or path.count("/") > 3:
            continue
        kind = rng.choices(("dir", "file", "link"), weights=(5, 2, 3))[0]
        node = Node(
            kind,
            uid=rng.choice(_OWNERS),
            gid=rng.choice(_GROUPS),
            mode=rng.choice(_MODES) & (0o1777 if kind == "dir" else 0o777),
            ident=ident,
        )
        if kind == "link":
            node.mode = 0o777
            hops = rng.randint(1, 3)
            parts = [rng.choice(("..", *_NAMES)) for _ in range(hops)]
            target = "/".join(parts)
            node.target = ("/" + target) if rng.random() < 0.5 else target
        tree.nodes[path] = node
    return tree


def random_case(rng: random.Random) -> tuple[Tree, str]:
    tree = random_tree(rng, size=rng.randint(3, 11))
    candidates = [p for p in tree.nodes if p != "/"]
    if not candidates or rng.random() < 0.1:
        path = "/" + "/".join(rng.choice(_NAMES) for _ in range(rng.randint(1, 3)))
    else:
        path = rng.choice(candidates)
    return tree, path
