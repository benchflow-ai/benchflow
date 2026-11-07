"""Admit artifact links without translating their sandbox meaning."""


def plain_link_targets(links: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """Keep original text only for plain relative paths through no other link.

    Split the raw text, never a normalized Path: empty/dot/parent components
    and absolute paths are outside this representation. Every target prefix,
    including the leaf, must be link-free in the complete source inventory.
    This also refuses chains through links that are themselves refused. Plain
    dangling paths keep their missing target; no host resolution is involved.
    """
    admitted: dict[str, str] = {}
    for name, text in links.items():
        parts = text.split("/")
        if "\0" in text or any(part in {"", ".", ".."} for part in parts):
            continue
        target = name.split("/")[:-1] + parts
        if any("/".join(target[:end]) in links for end in range(1, len(target) + 1)):
            continue
        admitted[name] = text
    return admitted, links.keys() - admitted.keys()
