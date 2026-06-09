#!/usr/bin/env python3
"""Regenerate network-map.yaml from a SkillsBench checkout.

    python generate_network_map.py /path/to/skillsbench

Clone first:  git clone --depth 1 https://github.com/benchflow-ai/skillsbench

For each task: parse task.toml, scan solution/ + environment/ (excluding skill-doc
markdown) + instruction.md for runtime hosts, map LLM clients to provider hosts,
exclude build/CDN/doc hosts. The `api` allowlists below are the curated result of
that scan with local mocks and agent-own-LLM hosts removed (see README.md).
"""
import re
import sys
import tomllib
from pathlib import Path

# Curated api-mode allowlists (agent-own-LLM hosts and local mocks excluded).
API_MAP = {
    "citation-check": ("data-api", ["api.crossref.org", "api.semanticscholar.org",
        "api.datacite.org", "eutils.ncbi.nlm.nih.gov", "export.arxiv.org", "doi.org"]),
    "find-topk-similiar-chemicals": ("data-api", ["pubchem.ncbi.nlm.nih.gov"]),
    "shock-analysis-supply": ("data-api", ["data.ecb.europa.eu"]),
    "scheduling-email-assistant": ("cloud-api", ["www.googleapis.com", "oauth2.googleapis.com"]),
    "pedestrian-traffic-counting": ("llm-provider", ["generativelanguage.googleapis.com"]),
    "video-tutorial-indexer": ("llm-provider", ["api.openai.com"]),
    "diff-transformer_impl": ("compute-backend", ["modal.com"]),
    "mhc-layer-impl": ("compute-backend", ["modal.com"]),
}


def read(p):
    try:
        return p.read_text(errors="ignore")
    except OSError:
        return ""


def skills_of(t):
    sk = t / "environment" / "skills"
    return [s.parent.name for s in sorted(sk.rglob("SKILL.md"))] if sk.is_dir() else []


def keywords(meta, skills):
    out = []
    for field in ("tags", "task_type", "skill_type"):
        for x in meta.get(field) or []:
            out.append(str(x).lower())
    out += [s.lower() for s in skills]
    seen, dedup = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup[:12]


def main(root: Path):
    lines = [
        "# SkillsBench network-mode + glossary map — generated, do not hand-edit.",
        "# Modes: disabled (offline; agent LLM egress still allowed) | api (allowlist, no web search)",
        "#        | enabled-blacklist (full internet minus denylist; SkillsBench source default only).",
        "# allowed_hosts are the concrete endpoints the reference solution contacts at runtime.",
        "tasks:",
    ]
    for base in [root / "tasks", root / "tasks-extra"]:
        for t in sorted(base.iterdir()) if base.is_dir() else []:
            toml_p = t / "task.toml"
            if not toml_p.exists():
                continue
            data = tomllib.loads(read(toml_p))
            meta = data.get("metadata", {})
            env = data.get("environment", {})
            declared = bool(env.get("allow_internet"))
            skills = skills_of(t)
            if t.name in API_MAP:
                subtype, hosts = API_MAP[t.name]
                mode = "api"
            else:
                subtype, hosts, mode = None, [], "disabled"
            lines.append(f"  {t.name}:")
            lines.append(f"    group: {base.name}")
            lines.append(f"    category: {meta.get('category')}")
            lines.append(f"    declared_allow_internet: {str(declared).lower()}")
            lines.append(f"    network_mode: {mode}")
            if subtype:
                lines.append(f"    api_subtype: {subtype}")
            if hosts:
                lines.append("    allowed_hosts:")
                lines += [f"      - {h}" for h in hosts]
            if declared and mode == "disabled":
                lines.append("    tightened_from: enabled  # declared full internet but contacts no runtime host")
            if skills:
                lines.append(f"    skills: [{', '.join(skills)}]")
            lines.append(f"    glossary_keywords: [{', '.join(keywords(meta, skills))}]")
    out = Path(__file__).with_name("network-map.yaml")
    out.write_text("\n".join(lines) + "\n")
    n = sum(1 for _ in re.finditer(r"network_mode:", "\n".join(lines)))
    print(f"wrote {out} ({n} tasks)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: generate_network_map.py /path/to/skillsbench")
    main(Path(sys.argv[1]).resolve())
