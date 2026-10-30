"""ASI-Bench to BenchFlow converter.

Downloads pinned seed31415 material from Hugging Face and GitHub, then
generates a self-contained BenchFlow task directory for each (task, level,
instance) triple.  Only seed31415 is supported; seed42 is always rejected.

Usage::

    python -m benchmarks.asi-bench.main \\
        --output-dir /tmp/asi-bench-tasks \\
        [--task-ids control.mpsc,numerical_analysis.levin] \\
        [--levels B1,B2] \\
        [--limit 5] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml

logger = logging.getLogger(__name__)

# ── pinned revisions ─────────────────────────────────────────────────────────
# Both must be immutable commit IDs; update the pair when source material changes.
ASI_REVISION = "f13175a89dc9b4873f6306a3d31e46927c38f1a9"
HF_REVISION = "0fa14219cafdbab634d8b3cfbce238a8735a214f"

SUPPORTED_SEED = 31415
PROMPT_LEVELS = ("B1", "B2", "B3", "B4")

HF_REPO = "Apexintelligence-AI/ASI-Bench-seed31415"
GITHUB_REPO = "apexin-ai/ASI-Bench"

# Tasks skipped at conversion time with explicit reasons.
_SKIP_SCORER = {"llm_judge", "multimodal"}
_SKIP_RUNNER = {"submission_sandbox"}

_SCRIPT_DIR = Path(__file__).resolve().parent


def _validate_source_revisions() -> None:
    """Reject moving source refs before conversion or download can begin."""
    for name, revision in (
        ("ASI_REVISION", ASI_REVISION),
        ("HF_REVISION", HF_REVISION),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"{name} must be a 40-character commit SHA")


# ── public data classes ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ASIBenchSource:
    """A fully materialized, pinned ASI-Bench source checkout."""
    source_dir: Path
    asi_revision: str
    hf_revision: str


@dataclass(frozen=True)
class ASIBenchInstance:
    """One materialized seed31415 instance selected for conversion."""
    task_id: str
    instance_id: str
    prompt_level: str
    seed: int
    instance_dir: Path
    task_bundle_dir: Path


# ── validation ────────────────────────────────────────────────────────────────

def validate_instance(instance: ASIBenchInstance) -> None:
    """Validate the non-negotiable public conversion boundary."""
    if instance.seed != SUPPORTED_SEED:
        raise ValueError(
            f"ASI-Bench BenchFlow integration only supports seed {SUPPORTED_SEED}; "
            f"got {instance.seed}"
        )
    if instance.prompt_level not in PROMPT_LEVELS:
        raise ValueError(
            f"prompt_level must be one of {PROMPT_LEVELS}; "
            f"got {instance.prompt_level!r}"
        )
    if not instance.instance_id:
        raise ValueError("instance_id must not be empty")
    if not instance.task_id:
        raise ValueError("task_id must not be empty")


def _validate_task_id(task_id: str) -> tuple[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", task_id):
        raise ValueError(f"task_id must be <domain>.<name>; got {task_id!r}")
    domain, name = task_id.split(".", 1)
    return domain, name


def _safe_path(value: str) -> Path:
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts) or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe source path: {value!r}")
    return Path(*parts)


def _copy_instance_data(source: Path, destination: Path) -> None:
    """Copy materialized instance inputs without following links or special files."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Instance data is not a real directory: {source}")
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise ValueError(f"Instance data contains a symlink: {relative}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise ValueError(f"Instance data contains unsupported input: {relative}")


def _checked_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source did not resolve to an immutable Git commit")
    return value


# ── download helpers ──────────────────────────────────────────────────────────

class _Budget:
    _MAX_FILES = 10_000
    _MAX_FILE_BYTES = 256 * 1024 * 1024
    _MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024

    def __init__(self) -> None:
        self.total = 0
        self.paths: set[str] = set()

    def add(self, path: str, size: int) -> None:
        _safe_path(path)
        key = path.casefold()
        if key in self.paths:
            raise ValueError(f"duplicate or case-colliding path: {path}")
        if size < 0 or size > self._MAX_FILE_BYTES:
            raise ValueError(f"file size exceeds acquisition limit: {path}")
        self.paths.add(key)
        self.total += size
        if len(self.paths) > self._MAX_FILES or self.total > self._MAX_TOTAL_BYTES:
            raise ValueError("asset acquisition limit exceeded")


def _download_instance(
    task_id: str,
    destination: Path,
    hf_revision: str,
    hf_cache_dir: Path | None,
) -> str:
    """Download exactly one published seed31415 instance from HF."""
    try:
        from huggingface_hub import HfApi, RepoFile, hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required; install with: pip install huggingface-hub"
        ) from exc

    token = os.environ.get("HF_TOKEN") or False
    api = HfApi(token=token)
    sha = _checked_sha(api.dataset_info(HF_REPO, revision=hf_revision).sha)
    prefix = f"tasks/{task_id}__seed31415"
    budget = _Budget()
    for entry in api.list_repo_tree(
        HF_REPO,
        path_in_repo=prefix,
        recursive=True,
        repo_type="dataset",
        revision=sha,
    ):
        if not isinstance(entry, RepoFile):
            continue
        relative = str(PurePosixPath(entry.path).relative_to(prefix))
        budget.add(relative, entry.size)
        cached = Path(
            hf_hub_download(
                HF_REPO,
                entry.path,
                repo_type="dataset",
                revision=sha,
                token=token,
                cache_dir=hf_cache_dir,
            )
        )
        target = destination / _safe_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)

    prompt_count = sum(p.is_file() for p in destination.glob("prompt_b[1-4].md"))
    if prompt_count != 4:
        raise ValueError(
            f"seed31415 instance prompt count mismatch: expected=4, actual={prompt_count}"
        )
    reference = destination / "reference"
    if not reference.is_dir() or not any(p.is_file() for p in reference.rglob("*")):
        raise ValueError("seed31415 instance is missing populated reference/")
    return sha


def _download_task_bundle(task_id: str, destination: Path, asi_revision: str) -> str:
    """Walk a pinned Git subtree and download every regular blob."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required; install with: pip install httpx"
        ) from exc

    domain, name = _validate_task_id(task_id)
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    budget = _Budget()

    with httpx.Client(
        base_url=f"https://api.github.com/repos/{GITHUB_REPO}/",
        headers=headers,
        timeout=60,
    ) as client:

        def _get(path: str) -> Any:
            for attempt in range(3):
                try:
                    response = client.get(path)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as error:
                    if attempt == 2 or error.response.status_code not in (429, 502, 503, 504):
                        raise
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                time.sleep(2 ** attempt)
            raise AssertionError("unreachable")

        def _tree(sha: str) -> list[dict]:
            payload = _get(f"git/trees/{sha}")
            if payload.get("truncated"):
                raise ValueError("GitHub returned an incomplete tree")
            return payload["tree"]

        commit = _get(f"commits/{quote(asi_revision, safe='')}")
        sha = _checked_sha(commit["sha"])
        tree_sha = commit["commit"]["tree"]["sha"]
        for part in ("tasks", domain, name):
            matches = [e for e in _tree(tree_sha) if e["path"] == part]
            if len(matches) != 1 or matches[0]["type"] != "tree":
                raise ValueError(f"task folder not found: {task_id}")
            tree_sha = matches[0]["sha"]

        pending: list[tuple[str, str]] = [("", tree_sha)]
        directories = 0
        while pending:
            prefix, current_sha = pending.pop()
            directories += 1
            if directories > budget._MAX_FILES:
                raise ValueError("too many task subdirectories")
            for entry in _tree(current_sha):
                relative = prefix + entry["path"]
                path = _safe_path(relative)
                if entry["type"] == "tree":
                    if len(path.parts) > 64:
                        raise ValueError("task directory nesting limit exceeded")
                    pending.append((relative + "/", entry["sha"]))
                    continue
                if entry["type"] != "blob" or entry["mode"] not in ("100644", "100755"):
                    raise ValueError(f"unsupported Git entry: {relative}")
                budget.add(relative, entry["size"])
                blob = _get(f"git/blobs/{entry['sha']}")
                if blob["encoding"] != "base64":
                    raise ValueError(f"unsupported blob encoding: {relative}")
                data = base64.b64decode("".join(blob["content"].split()), validate=True)
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if len(data) != entry["size"] or digest != entry["sha"]:
                    raise ValueError(f"Git blob integrity mismatch: {relative}")
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if entry["mode"] == "100755" else 0o644)

    return sha


# ── evaluator file materializer ───────────────────────────────────────────────

def _materialize_evaluator_files(ef_spec: dict, evaluator_dir: Path) -> None:
    """Copy evaluator framework files specified in evaluator_files.json."""
    if (ef_spec.get("upstream") or {}).get("revision") != ASI_REVISION:
        raise ValueError("evaluator_files.json revision differs from ASI_REVISION")
    asi_root_candidates: list[Path] = []
    asi_source = os.environ.get("ASI_BENCH_SOURCE")
    if asi_source:
        asi_root_candidates.append(Path(asi_source))
    # Look relative to benchmarks/asi-bench → repo root
    asi_root_candidates.append(_SCRIPT_DIR.parent.parent.parent)

    asi_root: Path | None = None
    for candidate in asi_root_candidates:
        if (candidate / "ai4sci_bench").is_dir():
            asi_root = candidate
            break

    entries = ef_spec.get("files") or []
    if not entries:
        raise ValueError("evaluator_files.json declares no files")
    if asi_root is None and any(e.get("kind", "copy") == "copy" for e in entries):
        raise FileNotFoundError(
            "Cannot locate an ASI-Bench checkout containing ai4sci_bench/; "
            "set ASI_BENCH_SOURCE to the repository root"
        )

    for file_entry in entries:
        dest_rel = file_entry.get("path", "")
        if not dest_rel:
            continue
        dest = evaluator_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        kind = file_entry.get("kind", "copy")
        if kind == "initializer":
            dest.write_text(
                "# Minimal evaluator init — no harness, judge, or orchestrator imports\n",
                encoding="utf-8",
            )
        elif kind in ("copy", "runtime_adapter"):
            src_rel = file_entry.get("source") or file_entry.get("template")
            if not src_rel:
                raise ValueError(f"evaluator file entry has no source: {dest_rel}")
            if kind == "runtime_adapter":
                src = _SCRIPT_DIR / "verifier_template" / Path(src_rel).name
            else:
                assert asi_root is not None  # guarded above
                src = asi_root / src_rel
            if not src.is_file():
                raise FileNotFoundError(
                    f"Evaluator source not found: {src} (declared as {dest_rel}). "
                    "The verifier cannot import ai4sci_bench without it."
                )
            if kind == "copy":
                expected = file_entry.get("source_sha256")
                actual = hashlib.sha256(src.read_bytes()).hexdigest()
                if not expected or actual != expected:
                    raise ValueError(f"Evaluator source SHA-256 mismatch: {src_rel}")
            shutil.copy2(src, dest)


def _materialize_task_helpers(
    ef_spec: dict, task_id: str, bundle_dir: Path, evaluator_dir: Path
) -> None:
    """Copy only pinned public helpers needed by this task's custom scorer."""
    helpers = ef_spec.get("task_helper_files") or {}
    if not isinstance(helpers, dict):
        raise ValueError("task_helper_files must be a mapping")
    entries = helpers.get(task_id, [])
    if not isinstance(entries, list):
        raise ValueError(f"task_helper_files[{task_id!r}] must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid task helper entry for {task_id}")
        name = entry.get("path")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name):
            raise ValueError(f"unsafe task helper path: {name!r}")
        source = bundle_dir / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Pinned task helper missing or unsafe: {source}")
        expected = entry.get("source_sha256")
        if not isinstance(expected, str) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Task helper SHA-256 mismatch: {task_id}/{name}")
        destination = evaluator_dir / name
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Task helper would overwrite evaluator file: {name}")
        shutil.copy2(source, destination)


# ── task skip logic ───────────────────────────────────────────────────────────

def _should_skip(task_meta: dict, task_eval: dict | None) -> tuple[bool, str]:
    """Return (skip, reason) for tasks we cannot support."""
    if task_meta.get("difficulty", {}).get("requires_network"):
        return True, "requires_network=true; network allowlist not yet configured"

    evaluation = (task_eval or {}).get("evaluation") or {}
    for section in ("gates", "scoring"):
        for entry in evaluation.get(section) or []:
            scorer = entry.get("scorer", "")
            if scorer in _SKIP_SCORER:
                return True, f"unsupported scorer: {scorer}"

    # submission_sandbox runner
    runner = task_meta.get("runner") or {}
    runner_eval = (task_eval or {}).get("runner") or {}
    if runner.get("type") in _SKIP_RUNNER or runner_eval.get("type") in _SKIP_RUNNER:
        return True, "unsupported runner type: submission_sandbox"

    # runner.task_image (cmos-style)
    if runner.get("task_image"):
        return True, "runner.task_image tasks require a pre-built task image"

    return False, ""


# ── task directory generation ─────────────────────────────────────────────────

_PY_VERSION_RE = re.compile(r">=\s*(\d+)\.(\d+)")


def _pick_python_image(python_spec: str | None) -> str:
    if python_spec is None:
        return "python:3.11-slim"
    m = _PY_VERSION_RE.search(str(python_spec))
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if (major, minor) >= (3, 12):
            return f"python:{major}.{minor}-slim"
    return "python:3.11-slim"


_SAFE_PKG_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:(?:===|==|~=|>=|<=|!=|>|<)[A-Za-z0-9.*+!_-]+"
    r"(?:,[A-Za-z0-9.*+!_-]+(?:[._-][A-Za-z0-9.*+!_-]+)*)*)*"
)


def _build_dockerfile(
    task_meta: dict, task_eval: dict | None = None, *, custom_scorer: bool = False
) -> str:
    runtime = task_meta.get("runtime") or {}
    python_image = _pick_python_image(runtime.get("python"))
    packages: list[str] = list(runtime.get("packages") or [])
    for pkg in packages:
        if not _SAFE_PKG_RE.fullmatch(pkg):
            raise ValueError(f"unsafe package spec in runtime.packages: {pkg!r}")

    base_pkgs = ["numpy", "PyYAML", "packaging"]
    all_pkgs = list(dict.fromkeys(base_pkgs + packages))
    eval_runtime = (task_eval or {}).get("evaluation") or {}
    eval_pkgs = list(dict.fromkeys(base_pkgs + ["scipy"] + (
        packages if eval_runtime.get("runtime") == "task" or custom_scorer else []
    )))

    lines = [
        f"FROM {python_image}",
        "",
        "# Evaluation framework venv (root-owned, not on PATH)",
        "RUN python -m venv /opt/asi-eval && \\",
        "    /opt/asi-eval/bin/pip install --no-cache-dir \\",
        "        " + " ".join(shlex.quote(pkg) for pkg in eval_pkgs),
        "",
        "# Task runtime packages (agent environment)",
    ]
    if all_pkgs:
        pkgs_joined = " \\\n        ".join(shlex.quote(pkg) for pkg in all_pkgs)
        lines += [
            "RUN pip install --no-cache-dir \\",
            f"        {pkgs_joined}",
        ]
    else:
        lines.append("# (no additional task packages)")

    lines += [
        "",
        "WORKDIR /workspace",
        "COPY inputs/ /workspace/",
        'CMD ["sleep", "infinity"]',
    ]
    return "\n".join(lines) + "\n"


def _render_task_md(
    prompt_text: str,
    task_id: str,
    instance_id: str,
    prompt_level: str,
    task_meta: dict,
    asi_revision: str,
    hf_revision: str,
) -> str:
    requires_network = task_meta.get("difficulty", {}).get("requires_network", False)
    timeout_sec = int(task_meta.get("timeout_seconds") or 3600)
    verifier_timeout_sec = max(900, timeout_sec // 4)

    # BenchFlow attaches NET_ADMIN (needed for the UID egress firewall) only when
    # network_mode is "denylist".  "no-network" skips NET_ADMIN, so the container-
    # internal iptables call fails with "Permission denied".  We use "denylist" with
    # a catch-all blocked_hosts entry to achieve the same full-block semantics while
    # giving the container the capability it needs to enforce the firewall.
    # Tasks that genuinely require network access keep "public" unchanged.
    if requires_network:
        network_mode = "public"
        sandbox_extra: dict[str, Any] = {}
    else:
        network_mode = "denylist"
        # Block every external host.  BenchFlow still allows the loopback LiteLLM
        # proxy (the model gateway) regardless of the denylist.
        # blocked_hosts must be non-empty for denylist mode; use a placeholder
        # hostname.  The UID firewall (iptables owner rule) is the actual
        # enforcement layer and blocks ALL external traffic from the agent user
        # regardless of this list.
        sandbox_extra = {"blocked_hosts": ["placeholder.invalid"]}

    frontmatter: dict[str, Any] = {
        "schema_version": "1.0",
        "metadata": {
            "benchmark": "ASI-Bench",
            "task_id": task_id,
            "instance_id": instance_id,
            "prompt_level": prompt_level,
            "seed": SUPPORTED_SEED,
            "official": False,
            "asi_revision": asi_revision,
            "hf_revision": hf_revision,
        },
        "agent": {"timeout_sec": timeout_sec},
        "verifier": {"timeout_sec": verifier_timeout_sec},
        "sandbox": {
            "cpus": 2,
            "memory_mb": 4096,
            "workdir": "/workspace",
            "network_mode": network_mode,
            **sandbox_extra,
        },
    }
    fm_text = yaml.dump(
        frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    return f"---\n{fm_text}---\n\n{prompt_text.strip()}\n"


def _safe_rel_output(value: str) -> Path:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe output path: {value!r}")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe output path: {value!r}")
    return Path(value)


# ── main convert function ─────────────────────────────────────────────────────

def convert(
    source_instance: ASIBenchInstance,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert one materialized seed31415 instance into a BenchFlow task directory."""
    validate_instance(source_instance)

    _validate_source_revisions()

    instance_dir = source_instance.instance_dir
    bundle_dir = source_instance.task_bundle_dir
    task_id = source_instance.task_id
    level_lower = source_instance.prompt_level.lower()

    if not instance_dir.is_dir():
        raise ValueError(f"instance_dir does not exist: {instance_dir}")
    if not bundle_dir.is_dir():
        raise ValueError(f"task_bundle_dir does not exist: {bundle_dir}")

    meta_path = bundle_dir / "task_meta.yaml"
    eval_path = bundle_dir / "task_eval.yaml"
    if not meta_path.is_file():
        raise ValueError(f"task_meta.yaml not found in {bundle_dir}")

    with meta_path.open(encoding="utf-8") as fh:
        task_meta: dict = yaml.safe_load(fh) or {}
    task_eval: dict | None = None
    if eval_path.is_file():
        with eval_path.open(encoding="utf-8") as fh:
            task_eval = yaml.safe_load(fh) or {}

    if task_meta.get("id") != task_id:
        raise ValueError(
            f"task_meta.yaml id mismatch: expected {task_id!r}, "
            f"got {task_meta.get('id')!r}"
        )
    if task_eval and task_eval.get("task_id") not in (None, task_id):
        raise ValueError(
            f"task_eval.yaml task_id mismatch: expected {task_id!r}, "
            f"got {task_eval.get('task_id')!r}"
        )

    if source_instance.seed != SUPPORTED_SEED:
        raise ValueError(f"Only seed {SUPPORTED_SEED} supported; got {source_instance.seed}")
    instance_meta_path = instance_dir / "instance_meta.json"
    if instance_meta_path.is_file():
        with instance_meta_path.open(encoding="utf-8") as fh:
            instance_meta_data = json.load(fh)
        meta_iid = instance_meta_data.get("instance_id")
        if meta_iid is not None and meta_iid != source_instance.instance_id:
            raise ValueError(
                f"instance_meta.json instance_id mismatch: "
                f"expected {source_instance.instance_id!r}, got {meta_iid!r}"
            )

    skip, reason = _should_skip(task_meta, task_eval)
    if skip:
        raise ValueError(f"Task {task_id} skipped: {reason}")

    prompt_file = instance_dir / f"prompt_{level_lower}.md"
    if not prompt_file.is_file():
        raise ValueError(f"Prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8")

    reference_dir = instance_dir / "reference"
    if not reference_dir.is_dir() or not any(reference_dir.rglob("*")):
        raise ValueError(f"Reference directory missing or empty: {reference_dir}")

    domain, name = _validate_task_id(task_id)
    short_iid = source_instance.instance_id.replace(f"{task_id}__", "")[:16]
    task_dir_name = (
        f"asi-{domain}-{name}-{source_instance.prompt_level.lower()}-{short_iid}"
    )
    task_dir = output_dir / task_dir_name

    if task_dir.exists():
        if not overwrite:
            logger.info("Skipping existing task %s", task_dir)
            return task_dir
        shutil.rmtree(task_dir)

    staging = Path(
        tempfile.mkdtemp(prefix=f".asi-bench-{task_dir_name}-", dir=output_dir)
    )
    try:
        _generate_task_dir(
            staging=staging,
            task_id=task_id,
            instance_id=source_instance.instance_id,
            prompt_level=source_instance.prompt_level,
            task_meta=task_meta,
            task_eval=task_eval,
            prompt_text=prompt_text,
            instance_dir=instance_dir,
            bundle_dir=bundle_dir,
            reference_dir=reference_dir,
            asi_revision=ASI_REVISION,
            hf_revision=HF_REVISION,
        )
        staging.rename(task_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info("Generated task %s at %s", task_id, task_dir)
    return task_dir


def _generate_task_dir(
    *,
    staging: Path,
    task_id: str,
    instance_id: str,
    prompt_level: str,
    task_meta: dict,
    task_eval: dict | None,
    prompt_text: str,
    instance_dir: Path,
    bundle_dir: Path,
    reference_dir: Path,
    asi_revision: str,
    hf_revision: str,
) -> None:
    # 1. task.md
    task_md = _render_task_md(
        prompt_text=prompt_text,
        task_id=task_id,
        instance_id=instance_id,
        prompt_level=prompt_level,
        task_meta=task_meta,
        asi_revision=asi_revision,
        hf_revision=hf_revision,
    )
    (staging / "task.md").write_text(task_md, encoding="utf-8")

    # 2. environment/Dockerfile
    env_dir = staging / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text(
        _build_dockerfile(
            task_meta, task_eval,
            custom_scorer=(bundle_dir / "custom_scorer.py").is_file(),
        ),
        encoding="utf-8",
    )

    # 3. Complete materialized inputs for the agent and an immutable verifier copy.
    # Input names may contain expansion templates, so copying literal declarations
    # can omit files that the original ASI scorer sees.
    inputs_dir = env_dir / "inputs"
    inputs_dir.mkdir()
    input_files = (task_meta.get("input") or {}).get("files") or []
    instance_data = instance_dir / "data"
    if input_files and not instance_data.is_dir():
        raise ValueError(f"Required instance data directory is missing: {instance_data}")
    if instance_data.exists() or instance_data.is_symlink():
        _copy_instance_data(instance_data, inputs_dir / "data")

    # 4. verifier/ — template files
    verifier_dir = staging / "verifier"
    verifier_dir.mkdir()
    if instance_data.exists() or instance_data.is_symlink():
        _copy_instance_data(instance_data, verifier_dir / "instance_data" / "data")
        # ASI custom scorers may read inputs beside reference/ via
        # ref_dir.parent / "data", in addition to the staged prediction data/.
        _copy_instance_data(instance_data, verifier_dir / "data")
    verifier_template_dir = _SCRIPT_DIR / "verifier_template"
    for tmpl in ("test.sh", "score_entry.py"):
        src = verifier_template_dir / tmpl
        if not src.is_file():
            raise FileNotFoundError(f"Verifier template not found: {src}")
        dst = verifier_dir / tmpl
        shutil.copy2(src, dst)
        if tmpl == "test.sh":
            dst.chmod(0o755)

    # 5. verifier/evaluator/ — ASI scoring framework
    evaluator_dir = verifier_dir / "evaluator"
    evaluator_dir.mkdir()

    eval_yaml_src = bundle_dir / "task_eval.yaml"
    if eval_yaml_src.is_file():
        shutil.copy2(eval_yaml_src, evaluator_dir / "task_eval.yaml")

    custom_scorer_src = bundle_dir / "custom_scorer.py"
    if custom_scorer_src.is_file():
        shutil.copy2(custom_scorer_src, evaluator_dir / "custom_scorer.py")

    ef_path = _SCRIPT_DIR / "evaluator_files.json"
    if not ef_path.is_file():
        raise FileNotFoundError(
            f"evaluator_files.json not found at {ef_path}; the generated verifier "
            "would lack ai4sci_bench and fail at scoring time"
        )
    with ef_path.open(encoding="utf-8") as fh:
        ef_spec = json.load(fh)
    _materialize_evaluator_files(ef_spec, evaluator_dir)
    _materialize_task_helpers(ef_spec, task_id, bundle_dir, evaluator_dir)

    # 6. verifier/reference/
    ref_dst = verifier_dir / "reference"
    shutil.copytree(reference_dir, ref_dst, symlinks=False)

    # 7. verifier/instance.json
    instance_meta_path = instance_dir / "instance_meta.json"
    params: dict = {}
    if instance_meta_path.is_file():
        with instance_meta_path.open(encoding="utf-8") as fh:
            imd = json.load(fh)
        params = dict(imd.get("params_used") or {})

    instance_json = {
        "schema_version": 1,
        "task_id": task_id,
        "instance_id": instance_id,
        "prompt_level": prompt_level.upper(),
        "seed": SUPPORTED_SEED,
        "official": False,
        "asi_revision": asi_revision,
        "hf_revision": hf_revision,
        "parameters": params,
        "requires_instance_data": bool(input_files),
    }
    (verifier_dir / "instance.json").write_text(
        json.dumps(instance_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 8. verifier/output_specs.json
    output_specs = []
    for item in (task_meta.get("output") or {}).get("files") or []:
        name = item.get("name", "")
        if not name:
            continue
        try:
            _safe_rel_output(name)
        except ValueError:
            logger.warning("Skipping unsafe output spec: %r", name)
            continue
        output_specs.append({"name": name, "type": item.get("type", "data")})
    (verifier_dir / "output_specs.json").write_text(
        json.dumps(output_specs, indent=2) + "\n", encoding="utf-8"
    )

    # 9. verifier/score_divisor.json
    evaluation = (task_eval or {}).get("evaluation") or {}
    score_divisor = evaluation.get("score_divisor", 1.0)
    (verifier_dir / "score_divisor.json").write_text(
        json.dumps({"score_divisor": score_divisor}) + "\n", encoding="utf-8"
    )


# ── convert_all ───────────────────────────────────────────────────────────────

def convert_all(
    source_dir: Path | None,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    levels: list[str] | None = None,
    hf_cache_dir: Path | None = None,
) -> list[Path]:
    """Download seed31415 material and convert to BenchFlow task directories.

    Each (task_id, level) pair becomes one task directory.  Skipped tasks
    (llm_judge, submission_sandbox, requires_network) are logged but not raised.

    Args:
        source_dir: Pre-downloaded source root.  When None the function
            downloads directly from HF and GitHub using ASI_REVISION /
            HF_REVISION.
        output_dir: Where to write generated task directories.
        overwrite: Replace existing task directories.
        limit: Cap total task directories generated (for smoke tests).
        task_ids: Explicit list of ``domain.name`` task IDs; default is all tasks
            found in the source directory.
        levels: Subset of PROMPT_LEVELS to generate (default: all four).
        hf_cache_dir: Override the HuggingFace local cache directory.

    Returns:
        List of generated task directory paths.
    """
    _validate_source_revisions()

    selected_levels = [lvl.upper() for lvl in (levels or list(PROMPT_LEVELS))]
    for lvl in selected_levels:
        if lvl not in PROMPT_LEVELS:
            raise ValueError(f"Unknown prompt level: {lvl!r}; must be one of {PROMPT_LEVELS}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tasks from source_dir or download them
    if source_dir is not None:
        task_bundle_root, instance_root = _locate_source_roots(source_dir)
        available_task_ids = _discover_task_ids(task_bundle_root)
    else:
        # Download mode: download each requested task_id
        if task_ids is None:
            raise ValueError(
                "--task-ids is required when --source-dir is not given; "
                "provide a comma-separated list of domain.name task IDs"
            )
        # Create a temp download root
        download_root = output_dir / ".asi-bench-downloads"
        download_root.mkdir(exist_ok=True)
        task_bundle_root = download_root / "task_bundle" / "tasks"
        instance_root = download_root / "instance"
        for tid in task_ids:
            _download_task_assets(
                tid, download_root, HF_REVISION, ASI_REVISION, hf_cache_dir
            )
        available_task_ids = task_ids

    # Filter by requested task_ids
    if task_ids is not None:
        requested = set(task_ids)
        available_task_ids = [t for t in available_task_ids if t in requested]

    generated: list[Path] = []
    skipped: list[tuple[str, str]] = []

    for task_id in available_task_ids:
        if limit is not None and len(generated) >= limit:
            break

        domain, name = _validate_task_id(task_id)
        bundle_dir = task_bundle_root / domain / name
        instance_id = f"{task_id}__seed31415"
        instance_dir = instance_root / instance_id

        if not bundle_dir.is_dir():
            logger.warning("Task bundle not found, skipping: %s", bundle_dir)
            continue
        if not instance_dir.is_dir():
            logger.warning("Instance dir not found, skipping: %s", instance_dir)
            continue

        for level in selected_levels:
            if limit is not None and len(generated) >= limit:
                break

            inst = ASIBenchInstance(
                task_id=task_id,
                instance_id=instance_id,
                prompt_level=level,
                seed=SUPPORTED_SEED,
                instance_dir=instance_dir,
                task_bundle_dir=bundle_dir,
            )
            try:
                task_dir = convert(inst, output_dir, overwrite=overwrite)
                generated.append(task_dir)
            except ValueError as exc:
                reason = str(exc)
                skipped.append((f"{task_id}/{level}", reason))
                logger.info("Skipped %s/%s: %s", task_id, level, reason)
            except Exception as exc:
                logger.error("Error converting %s/%s: %s", task_id, level, exc)
                raise

    if skipped:
        logger.info(
            "Skipped %d task/level combinations:\n%s",
            len(skipped),
            "\n".join(f"  {tid}: {reason}" for tid, reason in skipped),
        )

    logger.info("Generated %d BenchFlow tasks in %s", len(generated), output_dir)
    return generated


def _locate_source_roots(source_dir: Path) -> tuple[Path, Path]:
    """Find task_bundle/tasks/ and instance/ inside a source directory."""
    # Layout from sources.py: source_dir/task_bundle/tasks/<domain>/<name>/
    #                          source_dir/instance/<task_id>__seed31415/
    task_bundle_root = source_dir / "task_bundle" / "tasks"
    instance_root = source_dir / "instance"
    if not task_bundle_root.is_dir():
        # Also accept a plain ASI-Bench checkout: tasks/<domain>/<name>/
        alt = source_dir / "tasks"
        if alt.is_dir():
            task_bundle_root = alt
        else:
            raise FileNotFoundError(
                f"Cannot find task bundle root in {source_dir}; "
                "expected task_bundle/tasks/ or tasks/"
            )
    if not instance_root.is_dir():
        raise FileNotFoundError(
            f"Cannot find instance root in {source_dir}; "
            "expected instance/ directory with seed31415 instances"
        )
    return task_bundle_root, instance_root


def _discover_task_ids(task_bundle_root: Path) -> list[str]:
    """Walk task_bundle_root and return all domain.name task IDs."""
    task_ids: list[str] = []
    for domain_dir in sorted(task_bundle_root.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
            continue
        for name_dir in sorted(domain_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            if (name_dir / "task_meta.yaml").is_file():
                task_ids.append(f"{domain_dir.name}.{name_dir.name}")
    return task_ids


def _download_task_assets(
    task_id: str,
    download_root: Path,
    hf_revision: str,
    asi_revision: str,
    hf_cache_dir: Path | None,
) -> None:
    """Download HF instance + GitHub task bundle for one task_id."""
    domain, name = _validate_task_id(task_id)
    instance_id = f"{task_id}__seed31415"
    instance_dir = download_root / "instance" / instance_id
    bundle_dir = download_root / "task_bundle" / "tasks" / domain / name

    _download_pinned_directory(
        instance_dir,
        hf_revision,
        lambda staging: _download_instance(task_id, staging, hf_revision, hf_cache_dir),
        required_files=tuple(f"prompt_b{level}.md" for level in range(1, 5)),
    )
    _download_pinned_directory(
        bundle_dir,
        asi_revision,
        lambda staging: _download_task_bundle(task_id, staging, asi_revision),
        required_files=("task_meta.yaml", "task_eval.yaml"),
    )


def _download_pinned_directory(
    destination: Path,
    revision: str,
    download: Callable[[Path], str],
    *,
    required_files: tuple[str, ...],
) -> None:
    """Publish only a complete download; never reuse a partial cache directory."""
    if destination.is_symlink():
        raise ValueError(f"Download destination is a symlink: {destination}")
    marker = destination / ".asibench-download-complete.json"
    if destination.is_dir() and marker.is_file():
        try:
            complete = json.loads(marker.read_text()) == {"revision": revision}
        except (OSError, ValueError):
            complete = False
        if complete and all((destination / name).is_file() for name in required_files):
            return

    logger.info("Downloading pinned assets: %s", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        actual_revision = download(staging)
        if actual_revision != revision:
            raise ValueError(
                f"Downloaded revision mismatch: expected {revision}, got {actual_revision}"
            )
        if not all((staging / name).is_file() for name in required_files):
            raise ValueError(f"Download missing required files: {destination}")
        (staging / marker.name).write_text(json.dumps({"revision": revision}) + "\n")
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise ValueError(
                    f"Download destination is not a directory: {destination}"
                )
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Convert pinned seed31415 ASI-Bench material to BenchFlow tasks.\n\n"
            "Requires either --source-dir (pre-downloaded material) or --task-ids "
            "(downloads on-the-fly from HF and GitHub)."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write generated BenchFlow task directories into."
    )
    parser.add_argument(
        "--source-dir", type=Path, default=None,
        help=(
            "Pre-downloaded ASI-Bench source root produced by sources.py acquire(). "
            "Expects task_bundle/tasks/ and instance/ sub-directories. "
            "Also accepts a plain ASI-Bench git checkout (tasks/ layout) combined "
            "with --instance-dir."
        ),
    )
    parser.add_argument(
        "--task-ids", default=None,
        help="Comma-separated domain.name task IDs to convert (default: all).",
    )
    parser.add_argument(
        "--levels", default=None,
        help="Comma-separated prompt levels to generate (default: B1,B2,B3,B4).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total task directories generated (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing task directories.",
    )
    parser.add_argument(
        "--hf-cache-dir", type=Path, default=None,
        help="Override the HuggingFace local cache directory.",
    )
    args = parser.parse_args()

    generated = convert_all(
        args.source_dir,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=_parse_csv(args.task_ids),
        levels=_parse_csv(args.levels),
        hf_cache_dir=args.hf_cache_dir,
    )
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
