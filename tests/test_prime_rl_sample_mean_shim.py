"""BenchFlow's sample-mean loss shim for Prime-RL's SFT trainer.

``bench train run sft --loss-normalization sample_mean`` rewrites
prime_rl/trainer/sft/train.py as the trainer imports it. These tests run the
shim against verbatim excerpts of that file from prime-rl releases, kept in
tests/fixtures/prime_rl_sft/ with their upstream commit and line numbers.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import benchflow.training.backends.prime_rl as prime_rl
from benchflow.cli.main import app

FIXTURES = Path(__file__).parent / "fixtures" / "prime_rl_sft"
LEGACY = "prime-rl v0.5.0 to v0.7.0"
V010 = "prime-rl v0.10.0"
V010_MAIN = "prime-rl v0.10 main after #4237"
_SHIM_ENV_KEYS = (
    "BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS",
    "BENCHFLOW_PRIME_RL_PRETOKENIZED_SFT_DATA",
    "BENCHFLOW_PRIME_RL_STUB_FLASH_ATTN",
)

runner = CliRunner()


@pytest.fixture
def shim(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The generated sitecustomize, loaded without installing its import hooks."""
    for key in _SHIM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    namespace: dict[str, Any] = {"__name__": "benchflow_prime_rl_sft_shim"}
    source = prime_rl._PRIME_RL_SFT_COMPAT_SITE_CUSTOMIZE
    exec(compile(source, "sitecustomize.py", "exec"), namespace)
    return namespace


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _compute_loss(source: str) -> str:
    """The text of the ``compute_loss`` closure inside a train.py."""
    lines = source.splitlines(keepends=True)
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("    def compute_loss(")
    )
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end].startswith("        ")
    ):
        end += 1
    return "".join(lines[start:end])


def _fake_prime_rl(root: Path, train_source: str) -> Path:
    sft = root / "fake" / "prime_rl" / "trainer" / "sft"
    sft.mkdir(parents=True)
    for package in (sft.parent.parent, sft.parent, sft):
        (package / "__init__.py").write_text("", encoding="utf-8")
    train_py = sft / "train.py"
    train_py.write_text(train_source, encoding="utf-8")
    return train_py


def _import_train_py(root: Path) -> subprocess.CompletedProcess[str]:
    plan = prime_rl._write_prime_rl_sft_compat_shim(
        root, sample_mean=True, pretokenized_data=False, stub_flash_attn=False
    )
    env = os.environ.copy()
    env.update(plan.env)
    env["PYTHONPATH"] = os.pathsep.join((plan.env["PYTHONPATH"], str(root / "fake")))
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import prime_rl.trainer.sft.train as train; "
            "print(train.__benchflow_sample_mean_layout__)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sample_mean_shim_rewrites_prime_rl_0_7_exactly_as_before(shim) -> None:
    """prime-rl v0.5.0 to v0.7.0 keep the 0.9.0-rc rewrite byte for byte.

    Guards prime-rl v0.10 support against changing the loss for the
    versions the shim already supported: compute_loss must match the golden
    that the 0.9.0-rc shim (e118a8d3) produced. The one addition is a check,
    run before the model loads, that refuses prime-rl's cat-packing default.
    """
    source = _fixture("train_v0.7.0.txt")

    patched, layout = shim["_patch_sample_mean_loss"](source, "train.py", "0.7.0")

    assert layout == LEGACY
    golden = _fixture("train_v0.7.0.sample_mean_compute_loss.txt")
    assert _compute_loss(patched) == golden[golden.index("    def compute_loss(") :]
    assert (
        'def train(config: SFTConfig):\n    if config.data.pack_function != "stack":\n'
    ) in patched
    assert "# All-reduce sample counts and rescale gradients" in patched


@pytest.mark.parametrize(
    ("fixture", "layout", "offloading"),
    [
        (
            "train_v0.10.0.txt",
            V010,
            "with maybe_activation_offloading(config.model.ac_offloading):",
        ),
        (
            "train_main_14ef9140.txt",
            V010_MAIN,
            "with maybe_activation_offloading(config.model.ac_offloading, model):",
        ),
    ],
)
def test_sample_mean_shim_rewrites_prime_rl_0_10_loss(
    shim, fixture: str, layout: str, offloading: str
) -> None:
    """prime-rl v0.10 computes the SFT loss inside the LM head (prime-rl #3673).

    The shim refused every v0.10 train.py. It now asks the head
    for per-token logprobs, averages each packed sample at its seq_lens and
    returns the sample count. Nothing outside compute_loss changes, so the
    step loop still divides the gradients by the count compute_loss returns.
    """
    source = _fixture(fixture)

    patched, got = shim["_patch_sample_mean_loss"](source, "train.py", "0.10.0")

    assert got == layout
    loss = _compute_loss(patched)
    compile(textwrap.dedent(loss), fixture, "exec")
    assert 'loss_sum = out["loss"]' not in loss
    assert "labels=target_ids,\n                temperature=temperature," in loss
    assert "torch.split(row_loss, sample_lens)" in loss
    assert f"        {offloading}\n" in loss
    assert loss.endswith("        return loss_sum, sample_count\n")
    assert patched.replace(loss, "") == source.replace(_compute_loss(source), "")


def test_sample_mean_shim_refuses_prime_rl_0_9_naming_versions(shim) -> None:
    """A train.py of unverified layout is refused, with a reason.

    prime-rl v0.8 and v0.9 compute the loss in a layout the shim does not
    rewrite (since prime-rl #3044). The refusal names the file, the installed
    version, the supported versions and the first line that differs from the
    nearest known layout.
    """
    origin = "/opt/prime-rl/src/prime_rl/trainer/sft/train.py"

    with pytest.raises(RuntimeError) as excinfo:
        shim["_patch_sample_mean_loss"](_fixture("train_v0.9.0.txt"), origin, "0.9.0")

    message = str(excinfo.value)
    assert f"{origin} (prime-rl 0.9.0) matches no layout BenchFlow has verified" in (
        message
    )
    assert "Supported: prime-rl v0.5.0 to v0.7.0, and v0.10.0" in message
    assert (
        "Nearest: prime-rl v0.10.0, which differs at '# Labels without a "
        "temperature make the LM head return the summed cross-entropy directly.'"
    ) in message
    assert "Use --loss-normalization token_mean" in message


def test_sample_mean_shim_refuses_when_the_step_loop_stops_using_its_count(
    shim,
) -> None:
    """A known loss block alone is not enough to patch.

    The rewrite yields a per-sample mean only because the step loop divides
    the gradients by the count compute_loss returns. A train.py whose loop
    counts tokens some other way must be refused, not silently mis-scaled.
    """
    source = _fixture("train_v0.10.0.txt").replace(
        "                step_local_token_count += batch_token_count\n",
        '                step_local_token_count += micro_batch["loss_mask"].sum()\n',
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Nearest: prime-rl v0\.10\.0, which differs at "
            r"'step_local_token_count \+= batch_token_count'"
        ),
    ):
        shim["_patch_sample_mean_loss"](source, "train.py", "0.10.0")


def test_sample_mean_shim_refuses_a_loss_block_that_occurs_twice(shim) -> None:
    """Each anchor must occur once, so no copy stays unpatched."""
    anchor = shim["_SAMPLE_MEAN_LAYOUTS"][1]["patches"][0][0]
    source = _fixture("train_v0.10.0.txt") + "\n" + anchor

    with pytest.raises(RuntimeError, match="occurs 2 times"):
        shim["_patch_sample_mean_loss"](source, "train.py", "0.10.0")


@pytest.mark.parametrize(
    ("fixture", "layout"),
    [
        ("train_v0.7.0.txt", LEGACY),
        ("train_v0.10.0.txt", V010),
        ("train_main_14ef9140.txt", V010_MAIN),
    ],
)
def test_sample_mean_shim_patches_prime_rl_when_the_trainer_imports_it(
    tmp_path: Path, fixture: str, layout: str
) -> None:
    """End to end: the trainer's own import of
    prime_rl.trainer.sft.train is rewritten, and its stderr names the layout
    and the digest of the source that was patched."""
    train_py = _fake_prime_rl(tmp_path, _fixture(fixture))

    completed = _import_train_py(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == layout
    digest = hashlib.sha256(train_py.read_bytes()).hexdigest()[:12]
    assert (
        f"train.py as {layout} (prime-rl version unknown, train.py sha256 {digest})\n"
    ) in completed.stderr


def test_sample_mean_shim_stops_the_trainer_on_an_unknown_layout(
    tmp_path: Path,
) -> None:
    """An unverified train.py fails at import, before any GPU work."""
    _fake_prime_rl(tmp_path, _fixture("train_v0.9.0.txt"))

    completed = _import_train_py(tmp_path)

    assert completed.returncode != 0
    assert "matches no layout BenchFlow has verified" in completed.stderr


class _OnCpu:
    """A batch tensor whose ``.to("cuda", ...)`` keeps it on the CPU."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return self.tensor


def _hand_cross_entropy(logits: list[float], target: int) -> float:
    peak = max(logits)
    log_z = peak + math.log(sum(math.exp(value - peak) for value in logits))
    return log_z - logits[target]


def test_sample_mean_shim_prime_rl_0_7_loss_is_the_mean_per_row(shim) -> None:
    """The unchanged v0.5-v0.7 rewrite is still a mean per row
    (needs torch; CI skips this).

    Runs the rewritten v0.7.0 compute_loss on CPU on two stack-packed rows
    with 2 and 4 trainable tokens: the loss must be the two row means summed,
    computed by hand, with a count of 2 rows.
    """
    torch = pytest.importorskip("torch")
    patched, _ = shim["_patch_sample_mean_loss"](
        _fixture("train_v0.7.0.txt"), "train.py", "0.7.0"
    )
    weight = torch.randn(7, 7, generator=torch.Generator().manual_seed(0))
    input_ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 1, 2, 3]])
    target_ids = torch.tensor([[2, 3, 4, 0, 0], [6, 1, 2, 3, 4]])
    loss_mask = torch.tensor(
        [[True, True, False, False, False], [False, True, True, True, True]]
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "config": types.SimpleNamespace(
            model=types.SimpleNamespace(lora=None, ac_offloading=None),
            loss_impl="torch",
        ),
        "cp_enabled": False,
        "forward": lambda model, ids, positions: {"logits": model[ids]},
        "model": weight,
        "ce_loss": torch.nn.CrossEntropyLoss(reduction="none"),
        "maybe_activation_offloading": lambda *args: contextlib.nullcontext(),
    }
    exec(textwrap.dedent(_compute_loss(patched)), namespace)

    loss_sum, sample_count = namespace["compute_loss"](
        {
            "input_ids": _OnCpu(input_ids),
            "position_ids": _OnCpu(torch.arange(5).repeat(2, 1)),
            "target_ids": _OnCpu(target_ids),
            "loss_mask": _OnCpu(loss_mask),
        }
    )

    logits = weight.double()[input_ids].tolist()
    expected = 0.0
    for row in range(2):
        tokens = [t for t in range(5) if loss_mask[row, t]]
        expected += sum(
            _hand_cross_entropy(logits[row][t], int(target_ids[row, t])) for t in tokens
        ) / len(tokens)
    assert int(sample_count) == 2
    assert math.isclose(float(loss_sum), expected, rel_tol=1e-5)


@pytest.mark.parametrize("head", ["fused", "vanilla"])
def test_sample_mean_shim_prime_rl_0_10_loss_is_the_mean_per_packed_sample(
    shim, head: str
) -> None:
    """Guards the arithmetic on prime-rl v0.10 (needs torch; CI skips this).

    Runs the rewritten v0.10.0 compute_loss on CPU with a stand-in model that
    follows prime-rl's LM-head contract: without a temperature both heads
    return the summed cross-entropy; with one, the fused head returns
    per-token logprobs and the vanilla head returns logits. One packed row
    holds three samples (seq_lens 4, 3 and 5; the last ends in padding) with
    2, 3 and 1 trainable tokens. The loss must be the three sample means
    summed, computed by hand; the count must be 3; and the gradient of
    loss / count must match a direct per-sample mean's.
    """
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    source = _fixture("train_v0.10.0.txt")
    patched, _ = shim["_patch_sample_mean_loss"](source, "train.py", "0.10.0")
    vocab = 7
    weight = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(0))
    weight.requires_grad_(True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 0, 0]])
    target_ids = torch.tensor([[2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 0]])
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]], dtype=torch.bool)
    samples = [[1, 2], [4, 5, 6], [9]]

    def forward(model, ids, positions, *, labels, temperature=None, **kwargs):
        logits = model[ids]
        out = {"logits": None, "logprobs": None, "entropy": None, "loss": None}
        if temperature is None:
            out["loss"] = functional.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
        elif head == "fused":
            logprobs = torch.log_softmax(logits / temperature.unsqueeze(-1), dim=-1)
            out["logprobs"] = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        else:
            out["logits"] = logits
        return out

    def build(train_source: str) -> Any:
        namespace: dict[str, Any] = {
            "torch": torch,
            "config": types.SimpleNamespace(
                model=types.SimpleNamespace(lora=None, ac_offloading=None)
            ),
            "cp_enabled": False,
            "gradient_manager": None,
            "forward": forward,
            "model": weight,
            "maybe_activation_offloading": lambda *args: contextlib.nullcontext(),
            "IGNORE_INDEX": -100,
        }
        exec(textwrap.dedent(_compute_loss(train_source)), namespace)
        return namespace["compute_loss"]

    batch = {
        "input_ids": _OnCpu(input_ids),
        "position_ids": _OnCpu(torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4]])),
        "target_ids": _OnCpu(target_ids),
        "loss_mask": _OnCpu(loss_mask),
        "seq_lens": _OnCpu(torch.tensor([4, 3, 5])),
    }

    loss_sum, sample_count = build(patched)(batch)

    logits = weight.detach().double()[input_ids[0]].tolist()
    per_token = {
        t: _hand_cross_entropy(logits[t], int(target_ids[0, t]))
        for sample in samples
        for t in sample
    }
    expected = sum(sum(per_token[t] for t in s) / len(s) for s in samples)
    assert int(sample_count) == 3
    assert math.isclose(float(loss_sum.detach()), expected, rel_tol=1e-5)

    native_sum, token_count = build(source)(batch)
    assert int(token_count) == 6
    assert math.isclose(
        float(native_sum.detach()), sum(per_token.values()), rel_tol=1e-5
    )

    (loss_sum / sample_count).backward()
    reference = weight.detach().clone().requires_grad_(True)
    torch.stack(
        [
            functional.cross_entropy(
                reference[input_ids[0, sample]], target_ids[0, sample]
            )
            for sample in samples
        ]
    ).mean().backward()
    torch.testing.assert_close(weight.grad, reference.grad)


def _write_sft_config(path: Path, *extra_lines: str) -> Path:
    path.write_text(
        "\n".join(["max_steps = 1", *extra_lines, "[data]", "batch_size = 8"]) + "\n",
        encoding="utf-8",
    )
    return path


def test_train_run_sft_sample_mean_launches_without_pack_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prime-rl v0.10 has no data.pack_function.

    Its config rejects the key, and it records each packed sample's length
    in seq_lens, so sample_mean must launch without a pack function and
    leave the shim to split the rows.
    """
    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    config = _write_sft_config(tmp_path / "sft.toml")
    work_dir = tmp_path / "train-run"
    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
            "--loss-normalization",
            "sample_mean",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--data.pack_function" not in captured["argv"]
    assert captured["env"]["BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS"] == "1"
    manifest = json.loads((work_dir / "train-run.json").read_text())
    assert (
        "prime-rl v0.10: each packed row is split into its examples at seq_lens"
        in manifest["extra"]["prime_rl_sft_shim"]["guards"]
    )


def test_train_run_sft_sample_mean_refuses_full_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With model.offload=full, prime-rl v0.10 scales the
    gradients by the step's token count before compute_loss runs, so the
    per-sample count the shim returns would be ignored."""
    config = _write_sft_config(tmp_path / "sft.toml", "[model]", 'offload = "full"')
    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")

    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--loss-normalization",
            "sample_mean",
        ],
    )

    assert result.exit_code == 1
    assert "sample_mean does not support model.offload=full" in result.output
