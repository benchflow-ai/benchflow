# use-computer cookbook smoke

Tiny public smoke slice for the 0.7 universal environment adapter work.

The checked-in fixture mirrors the public
`datasets/osworld/ubuntu/smoke__ubuntu-osworld` task shape from the
use-computer cookbook:

- `instruction.md`
- `task.toml`
- `tests/osworld_task.json`

The BenchFlow adapter materializes the OSWorld setup/evaluator JSON into native
setup, verifier, oracle, and environment files. The parity runner compares a
direct Cua SDK execution with the BenchFlow Cua sandbox run.

To emit verifier evidence and close the adoption loop:

```bash
mkdir -p /tmp/benchflow-adapter-parity/use-computer-cookbook-smoke
BENCHFLOW_CUA_LOCAL=1 BENCHFLOW_CUA_LINUX_KIND=container \
  uv run python benchmarks/use-computer-cookbook-smoke/parity_test.py \
  --parity-out /tmp/benchflow-adapter-parity/use-computer-cookbook-smoke/parity_experiment.json
uv run bench agent verify use-computer-cookbook-smoke \
  --benchmarks-dir /tmp/benchflow-adapter-parity
```

The parity run also writes
`/tmp/benchflow-adapter-parity/use-computer-cookbook-smoke/adoption_report.json`,
a scrubbed review manifest with the sandbox, environment adapter, agent
adapter, benchmark adapter, parity counts, artifact counts, timing coverage,
and cleanup summary. It also writes `loop_state.json`, a resumable
adapter-adoption flight recorder with source, commands, artifacts, role status,
cleanup, unsupported-summary, and next queue items. Use
`parity_experiment.json` for the gate, `adoption_report.json` for human/CI
inspection, and `loop_state.json` when another agent or later session needs to
resume the loop.

The importer can copy selected upstream smoke task dirs into a temporary local
directory for dogfood:

```bash
uv run python benchmarks/use-computer-cookbook-smoke/import_upstream.py \
  --upstream-repo <path-to-use-computer-cookbook> \
  --out-dir <tmp>/tasks \
  --dataset cuagym \
  --overwrite
```

The current CUA-Gym support is intentionally narrow: BenchFlow accepts the
public `smoke__ubuntu-infra` setup-marker slice, preserves its setup hook, and
uses a BenchFlow-native verifier contract.

The importer can also copy one raw extracted CUA-Gym task directory into the
same smoke-task shape:

```bash
uv run python benchmarks/use-computer-cookbook-smoke/import_upstream.py \
  --cuagym-task-dir <raw-cuagym-dataset>/tasks/<task-id> \
  --out-dir <tmp>/tasks \
  --overwrite
```

It can also scan a raw extracted CUA-Gym `tasks/` root and import a bounded
supported slice:

```bash
uv run python benchmarks/use-computer-cookbook-smoke/import_upstream.py \
  --cuagym-tasks-root <raw-cuagym-dataset>/tasks \
  --cuagym-limit 1 \
  --out-dir <tmp>/tasks \
  --support-report-out <tmp>/cuagym-support-report.json \
  --overwrite
```

Use `--cuagym-app-type` and `--cuagym-difficulty` to narrow the scan. The limit
defaults to `1` for smoke safety; use `--cuagym-limit 0` only when intentionally
materializing every supported match. Dataset-root scans print top skip reasons
so the next missing app/runtime or reward mapping is visible. With
`--support-report-out`, the importer also writes a scrubbed per-task support
report containing task id, source path, app type, difficulty, status, reason,
and normalized issue code. It does not copy raw instructions, setup files,
`reward.py`, screenshots, or model output.

Raw CUA-Gym support is also intentionally narrow: tasks must avoid mock
placeholders, use only replayable setup kinds, and avoid unmapped setup app
launchers. Supported setup kinds include `download`, `command`, `execute`,
`sleep`, and `launch`; `launch` follows the upstream non-blocking
`subprocess.Popen` command semantics. Setup kind `open` remains unsupported
until there is a provider-honest app opener mapping. Supported reward imports
are Python's `sys.stdlib_module_names` set plus explicit package mappings,
currently `PIL -> Pillow` for image reward scripts, `PyPDF2 -> PyPDF2` for PDF
reward scripts, and `openpyxl -> openpyxl` for spreadsheet reward scripts. The
same explicit mapping includes `gimpformats -> gimpformats` for GIMP/XCF
reward scripts, `docx -> python-docx` for Word-document reward scripts,
`numpy -> numpy` for numeric/image reward scripts, and `pandas -> pandas` for
spreadsheet/table reward scripts. It also includes `odf -> odfpy` for ODS
reward scripts, `pyperclip -> pyperclip` for clipboard reward scripts, and
`pptx -> python-pptx` for PowerPoint reward scripts. The
one supported evaluator postconfig shape is the common
`pyautogui.hotkey("ctrl", "s")` save action, which BenchFlow replays through
Xlib on the Cua desktop display before running `reward.py`. Other postconfig
commands and unmapped reward dependencies still return structured
unsupported-task reports with diagnostic detail fields until their app/runtime
and reward mappings are provider-honest. Reward scripts must also compile as
Python before import; invalid original `reward.py` files are reported
unsupported rather than being treated as runtime reward-zero tasks.

Current local raw CUA-Gym scan coverage is `333 / 10910` tasks after the
dynamic stdlib import allowlist plus the `Pillow`, `PyPDF2`, `docx`,
`gimpformats`, `numpy`, `odf`, `openpyxl`, `pandas`, `pptx`, and `pyperclip`
dependency mappings plus the invalid-reward compile check. The main remaining
blockers are unmapped desktop app launchers, mock app types, setup kind `open`,
invalid original rewards, and one-off unmapped reward packages such as
`mutagen`, `requests`, `pypdf`, `yaml`, `pikepdf`, and `bs4`.
