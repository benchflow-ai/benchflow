from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchflow.adapters.use_computer_cookbook import UseComputerCookbookAdapter


def _write_upstream_cuagym_smoke(root: Path) -> Path:
    task_dir = root / "datasets" / "cuagym" / "smoke__ubuntu-infra"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Take one screenshot, then stop.\n")
    (task_dir / "task.toml").write_text(
        """\
[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = ["cuagym", "ubuntu", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    setup = task_dir / "tests" / "setup"
    setup.mkdir(parents=True)
    (setup / "pre_command.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf 'setup-ok\\n' > /tmp/runner-cuagym-setup-ok\n"
    )
    return task_dir


def _write_raw_cuagym_task(
    root: Path,
    *,
    task_id: str = "d0641d59-1751-5d45-8ced-5e45d615a68c",
    app_type: str = "vscode",
    difficulty: str = "medium",
    reward_source: str = "import json, os, sqlite3\nprint('REWARD: 0.0')\n",
    setup_kind: str = "execute",
) -> Path:
    task_dir = root / "raw" / task_id
    task_dir.mkdir(parents=True)
    task_json = {
        "evaluator": {"type": "python", "url": "./reward.py"},
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": "./initial_setup.py",
                            "path": "/home/user/initial_setup.py",
                        }
                    ]
                },
            },
            {
                "type": setup_kind,
                "parameters": {"command": "python3 /home/user/initial_setup.py"},
            },
        ],
        "id": task_id,
        "difficulty": difficulty,
        "instruction": "Use Go to Definition to navigate to calculateTax.",
        "app_type": app_type,
    }
    (task_dir / "task.json").write_text(json.dumps(task_json) + "\n")
    (task_dir / "initial_setup.py").write_text("print('setup-ok')\n")
    (task_dir / "reward.py").write_text(reward_source)
    return task_dir


def test_import_upstream_cuagym_smoke_slice(tmp_path: Path) -> None:
    """Guards importing a selected public cookbook CUA-Gym smoke task."""
    upstream = tmp_path / "upstream"
    _write_upstream_cuagym_smoke(upstream)
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--upstream-repo",
            str(upstream),
            "--out-dir",
            str(out_dir),
            "--dataset",
            "cuagym",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "smoke__ubuntu-infra"
    assert str(imported) in result.stdout

    report = UseComputerCookbookAdapter.support_report(imported)
    assert report is not None
    assert report.supported is True
    assert report.dataset == "cuagym"


def test_import_raw_cuagym_python_reward_task(tmp_path: Path) -> None:
    """Guards importing a selected raw CUA-Gym Python-reward task."""
    raw_task = _write_raw_cuagym_task(tmp_path)
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "vscode__d0641d59-1751-5d45-8ced-5e45d615a68c"
    assert str(imported) in result.stdout
    assert (imported / "tests" / "cuagym" / "original" / "reward.py").is_file()
    assert (
        imported / "tests" / "setup" / "files" / "original" / "initial_setup.py"
    ).is_file()

    report = UseComputerCookbookAdapter.support_report(imported)
    assert report is not None
    assert report.supported is True
    assert report.dataset == "cuagym"
    assert report.task_id == "d0641d59-1751-5d45-8ced-5e45d615a68c"


def test_import_raw_cuagym_python_reward_task_with_launch_setup(
    tmp_path: Path,
) -> None:
    """Guards importing upstream CUA-Gym launch setup steps."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="da5fc6ff-ab19-5a98-85a4-b3fd2f169aea",
        app_type="pdf",
        setup_kind="launch",
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "pdf__da5fc6ff-ab19-5a98-85a4-b3fd2f169aea"
    report = UseComputerCookbookAdapter.support_report(imported)
    assert report is not None
    assert report.supported is True
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["setup_kinds"] == ["download", "launch"]


def test_import_raw_cuagym_python_reward_with_pypdf2_dependency(
    tmp_path: Path,
) -> None:
    """Guards the first non-stdlib CUA-Gym reward dependency mapping."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="548686c0-3701-587c-9cc1-1f3455046f22",
        app_type="pdf",
        difficulty="medium",
        reward_source=(
            "from pathlib import Path\n"
            "from PyPDF2 import PdfReader\n"
            "print('REWARD: 0.0')\n"
        ),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "pdf__548686c0-3701-587c-9cc1-1f3455046f22"
    report = UseComputerCookbookAdapter.support_report(imported)
    assert report is not None
    assert report.supported is True
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["PyPDF2", "pathlib"]
    assert cuagym_task["reward_dependencies"] == ["PyPDF2"]


def test_import_raw_cuagym_python_reward_with_stdlib_imports(
    tmp_path: Path,
) -> None:
    """Guards CUA-Gym reward scripts that import broader stdlib modules."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="5249154b-d943-570f-88ec-73e72ce872d1",
        app_type="pdf",
        difficulty="medium",
        reward_source=(
            "from __future__ import annotations\n"
            "import traceback, zipfile\n"
            "from difflib import SequenceMatcher\n"
            "from pathlib import Path\n"
            "from PyPDF2 import PdfReader\n"
            "print('REWARD: 0.0')\n"
        ),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "pdf__5249154b-d943-570f-88ec-73e72ce872d1"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == [
        "PyPDF2",
        "__future__",
        "difflib",
        "pathlib",
        "traceback",
        "zipfile",
    ]
    assert cuagym_task["reward_dependencies"] == ["PyPDF2"]


def test_import_raw_cuagym_python_reward_with_pillow_dependency(
    tmp_path: Path,
) -> None:
    """Guards the PIL import to Pillow package reward mapping."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="58fb62b1-3b5c-5ed9-b96d-2765c8d33363",
        app_type="multi_apps",
        difficulty="medium",
        reward_source=("import os\nfrom PIL import Image\nprint('REWARD: 0.0')\n"),
        setup_kind="launch",
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "multi_apps__58fb62b1-3b5c-5ed9-b96d-2765c8d33363"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["PIL", "os"]
    assert cuagym_task["reward_dependencies"] == ["Pillow"]


def test_import_raw_cuagym_python_reward_with_openpyxl_dependency(
    tmp_path: Path,
) -> None:
    """Guards the openpyxl import to package reward mapping."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="5987423f-0ce6-50ce-af44-a3df55cb55e3",
        app_type="multi_apps",
        difficulty="medium",
        reward_source=("import os\nimport openpyxl\nprint('REWARD: 0.0')\n"),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "multi_apps__5987423f-0ce6-50ce-af44-a3df55cb55e3"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["openpyxl", "os"]
    assert cuagym_task["reward_dependencies"] == ["openpyxl"]


def test_import_raw_cuagym_python_reward_with_gimpformats_dependency(
    tmp_path: Path,
) -> None:
    """Guards the gimpformats import to package reward mapping."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="52d58660-0f80-5434-9a98-fb1e192a177b",
        app_type="gimp",
        difficulty="medium",
        reward_source=(
            "import os\n"
            "from gimpformats.gimpXcfDocument import GimpDocument\n"
            "print('REWARD: 0.0')\n"
        ),
        setup_kind="launch",
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "gimp__52d58660-0f80-5434-9a98-fb1e192a177b"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["gimpformats", "os"]
    assert cuagym_task["reward_dependencies"] == ["gimpformats"]


def test_import_raw_cuagym_python_reward_with_docx_dependency(
    tmp_path: Path,
) -> None:
    """Guards the docx import to python-docx package reward mapping."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="565e7178-6ff1-5a73-a147-dd3bbfc48044",
        app_type="multi_apps",
        difficulty="medium",
        reward_source=("import os\nfrom docx import Document\nprint('REWARD: 0.0')\n"),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "multi_apps__565e7178-6ff1-5a73-a147-dd3bbfc48044"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["docx", "os"]
    assert cuagym_task["reward_dependencies"] == ["python-docx"]


def test_import_raw_cuagym_python_reward_with_numpy_pandas_dependencies(
    tmp_path: Path,
) -> None:
    """Guards the numpy/pandas reward import to package mappings."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="5c0b6ff0-8fda-5fdb-bd45-04701dc1773e",
        app_type="multi_apps",
        difficulty="medium",
        reward_source=(
            "import numpy as np\nimport os\nimport pandas as pd\nprint('REWARD: 0.0')\n"
        ),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "multi_apps__5c0b6ff0-8fda-5fdb-bd45-04701dc1773e"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["numpy", "os", "pandas"]
    assert cuagym_task["reward_dependencies"] == ["numpy", "pandas"]


def test_import_raw_cuagym_python_reward_with_odf_pptx_pyperclip_dependencies(
    tmp_path: Path,
) -> None:
    """Guards the odf/pptx/pyperclip reward import to package mappings."""
    raw_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="703a7289-8113-5a67-9c16-de316b8ed145",
        app_type="multi_apps",
        difficulty="medium",
        reward_source=(
            "import os\n"
            "import pyperclip\n"
            "from odf.opendocument import load\n"
            "from pptx import Presentation\n"
            "print('REWARD: 0.0')\n"
        ),
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-task-dir",
            str(raw_task),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    imported = out_dir / "multi_apps__703a7289-8113-5a67-9c16-de316b8ed145"
    inbound = UseComputerCookbookAdapter.from_task_dir(imported)
    assert inbound.compatibility is not None
    cuagym_task = inbound.compatibility.config_extra["cuagym_task"]
    assert cuagym_task["reward_imports"] == ["odf", "os", "pptx", "pyperclip"]
    assert cuagym_task["reward_dependencies"] == [
        "odfpy",
        "pyperclip",
        "python-pptx",
    ]


def test_import_raw_cuagym_dataset_supported_slice(tmp_path: Path) -> None:
    """Guards importing supported tasks from a raw CUA-Gym dataset root."""
    _write_raw_cuagym_task(
        tmp_path,
        task_id="00aa356d-b197-53b3-ac88-0157cfbe9320",
        app_type="vscode",
        difficulty="easy",
    )
    _write_raw_cuagym_task(
        tmp_path,
        task_id="0364cd36-e284-5a7e-929d-44177bb5d8e3",
        app_type="multi_apps",
        difficulty="medium",
    )
    _write_raw_cuagym_task(
        tmp_path,
        task_id="unsupported-third-party-import",
        app_type="vscode",
        difficulty="easy",
        reward_source="import requests\nprint('REWARD: 0.0')\n",
    )
    _write_raw_cuagym_task(
        tmp_path,
        task_id="unsupported-setup-launcher",
        app_type="vscode",
        difficulty="easy",
    )
    launcher_setup = (
        tmp_path / "raw" / "unsupported-setup-launcher" / "initial_setup.py"
    )
    launcher_setup.write_text(
        "import subprocess\nsubprocess.Popen(['code', '/home/user/project'])\n"
    )
    postconfig_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="unsupported-postconfig",
        app_type="vscode",
        difficulty="easy",
    )
    postconfig_json = json.loads((postconfig_task / "task.json").read_text())
    postconfig_json["evaluator"]["postconfig"] = [
        {"type": "execute", "parameters": {"command": ["python", "-c", "print('x')"]}}
    ]
    (postconfig_task / "task.json").write_text(json.dumps(postconfig_json) + "\n")
    out_dir = tmp_path / "tasks"
    support_report = tmp_path / "cuagym-support.json"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-tasks-root",
            str(tmp_path / "raw"),
            "--cuagym-app-type",
            "vscode",
            "--cuagym-difficulty",
            "easy",
            "--cuagym-limit",
            "2",
            "--out-dir",
            str(out_dir),
            "--support-report-out",
            str(support_report),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "CUA-Gym import summary:" in result.stdout
    assert "CUA-Gym skip reasons:" in result.stdout
    assert "unsupported reward imports" in result.stdout
    imported = out_dir / "vscode__00aa356d-b197-53b3-ac88-0157cfbe9320"
    assert str(imported) in result.stdout
    assert "unsupported-third-party-import" not in result.stdout
    assert "unsupported-setup-launcher" not in result.stdout
    assert "unsupported-postconfig" not in result.stdout
    assert not (out_dir / "multi_apps__0364cd36-e284-5a7e-929d-44177bb5d8e3").exists()

    report = UseComputerCookbookAdapter.support_report(imported)
    assert report is not None
    assert report.supported is True
    assert report.dataset == "cuagym"
    assert report.task_id == "00aa356d-b197-53b3-ac88-0157cfbe9320"

    support = json.loads(support_report.read_text())
    assert support["schema"] == "benchflow.cuagym-import-support-report.v1"
    assert support["counts"] == {
        "scanned": 5,
        "imported": 1,
        "supported_seen": 1,
        "unsupported": 3,
        "filtered": 1,
        "skipped": 4,
    }
    unsupported = {item["task_id"]: item for item in support["unsupported"]}
    assert unsupported["unsupported-third-party-import"]["code"] == (
        "unsupported-reward-import"
    )
    assert unsupported["unsupported-setup-launcher"]["code"] == (
        "unmapped-setup-launcher"
    )
    assert unsupported["unsupported-postconfig"]["code"] == (
        "unsupported-evaluator-postconfig"
    )
    assert support["filtered"][0]["task_id"] == ("0364cd36-e284-5a7e-929d-44177bb5d8e3")
    serialized = json.dumps(support)
    assert "Use Go to Definition" not in serialized
    assert "import requests" not in serialized


def test_import_raw_cuagym_dataset_skips_uncompilable_reward(
    tmp_path: Path,
) -> None:
    """Guards invalid reward.py files being reported unsupported before runtime."""
    bad_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="00bad000-b197-53b3-ac88-0157cfbe9320",
        reward_source=(
            "print('not a future import yet')\n"
            "from __future__ import annotations\n"
            "print('REWARD: 0.0')\n"
        ),
    )
    good_task = _write_raw_cuagym_task(
        tmp_path,
        task_id="99good00-b197-53b3-ac88-0157cfbe9320",
    )
    out_dir = tmp_path / "tasks"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/use-computer-cookbook-smoke/import_upstream.py",
            "--cuagym-tasks-root",
            str(tmp_path / "raw"),
            "--cuagym-limit",
            "1",
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "invalid reward.py python" in result.stdout
    assert not (out_dir / f"vscode__{bad_task.name}").exists()
    imported = out_dir / f"vscode__{good_task.name}"
    assert str(imported) in result.stdout
    assert imported.exists()
