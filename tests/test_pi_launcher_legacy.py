"""Run also with Python 3.8/3.9's unittest: no BenchFlow install is needed."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PiLauncherCompatibility(unittest.TestCase):
    def test_launcher_reaches_exec_and_writes_provider_config(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "src/benchflow/agents/pi_acp_launcher.py"
        )
        for context_window in [128, "128", 128.0, None, True, -1, "invalid"]:
            with (
                self.subTest(context_window=context_window),
                tempfile.TemporaryDirectory() as home,
            ):
                env = dict(
                    os.environ,
                    HOME=home,
                    BENCHFLOW_PROVIDER_PROTOCOL="openai-completions",
                    BENCHFLOW_PROVIDER_BASE_URL="http://127.0.0.1:8999/v1",
                    BENCHFLOW_PROVIDER_MODEL="test-model",
                    BENCHFLOW_PROVIDER_API_KEY="dummy-marker",
                    BENCHFLOW_PROVIDER_MODELS=json.dumps(
                        [{"id": "test-model", "contextWindow": context_window}]
                    ),
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import os,runpy,sys; os.execv=lambda *args: print('EXEC-REACHED'); runpy.run_path(sys.argv[1],run_name='__main__')",
                        str(launcher),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "EXEC-REACHED")
                config = json.loads((Path(home) / ".pi/agent/models.json").read_text())
                # Provider keys use the neutral runtime prefix since f749dea7b.
                provider = config["providers"]["runtime-test-model"]
                self.assertGreater(provider["models"][0]["maxTokens"], 0)


if __name__ == "__main__":
    unittest.main()
