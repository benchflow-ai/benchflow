import subprocess
from pathlib import Path

SCRIPT = Path("/app/check_status.sh")
STATUS = Path("/app/notes/status.txt")
EXPECTED_OUTPUT = "benchflow-terminal-smoke: ok\n"


def main() -> None:
    assert SCRIPT.exists(), f"{SCRIPT} does not exist"
    assert SCRIPT.is_file(), f"{SCRIPT} is not a file"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == EXPECTED_OUTPUT
    assert STATUS.read_text() == "ready\n"


if __name__ == "__main__":
    main()
