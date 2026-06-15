Create a complete BenchFlow benchmark task at `/app/my-task/` that tests whether an agent can write a Python function that reverses a string.

The task must include:
1. `task.toml` with version "1.0", medium difficulty, 300s agent timeout
2. `instruction.md` that tells the agent to create `/app/reverse.py` with a `reverse_string(s: str) -> str` function
3. `environment/Dockerfile` based on `ubuntu:24.04` with Python 3 installed, WORKDIR /app
4. `tests/test.sh` that verifies:
   - `/app/reverse.py` exists
   - `reverse_string("hello")` returns `"olleh"`
   - `reverse_string("")` returns `""`
   - Writes 1 to `/logs/verifier/reward.txt` if all pass, 0 otherwise

Make sure `test.sh` is executable (chmod +x).
