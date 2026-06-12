# Digest test fixtures

Golden-test fixtures for `bench tasks digest`. The expected digests are
hardcoded in `tests/test_task_digest.py` — do not modify any file under the
task directories here without updating those golden values.

Filenames are ASCII on purpose: macOS checks out unicode filenames in NFD,
which would change the hashed path bytes across platforms. Unicode coverage
lives in file *content* (`beta-task/data/unicode.txt`).
