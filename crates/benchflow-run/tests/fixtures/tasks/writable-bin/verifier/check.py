from pathlib import Path

ok = Path("/app/sum.txt").is_file() and Path("/app/sum.txt").read_text().strip() == "12"
Path("/logs/verifier/reward.txt").write_text("1\n" if ok else "0\n")
