"""Proof: snapshot → mutate → restore round-trip on a live Daytona sandbox.

Usage: env -u ANTHROPIC_API_KEY python proof_snapshot.py

Creates a sandbox, writes a known file, snapshots, mutates, restores,
then verifies the original content is back. Prints a pass/fail summary.
"""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harbor.models.task.task import Task

from benchflow._env_setup import _create_environment
from benchflow._snapshot import list_snapshots, restore, snapshot

TASK_PATH = Path(__file__).parent / "acp_smoke"


async def main() -> None:
    task = Task(TASK_PATH)
    env = _create_environment(
        environment_type="daytona",
        task=task,
        task_path=TASK_PATH,
        trial_name="snapshot-proof",
        trial_paths=None,
    )
    try:
        await env.start(force_build=False)
        workspace = "/app"

        # Step 1: write a known file
        await env.exec(f"echo 'original-content' > {workspace}/proof.txt")
        r1 = await env.exec(f"cat {workspace}/proof.txt")
        assert "original-content" in r1.stdout, f"write failed: {r1.stdout}"
        print(f"1. WROTE proof.txt: {r1.stdout.strip()!r}")

        # Step 2: snapshot
        ref = await snapshot(env, "checkpoint-1", workspace=workspace)
        print(f"2. SNAPSHOT created: {ref}")

        # Step 3: list snapshots
        snaps = await list_snapshots(env)
        print(f"3. LIST snapshots: {snaps}")
        assert "checkpoint-1" in snaps, f"snapshot not listed: {snaps}"

        # Step 4: mutate the workspace
        await env.exec(f"echo 'MUTATED' > {workspace}/proof.txt")
        await env.exec(f"echo 'extra-file' > {workspace}/extra.txt")
        r2 = await env.exec(f"cat {workspace}/proof.txt")
        assert "MUTATED" in r2.stdout
        print(f"4. MUTATED proof.txt: {r2.stdout.strip()!r}")

        # Step 5: restore
        await restore(env, ref, workspace=workspace)
        print(f"5. RESTORED from: {ref}")

        # Step 6: verify original content is back
        r3 = await env.exec(f"cat {workspace}/proof.txt")
        assert "original-content" in r3.stdout, f"restore failed: got {r3.stdout!r}"
        print(f"6. VERIFIED proof.txt: {r3.stdout.strip()!r}")

        # Step 7: verify mutated file is gone
        r4 = await env.exec(
            f"test -f {workspace}/extra.txt && echo exists || echo gone"
        )
        assert "gone" in r4.stdout, f"extra file survived restore: {r4.stdout}"
        print(f"7. VERIFIED extra.txt removed: {r4.stdout.strip()!r}")

        print("\n=== SNAPSHOT PROOF: PASS ===")
        print(f"ref: {ref}")
        print("Semantics: in-place restore via tar; workspace cleared then untarred.")

    finally:
        await env.stop(delete=True)


if __name__ == "__main__":
    asyncio.run(main())
