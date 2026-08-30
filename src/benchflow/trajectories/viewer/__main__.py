"""``python -m benchflow.trajectories.viewer <rollout_dir_or_jsonl> [port]``"""

import sys

from .server import serve

if len(sys.argv) < 2:
    print(
        "Usage: python -m benchflow.trajectories.viewer <rollout_dir_or_jsonl> [port]"
    )
    sys.exit(1)
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
serve(sys.argv[1], port)
