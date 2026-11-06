#!/bin/sh
# Entry point of the trusted `simulator` service of an embodied task (benchflow.embodied.sidecar).
#
# Runs one episode server with the socket in the shared /rpc volume (the agent container mounts it read-only) and
# the episode record in /episode (simulator only). The task definition comes from $ROBO_TASK_MD (set in the
# task's docker-compose.yaml) or from a folder mounted at /task; $ROBO_EPISODE_FACTORY names the benchmark's
# `module:function` that builds the episode server. After the server exits the container stays up, so BenchFlow
# can run the verifier in it. Extra arguments go to `python -m benchflow.embodied.serve` (e.g. --seed N).
umask 000                     # the agent runs as another user; it must be able to connect to the socket
mkdir -p /rpc/ws /episode
rm -f /episode/serve.exit
TASK_DIR=/task
if [ -n "${ROBO_TASK_MD:-}" ]; then
    TASK_DIR=/run/embodied-task
    mkdir -p "$TASK_DIR"
    printf '%s\n' "$ROBO_TASK_MD" > "$TASK_DIR/task.md"
    unset ROBO_TASK_MD
fi
python -m benchflow.embodied.serve --factory "$ROBO_EPISODE_FACTORY" --task "$TASK_DIR" --run-dir /episode \
    --socket /rpc/episode.sock --workspace /rpc/ws "$@" > /episode/serve.log 2>&1
echo "$?" > /episode/serve.exit
exec sleep infinity
