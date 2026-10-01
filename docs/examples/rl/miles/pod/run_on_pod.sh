#!/usr/bin/env bash
# Runs on the machine that holds the keys: start a job.sh step on the pod, detached.
#
#   run_on_pod.sh [--daytona] <user@host[:port]> <job-name> <job.sh step> [args...]
#
# With --daytona the Daytona key is read from $DAYTONA_ENV_FILE (default
# ~/.config/benchflow/daytona.env), streamed over ssh's stdin and exported into the
# job's process tree only: never on a command line, never printed, never written to
# a file on the pod. Without it the job gets no key. The job's output goes to
# /work/logs/<job-name>.job.log on the pod and its exit code to /work/logs/<job-name>.rc;
# each step also writes its own log there (see job.sh).
set -euo pipefail
with_key=0
[ "${1:-}" = --daytona ] && { with_key=1; shift; }
target="$1"; name="$2"; shift 2
host="${target%:*}"; port=22; [ "$host" = "$target" ] || port="${target##*:}"
key_file="${DAYTONA_ENV_FILE:-$HOME/.config/benchflow/daytona.env}"
owner="${BENCHFLOW_DAYTONA_OWNER:-}"
ssh_opts=(-i "${POD_SSH_KEY:-$HOME/.ssh/prime_bf}" -o IdentitiesOnly=yes -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p "$port")

job=$(printf '%q ' bash /work/benchflow/docs/examples/rl/miles/pod/job.sh "$@")
remote="set -euo pipefail
if [ $with_key = 1 ]; then read -r DAYTONA_API_KEY; export DAYTONA_API_KEY; fi
export BENCHFLOW_DAYTONA_OWNER=$(printf '%q' "$owner")
mkdir -p /work/logs; rm -f /work/logs/$name.rc
setsid nohup bash -c $(printf '%q' "$job; echo \$? > /work/logs/$name.rc") > /work/logs/$name.job.log 2>&1 < /dev/null &
echo \"started $name (pid \$!)\""

if [ "$with_key" = 1 ]; then
  sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}DAYTONA_API_KEY=//p' "$key_file" | head -n 1 | tr -d "\"'" \
    | ssh "${ssh_opts[@]}" "$host" "bash -c $(printf '%q' "$remote")"
else
  ssh "${ssh_opts[@]}" "$host" "bash -c $(printf '%q' "$remote")" < /dev/null
fi
