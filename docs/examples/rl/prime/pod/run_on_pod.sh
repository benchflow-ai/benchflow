#!/usr/bin/env bash
# Runs on the machine that holds the keys (the VM): start a job script on a pod,
# detached, with the Daytona key in its environment and nowhere else.
#
#   run_on_pod.sh <user@host:port> <job-name> <script-on-pod> [args...]
#
# The key is read from $DAYTONA_ENV_FILE (default ~/.config/benchflow/daytona.env),
# streamed over ssh's stdin, and exported only into the job's process tree: it is
# never on a command line, never printed, and never written to a file on the pod.
# The job's output goes to $WORK/logs/<job-name>.log on the pod, its pid to .pid.
set -euo pipefail

target="$1"; name="$2"; shift 2
host="${target%:*}"; port="${target##*:}"
key_file="${DAYTONA_ENV_FILE:-$HOME/.config/benchflow/daytona.env}"
owner="${BENCHFLOW_DAYTONA_OWNER:-rl-prime}"
ssh_opts=(-i "${PRIME_SSH_KEY:-$HOME/.ssh/prime_bf}" -o IdentitiesOnly=yes -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/prime_bf_known_hosts"
  -o ServerAliveInterval=30 -p "$port")

quoted=$(printf '%q ' "$@")
remote="set -euo pipefail
read -r DAYTONA_API_KEY
export DAYTONA_API_KEY BENCHFLOW_DAYTONA_OWNER=$(printf '%q' "$owner")
WORK=\${WORK:-\$HOME/bf-rl}; mkdir -p \"\$WORK/logs\"; cd \"\$WORK\"
setsid nohup bash -c $(printf '%q' "$quoted") > \"\$WORK/logs/$name.log\" 2>&1 < /dev/null &
echo \$! > \"\$WORK/logs/$name.pid\"
echo \"started $name (pid \$!)\""

grep -m1 '^DAYTONA_API_KEY=' "$key_file" | cut -d= -f2- | tr -d "\"'" \
  | ssh "${ssh_opts[@]}" "$host" "bash -c $(printf '%q' "$remote")"
