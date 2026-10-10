# BenchFlow: run an agent and stream the JSON Lines file it records its steps in.
#
# usage: sh follow.sh 'PATTERN' COMMAND...
#
# Runs COMMAND, printing its stdout and stderr as they come. PATTERN is a
# glob (quoted, so the shell here expands it) for the JSON Lines file the
# agent appends to as it works; the newest match is followed. Every two
# seconds, and once more after the agent exits, each new complete line of it
# is printed as {"@bf":"session","item":<line>}. Exits with the agent's exit
# status. POSIX sh, coreutils or busybox.

pattern=$1
shift
dir="$HOME/.benchflow"
mkdir -p "$dir"
log="$dir/agent-output.txt"
: > "$log"
"$@" > "$log" 2>&1 < /dev/null &
pid=$!
trap 'kill "$pid" 2>/dev/null' TERM INT HUP
sent=0
file=""
lines=0

flush() {
  size=$(($(wc -c < "$log")))
  if [ "$size" -gt "$sent" ]; then
    tail -c +"$((sent + 1))" "$log" | head -c "$((size - sent))"
    sent=$size
  fi
  newest=$(ls -t $pattern 2> /dev/null | head -n 1)
  [ -n "$newest" ] || return 0
  if [ "$newest" != "$file" ]; then
    file=$newest
    lines=0
  fi
  n=$(($(wc -l < "$file")))
  if [ "$n" -gt "$lines" ]; then
    # Start on a line of its own when the agent's output ended mid-line.
    if [ "$sent" -gt 0 ] && [ -n "$(tail -c 1 "$log")" ]; then printf '\n'; fi
    sed -n "$((lines + 1)),${n}p" "$file" | sed -e 's/^/{"@bf":"session","item":/' -e 's/$/}/'
    lines=$n
  fi
}

while kill -0 "$pid" 2> /dev/null; do
  sleep 2
  flush
done
wait "$pid"
status=$?
flush
exit "$status"
