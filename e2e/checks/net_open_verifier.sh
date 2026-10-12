#!/bin/sh
# Run examples/tasks/net-open's and net-none's verifiers on what a sandbox
# reports with no network (Docker's `none`: no route, no address of its
# own), under a firewall `none`, open, and open on a host with no outside
# route. Each verifier may score 1 only for its own policy.
# Usage: net_open_verifier.sh <examples dir>
set -e
v=$(mktemp -d)
for t in net-open net-none; do
  sed -e "s#/app/net.txt#$v/net.txt#g" -e "s#/logs/verifier/reward.txt#$v/reward.txt#g" \
    "$1/tasks/$t/verifier/test.sh" > "$v/$t.sh"
done
score() { printf "$2" > "$v/net.txt"; bash "$v/$1.sh"; cat "$v/reward.txt"; }
check() { [ "$(score "$1" "$3")" = "$2" ] || { echo "$1 scored wrong on: $4"; exit 1; }; }
none_docker='reserved=unreachable\nown=no-address\nlocal=answered\n'
none_fw='reserved=refused\nown=answered\nlocal=answered\n'
open_net='reserved=no-answer\nown=answered\nlocal=answered\n'
open_noroute='reserved=unreachable\nown=answered\nlocal=answered\n'
check net-open 0 "$none_docker" "no network"
check net-open 0 "$none_fw" "a firewall none"
check net-open 1 "$open_net" "open"
check net-open 1 "$open_noroute" "open, host with no outside route"
check net-none 1 "$none_docker" "no network"
check net-none 1 "$none_fw" "a firewall none"
check net-none 0 "$open_net" "open"
check net-none 0 "$open_noroute" "open, host with no outside route"
# The answers each task.md asks for, read from its "Expected:" sentence,
# must score 1: what the prompt asks is what the verifier takes (review
# E7-E 1). net-none's must include the no-network answer with own=no-address.
here=$(dirname "$0")
for t in net-none net-open; do
  for answer in $(python3 "$here/expected_answers.py" "$1/tasks/$t/task.md"); do
    check "$t" 1 "$(printf %s "$answer" | tr ';' '\n')\n" "$t's task.md answer $answer"
  done
done
python3 "$here/expected_answers.py" "$1/tasks/net-none/task.md" | grep -qx 'reserved=unreachable;own=no-address;local=answered' \
  || { echo "net-none's task.md does not ask for own=no-address with no network"; exit 1; }
python3 "$here/expected_answers.py" "$1/tasks/net-open/task.md" | grep -qx 'reserved=unreachable;own=answered;local=answered' \
  || { echo "net-open's task.md does not allow reserved=unreachable on a host with no outside route"; exit 1; }
echo "net-open and net-none each score 1 only for their own policy, and for the answers their task.md asks for"
