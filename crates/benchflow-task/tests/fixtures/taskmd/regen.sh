#!/bin/bash
# Rebuild the task.md differential fixtures from a task-md checkout.
#
#   crates/benchflow-task/tests/fixtures/taskmd/regen.sh /path/to/task-md
#
# 1. examples/: every example package in the checkout (examples/<name>/, and the task packages a sequence or a dataset
#    holds, as <parent>--<name>).
# 2. imported/: the reference selftest's RewardKit fixtures (tools/fixtures/rewardkit/), imported with tools/convert.py
#    as the selftest imports them, and again with --convert-judges where that converts.
# 3. cases/: synthetic packages from make_cases.py.
# 4. golden/: what the reference checker prints for each (record.py), the reference scorer's results on a random
#    battery (make_battery.py), and SOURCE.json, the checkout it came from.
#
# Needs python3 (3.11 or later); PyYAML is optional.
set -euo pipefail

repo="$(cd "${1:?usage: regen.sh <task-md checkout>}" && pwd)"
here="$(cd "$(dirname "$0")" && pwd)"
reports="$(mktemp -d)"
trap 'rm -rf "$reports"' EXIT

rm -rf "$here/examples" "$here/imported" "$here/cases" "$here/golden"
mkdir -p "$here/examples" "$here/imported"

for d in "$repo"/examples/*/; do
  [ -f "$d/task.md" ] && cp -R "${d%/}" "$here/examples/$(basename "$d")"
done
for d in "$repo"/examples/*/tasks/*/ "$repo"/examples/*/*/tasks/*/; do
  [ -f "$d/task.md" ] || continue
  parent="$(basename "$(dirname "$(dirname "$d")")")"
  cp -R "${d%/}" "$here/examples/$parent--$(basename "$d")"
done

for d in "$repo"/tools/fixtures/rewardkit/*/; do
  name="$(basename "$d")"
  python3 "$repo/tools/convert.py" import "$d" "$here/imported/$name" --report "$reports/$name.json" >/dev/null 2>&1 \
    || rm -rf "$here/imported/$name"
  python3 "$repo/tools/convert.py" import "$d" "$here/imported/$name--judges" --convert-judges \
    --report "$reports/$name--judges.json" >/dev/null 2>&1 || rm -rf "$here/imported/$name--judges"
done

python3 "$here/make_cases.py" "$here/cases"
python3 "$here/record.py" "$repo" "$here"
python3 "$here/make_battery.py" "$repo" "$here/golden/scoring-battery.json"

head="$(sed -n 's/^ref: //p' "$repo/.git/HEAD" 2>/dev/null || true)"
commit="$( (cat "$repo/.git/$head" 2>/dev/null || grep " $head\$" "$repo/.git/packed-refs" 2>/dev/null | cut -d' ' -f1) | head -1)"
cat > "$here/golden/SOURCE.json" <<EOF
{
  "repository": "task.md",
  "commit": "${commit:-unknown}",
  "taskmd_py_sha256": "$(shasum -a 256 "$repo/tools/taskmd.py" | cut -d' ' -f1)",
  "rubrics_py_sha256": "$(shasum -a 256 "$repo/tools/rubrics.py" | cut -d' ' -f1)",
  "note": "examples/ are copies of the checkout's example packages; imported/ are tools/convert.py imports of tools/fixtures/rewardkit/; cases/ come from make_cases.py; golden/ is what tools/taskmd.py json and check print for each, run from its group's folder."
}
EOF
echo "fixtures rebuilt from $repo (${commit:-unknown})"
