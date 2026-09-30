#!/usr/bin/env bash
# Build the radixark/miles patch that adds the BenchFlow example, from a Miles
# checkout (read only apart from the files the patch adds).
#
#   make_upstream_patch.sh MILES_CHECKOUT OUT.patch
#
# It copies upstream/ (a mirror of Miles's tree) into a scratch copy of the
# checkout, adds BenchFlow's row to docs/user-guide/environments.md, formats the
# new Python files the way Miles's pre-commit does (isort and black at 119
# columns, ruff check), checks that the result applies to the checkout's HEAD,
# and writes the diff. Needs git and uvx.
set -euo pipefail

miles=$(cd "$1" && pwd)
out=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

git -C "$miles" diff --quiet HEAD -- docs/user-guide/environments.md \
  || { echo "environments.md has local changes in $miles" >&2; exit 1; }
git clone -q --no-hardlinks --depth 1 "file://$miles" "$scratch/miles"
cd "$scratch/miles"
rsync -a --exclude ruff.toml "$here/upstream/" ./

# The integrations table is alphabetical: BenchFlow goes before Harbor.
python3 - <<'EOF'
from pathlib import Path
path = Path("docs/user-guide/environments.md")
text = path.read_text()
row = ("| [BenchFlow](https://github.com/benchflow-ai/benchflow) | agent function | "
       "[example](https://github.com/radixark/miles/tree/main/examples/experimental/benchflow) |\n")
anchor = "| [Harbor](https://github.com/harbor-framework/harbor) |"
if row not in text:
    assert anchor in text, "the integrations table changed; add the row by hand"
    text = text.replace(anchor, row + anchor, 1)
path.write_text(text)
EOF

files=$(git ls-files --others --exclude-standard -- '*.py')
uvx isort==5.13.2 --profile black --line-length 119 --filter-files $files >/dev/null
uvx black==24.3.0 --quiet --line-length 119 $files
uvx ruff@0.14.7 check --quiet $files

git add -A
git diff --cached --stat
git diff --cached > "$out"
git -C "$miles" apply --check "$out"
echo "wrote $out ($(wc -l < "$out") lines); it applies to $(git -C "$miles" rev-parse --short HEAD)"
