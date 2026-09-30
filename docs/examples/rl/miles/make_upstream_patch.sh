#!/usr/bin/env bash
# Build the radixark/miles patch that adds the BenchFlow example.
#
#   make_upstream_patch.sh --checkout MILES_CHECKOUT OUT.patch   # and check it applies
#   make_upstream_patch.sh --ref COMMIT OUT.patch                # no checkout: reads the one
#                                                                # file it edits through `gh api`
#
# The patch adds upstream/ (a mirror of Miles's tree: examples/experimental/benchflow
# and its tests) and BenchFlow's row in docs/user-guide/environments.md. The new
# Python files are formatted the way Miles's pre-commit does (isort and black at
# 119 columns) and must pass Miles's ruff rules. Nothing is cloned: a scratch git
# repository holds only the edited file and the new ones. Needs git and uvx
# (and gh for --ref).
set -euo pipefail

mode=$1 source=$2 out=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docs=docs/user-guide/environments.md
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

cd "$scratch"
git init -q .
mkdir -p "$(dirname "$docs")"
case "$mode" in
  --checkout) git -C "$source" show "HEAD:$docs" > "$docs"; base=$(git -C "$source" rev-parse --short HEAD) ;;
  --ref) gh api "repos/radixark/miles/contents/$docs?ref=$source" --jq .content | base64 -d > "$docs"; base=$source ;;
  *) echo "usage: $0 --checkout DIR|--ref COMMIT OUT.patch" >&2; exit 2 ;;
esac
git add -A && git -c user.name=patch -c user.email=patch@localhost commit -qm base

rsync -a --exclude ruff.toml "$here/upstream/" ./
# The integrations table is alphabetical: BenchFlow goes before Harbor.
python3 - "$docs" <<'EOF'
import sys
from pathlib import Path
path = Path(sys.argv[1])
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
uvx isort==5.13.2 --quiet --profile black --line-length 119 -p miles -p miles_plugins $files
uvx black==24.3.0 --quiet --line-length 119 $files
uvx ruff@0.14.7 check --quiet --line-length 320 --select E,F,B,UP --ignore E402,E501 $files

git add -A
git diff --cached --stat
git diff --cached > "$out"
if [ "$mode" = --checkout ]; then
  git -C "$source" apply --check "$out"
  echo "applies to $base"
fi
echo "wrote $out ($(wc -l < "$out") lines) against radixark/miles $base"
