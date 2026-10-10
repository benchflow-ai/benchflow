#!/usr/bin/env bash
set -euo pipefail

output="${1:-${HARBOR_AGENT_OUTPUT:-/app/solution.md}}"
fail() { echo "objective failed: $*" >&2; exit 1; }
require() {
  local pattern="$1" label="${2:-$1}"
  grep -Eiq -- "$pattern" "$output" || fail "missing $label"
}

[[ -s "$output" ]] || fail "missing non-empty output file: $output"
require 'recipe explain' 'recipe explain command'
require 'scope' 'scope explanation'
require 'shared' 'shared scope'
require '(^|[^[:alnum:]_])user([^[:alnum:]_]|$)' 'user scope'
require 'machine-user|machine user' 'machine-user scope'
require '(^|[^[:alnum:]_])machine([^[:alnum:]_]|$)' 'machine scope'
require 'named location|named locations' 'named locations'
require 'managed|unmanaged' 'managed/unmanaged distinction'
require 'settings group|settings groups' 'settings groups'
require 'optional|not required|can be absent' 'groups are optional'
require 'read-only|metadata-only' 'read-only metadata boundary'
require 'redaction|sensitivity' 'redaction/sensitivity'
require 'lifecycle' 'lifecycle policy'
require 'native|export|import' 'native operation summary'
