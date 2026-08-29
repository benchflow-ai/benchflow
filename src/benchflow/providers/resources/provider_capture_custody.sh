#!/bin/sh
set -eu

mode=$1
runtime_dir=$2
callback_log=$runtime_dir/callback.jsonl
capture_state=$runtime_dir/capture_state.json

case "$mode" in
  harden)
    test -d "$runtime_dir"
    test ! -L "$runtime_dir"
    test -f "$callback_log"
    test ! -L "$callback_log"
    test -f "$capture_state"
    test ! -L "$capture_state"
    chown 0:0 "$runtime_dir" "$callback_log" "$capture_state"
    chmod 700 "$runtime_dir"
    chmod 600 "$callback_log" "$capture_state"
    ;;
  probe)
    for artifact in "$callback_log" "$capture_state"; do
      if cat "$artifact" >/dev/null 2>&1; then exit 0; fi
      if [ -r "$artifact" ] || [ -w "$artifact" ]; then exit 0; fi
    done
    if [ -r "$runtime_dir" ] || [ -w "$runtime_dir" ] || [ -x "$runtime_dir" ]; then
      exit 0
    fi
    if command -v sudo >/dev/null 2>&1 && \
      sudo -n cat "$capture_state" >/dev/null 2>&1; then
      exit 0
    fi
    if command -v doas >/dev/null 2>&1 && \
      doas -n cat "$capture_state" >/dev/null 2>&1; then
      exit 0
    fi
    if id -G 2>/dev/null | tr ' ' '\n' | grep -qx 0; then exit 0; fi
    effective_caps=$(awk '/^CapEff:/ {print $2}' /proc/self/status 2>/dev/null || true)
    if [ -n "$effective_caps" ] && [ "$effective_caps" != 0000000000000000 ]; then
      exit 0
    fi
    for privilege_socket in \
      /var/run/docker.sock \
      /run/containerd/containerd.sock \
      /run/podman/podman.sock; do
      if [ -S "$privilege_socket" ] && \
        [ -r "$privilege_socket" ] && \
        [ -w "$privilege_socket" ]; then
        exit 0
      fi
    done
    exit 1
    ;;
  verify)
    test "$(stat -c '%u:%a' "$runtime_dir")" = 0:700
    test "$(stat -c '%u:%a' "$callback_log")" = 0:600
    test "$(stat -c '%u:%a' "$capture_state")" = 0:600
    ;;
  *)
    exit 2
    ;;
esac
