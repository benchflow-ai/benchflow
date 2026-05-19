#!/bin/bash
set -euo pipefail

cat > /app/check_status.sh <<'SH'
#!/bin/sh
printf 'benchflow-terminal-smoke: ok\n'
SH
chmod +x /app/check_status.sh

mkdir -p /app/notes
printf 'ready\n' > /app/notes/status.txt
