#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/postgres-env.sh"

read -r PG_USER PG_PASS < <(
  python3 <<'PY'
import os
from urllib.parse import urlparse, unquote

raw = os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://", 1)
parsed = urlparse(raw)
print(unquote(parsed.username or ""), unquote(parsed.password or ""))
PY
)

if [[ -z "$PG_USER" || -z "$PG_PASS" ]]; then
  echo "[postgres] Could not parse user/password from DATABASE_URL" >&2
  exit 1
fi

ESC_PASS="${PG_PASS//\'/\'\'}"
if [[ "$(id -u)" -eq 0 ]] && getent passwd postgres >/dev/null 2>&1; then
  if command -v runuser >/dev/null 2>&1; then
    runuser -u postgres -- psql -v ON_ERROR_STOP=1 -c "ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';"
  else
    su -s /bin/sh postgres -c "psql -v ON_ERROR_STOP=1 -c \"ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';\""
  fi
elif command -v sudo >/dev/null 2>&1; then
  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';"
else
  psql -v ON_ERROR_STOP=1 -c "ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';"
fi

echo "[postgres] Password synced for role ${PG_USER} (from DATABASE_URL)"
