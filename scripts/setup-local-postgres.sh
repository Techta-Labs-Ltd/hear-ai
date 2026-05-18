#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ROOT/.env"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[postgres] DATABASE_URL not set — skip"
  exit 0
fi

read -r PG_HOST PG_USER PG_PASS PG_DB PG_PORT < <(
  python3 <<'PY'
import os
from urllib.parse import urlparse, unquote

raw = os.environ.get("DATABASE_URL", "")
raw = raw.replace("postgresql+psycopg2://", "postgresql://", 1)
parsed = urlparse(raw)
host = parsed.hostname or ""
user = unquote(parsed.username or "")
password = unquote(parsed.password or "")
dbname = (parsed.path or "").lstrip("/")
port = parsed.port or 5432
print(host, user, password, dbname, port)
PY
)

LOCAL_HOSTS="127.0.0.1 localhost ::1"
is_local=0
for h in $LOCAL_HOSTS; do
  if [[ "$PG_HOST" == "$h" ]]; then
    is_local=1
    break
  fi
done

if [[ "${INSTALL_LOCAL_POSTGRES:-auto}" == "false" ]]; then
  echo "[postgres] INSTALL_LOCAL_POSTGRES=false — skip"
  exit 0
fi

if [[ "$is_local" -ne 1 && "${INSTALL_LOCAL_POSTGRES:-auto}" != "true" ]]; then
  echo "[postgres] DATABASE_URL host is '$PG_HOST' (remote) — skip local install"
  exit 0
fi

if [[ -z "$PG_USER" || -z "$PG_DB" ]]; then
  echo "[postgres] Could not parse user/database from DATABASE_URL" >&2
  exit 1
fi

echo "[postgres] Installing PostgreSQL server (local host $PG_HOST)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib postgresql-client

if command -v pg_ctlcluster >/dev/null 2>&1; then
  PG_VER="$(ls /etc/postgresql 2>/dev/null | head -n1)"
  if [[ -n "$PG_VER" ]]; then
    pg_ctlcluster "$PG_VER" main start 2>/dev/null || true
  fi
fi
service postgresql start 2>/dev/null || true

for _ in $(seq 1 30); do
  if sudo -u postgres psql -tAc "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! sudo -u postgres psql -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "[postgres] PostgreSQL service did not become ready" >&2
  exit 1
fi

echo "[postgres] Ensuring role '$PG_USER' and database '$PG_DB'..."
ESC_PASS="${PG_PASS//\'/\'\'}"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${PG_USER}') THEN
    CREATE ROLE ${PG_USER} LOGIN PASSWORD '${ESC_PASS}';
  ELSE
    ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';
  END IF;
END
\$\$;
SQL

DB_EXISTS="$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${PG_DB}'")"
if [[ "$DB_EXISTS" != "1" ]]; then
  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${PG_DB} OWNER ${PG_USER};"
else
  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "ALTER DATABASE ${PG_DB} OWNER TO ${PG_USER};"
fi

sudo -u postgres psql -v ON_ERROR_STOP=1 -d "$PG_DB" -c "GRANT ALL ON SCHEMA public TO ${PG_USER};" >/dev/null

echo "[postgres] Local PostgreSQL ready on port ${PG_PORT} (database ${PG_DB})"
