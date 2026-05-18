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

_pg_hba_conf() {
  find /etc/postgresql -name pg_hba.conf 2>/dev/null | head -n1
}

_as_postgres() {
  if [[ "$(id -u)" -eq 0 ]] && getent passwd postgres >/dev/null 2>&1; then
    if command -v runuser >/dev/null 2>&1; then
      runuser -u postgres -- "$@"
    else
      su -s /bin/sh postgres -c "$(printf '%q ' "$@")"
    fi
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u postgres "$@"
  else
    "$@"
  fi
}

_pg_admin_shell() {
  if [[ "$(id -u)" -eq 0 ]] && getent passwd postgres >/dev/null 2>&1; then
    if command -v runuser >/dev/null 2>&1; then
      runuser -u postgres -- /bin/sh
    else
      su -s /bin/sh postgres
    fi
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u postgres /bin/sh
  else
    /bin/sh
  fi
}

_pg_admin() {
  if [[ $# -eq 0 ]]; then
    _pg_admin_shell
  else
    _as_postgres psql -v ON_ERROR_STOP=1 "$@"
  fi
}

_pg_ready() {
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -q 2>/dev/null && return 0
    pg_isready -h 127.0.0.1 -p "$PG_PORT" -q 2>/dev/null && return 0
  fi
  _as_postgres psql -tAc "SELECT 1" >/dev/null 2>&1 && return 0
  _as_postgres psql -h 127.0.0.1 -p "$PG_PORT" -tAc "SELECT 1" >/dev/null 2>&1
}

_pg_reload() {
  local hba ver
  hba="$(_pg_hba_conf)"
  if [[ -n "$hba" ]] && command -v pg_ctlcluster >/dev/null 2>&1; then
    ver="$(basename "$(dirname "$(dirname "$hba")")")")"
    pg_ctlcluster "$ver" main reload 2>/dev/null || true
  fi
  service postgresql reload 2>/dev/null || true
}

_pg_ensure_tcp_auth() {
  local hba line
  hba="$(_pg_hba_conf)"
  [[ -n "$hba" && -f "$hba" ]] || return 0
  if grep -qE '^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+127\.0\.0\.1/32' "$hba"; then
    return 0
  fi
  line="host    all             all             127.0.0.1/32            scram-sha-256"
  echo "$line" >> "$hba"
  if ! grep -qE '^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+::1/128' "$hba"; then
    echo "host    all             all             ::1/128                 scram-sha-256" >> "$hba"
  fi
  _pg_reload
}

_pg_start() {
  if command -v pg_ctlcluster >/dev/null 2>&1; then
    local ver
    ver="$(ls /etc/postgresql 2>/dev/null | sort -rn | head -n1 || true)"
    if [[ -n "$ver" ]]; then
      pg_ctlcluster "$ver" main start 2>/dev/null || true
    fi
  fi
  service postgresql start 2>/dev/null || true
}

echo "[postgres] Installing PostgreSQL server (local host $PG_HOST)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib postgresql-client

_pg_start

ready=0
for _ in $(seq 1 45); do
  if _pg_ready; then
    ready=1
    break
  fi
  _pg_start
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "[postgres] PostgreSQL service did not become ready" >&2
  echo "[postgres] Debug: id postgres=$(id postgres 2>&1 || echo missing); sudo=$(command -v sudo || echo none)" >&2
  _as_postgres psql -tAc "SELECT 1" 2>&1 | tail -5 >&2 || true
  exit 1
fi

echo "[postgres] Ensuring role '$PG_USER' and database '$PG_DB'..."
ESC_PASS="${PG_PASS//\'/\'\'}"
_pg_admin_shell <<SQL
psql -v ON_ERROR_STOP=1 <<'EOS'
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${PG_USER}') THEN
    CREATE ROLE ${PG_USER} LOGIN PASSWORD '${ESC_PASS}';
  ELSE
    ALTER ROLE ${PG_USER} WITH LOGIN PASSWORD '${ESC_PASS}';
  END IF;
END
\$\$;
EOS
SQL

DB_EXISTS="$(_pg_admin -tAc "SELECT 1 FROM pg_database WHERE datname='${PG_DB}'" | tr -d '[:space:]')"
if [[ "$DB_EXISTS" != "1" ]]; then
  _pg_admin -c "CREATE DATABASE ${PG_DB} OWNER ${PG_USER};"
else
  _pg_admin -c "ALTER DATABASE ${PG_DB} OWNER TO ${PG_USER};"
fi

_pg_admin -d "$PG_DB" -c "GRANT ALL ON SCHEMA public TO ${PG_USER};" >/dev/null

_pg_ensure_tcp_auth

if ! PGPASSWORD="$PG_PASS" psql -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "[postgres] App user connection test failed (127.0.0.1:${PG_PORT}/${PG_DB})" >&2
  PGPASSWORD="$PG_PASS" psql -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SELECT 1" 2>&1 | tail -8 >&2 || true
  exit 1
fi

echo "[postgres] Local PostgreSQL ready on port ${PG_PORT} (database ${PG_DB}, user ${PG_USER})"
